# -*- coding: utf-8 -*-
"""大豆病害识别模型训练脚本（迁移学习 MobileNetV3-Small，CPU 友好）。
输出：
  models/soybean_model.pth      模型权重
  models/class_indices.json     类别索引
  models/training_report.json   训练报告与指标
"""
import os
import json
import time
import copy
import random
from pathlib import Path

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

# 纯 numpy 实现的评估指标，避免额外依赖 sklearn
def accuracy_score(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def f1_score(y_true, y_pred, average="weighted"):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s, supports = [], []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1); supports.append(np.sum(y_true == c))
    if average == "weighted":
        return float(np.average(f1s, weights=supports))
    return float(np.mean(f1s))


def confusion_matrix(y_true, y_pred, labels=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    classes = np.array(labels) if labels is not None else np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def classification_report(y_true, y_pred, target_names=None, digits=3):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    names = target_names or [str(i) for i in np.unique(y_true)]
    lines = ["              precision    recall  f1-score   support", ""]
    for i, name in enumerate(names):
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        lines.append(f"{name:16s} {prec:9.{digits}f} {rec:8.{digits}f} {f1:8.{digits}f} {np.sum(y_true==i):7d}")
    lines.append("")
    lines.append(f"accuracy{'':9s} {accuracy_score(y_true, y_pred):{digits}.{digits}f}")
    return "\n".join(lines)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# 同时输出到控制台与日志文件，便于后台任务排查进度
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
sys.stdout = Tee(sys.stdout, open(MODELS / "train.log", "w", encoding="utf-8"))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 12
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = json.load(open(BASE / "data" / "classes.json", encoding="utf-8"))
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class SoyDataset(Dataset):
    def __init__(self, split, transform):
        self.samples = []
        self.transform = transform
        d = DATA / split
        for cls in CLASS_NAMES:
            cd = d / cls
            if cd.exists():
                for p in cd.glob("*.jpg"):
                    self.samples.append((p, CLASS_TO_IDX[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        img = Image.open(p).convert("RGB")
        return self.transform(img), y


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_model(n_classes):
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    except Exception as e:
        print("预训练权重加载失败，改用随机初始化:", e)
        model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, phase):
    running_loss = 0.0
    all_preds, all_labels = [], []
    is_train = phase == "train"
    model.train(is_train)
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return running_loss / len(loader.dataset), acc, all_preds, all_labels


def main():
    print(f"设备: {DEVICE} | 类别: {CLASS_NAMES}")
    train_tf, eval_tf = build_transforms()
    train_ds = SoyDataset("train", train_tf)
    val_ds = SoyDataset("val", eval_tf)
    test_ds = SoyDataset("test", eval_tf)
    print(f"样本数 train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    if len(train_ds) == 0:
        raise RuntimeError("训练集为空，请先运行 prepare_data.py")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(len(CLASS_NAMES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, "train")
        va_loss, va_acc, _, _ = run_epoch(model, val_loader, criterion, optimizer, "val")
        scheduler.step()
        history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        flag = ""
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())
            flag = " *"
        print(f"Epoch {ep:02d}/{EPOCHS} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.4f}{flag}")

    model.load_state_dict(best_state)
    te_loss, te_acc, preds, labels = run_epoch(model, test_loader, criterion, optimizer, "test")
    f1 = f1_score(labels, preds, average="weighted")
    print(f"\n测试集: acc={te_acc:.4f} f1={f1:.4f} | 耗时 {time.time()-t0:.0f}s")
    print("\n分类报告:\n", classification_report(labels, preds, target_names=CLASS_NAMES, digits=3))
    print("混淆矩阵:\n", confusion_matrix(labels, preds))

    torch.save(best_state, MODELS / "soybean_model.pth")
    json.dump(CLASS_NAMES, open(MODELS / "class_indices.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    report = {
        "model": "MobileNetV3-Small (ImageNet 预训练迁移学习)",
        "classes": CLASS_NAMES,
        "epochs": EPOCHS,
        "best_val_acc": best_acc,
        "test_acc": te_acc,
        "test_f1_weighted": f1,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "history": history,
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "elapsed_seconds": time.time() - t0,
    }
    json.dump(report, open(MODELS / "training_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n模型与报告已保存至 models/")


if __name__ == "__main__":
    main()
