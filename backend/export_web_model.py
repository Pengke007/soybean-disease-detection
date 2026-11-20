# -*- coding: utf-8 -*-
"""导出浏览器可运行资源：ONNX 模型 + 病害知识库 JSON。

产物放在 web/ 目录，供纯静态网页（GitHub Pages）直接加载。
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
WEB = BASE / "web"
WEB.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diseases import CLASS_INFO, MODEL_CLASSES  # noqa: E402

CLASS_NAMES = json.load(open(BASE / "data" / "classes.json", encoding="utf-8"))
N = len(CLASS_NAMES)


def build_model():
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    except Exception:
        model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, N)
    return model


def main():
    model = build_model()
    state = torch.load(MODELS / "soybean_model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = WEB / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    print("ONNX 导出:", onnx_path, round(onnx_path.stat().st_size / 1024 / 1024, 2), "MB")

    # 验证导出模型与原始模型输出一致
    import onnxruntime as ort
    import numpy as np

    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()
    onnx_out = ort_session.run(None, {"input": x})[0]
    diff = float(np.max(np.abs(torch_out - onnx_out)))
    print("PyTorch vs ONNX 最大差异:", diff)
    assert diff < 1e-3, "导出结果不一致"

    # 知识库 JSON（网页直接用）
    info = {}
    for cls in CLASS_NAMES:
        d = CLASS_INFO.get(cls, {})
        info[cls] = {
            "name_zh": d.get("name_zh", cls),
            "name_en": d.get("name_en", cls),
            "pathogen": d.get("pathogen", ""),
            "type": d.get("type", ""),
            "symptoms": d.get("symptoms", ""),
            "confusion": d.get("confusion", ""),
            "control": d.get("control", []),
            "color": d.get("color", "#2e9e5b"),
            "emoji": d.get("emoji", "🌿"),
            "model_index": CLASS_NAMES.index(cls),
        }
    # 科普但同时给出可展示的其它病害
    for cls, d in CLASS_INFO.items():
        if cls not in info:
            info[cls] = {
                "name_zh": d.get("name_zh", cls),
                "name_en": d.get("name_en", cls),
                "pathogen": d.get("pathogen", ""),
                "type": d.get("type", ""),
                "symptoms": d.get("symptoms", ""),
                "confusion": d.get("confusion", ""),
                "control": d.get("control", []),
                "color": d.get("color", "#2e9e5b"),
                "emoji": d.get("emoji", "🌿"),
                "model_index": d.get("model_index", None),
            }

    payload = {
        "classes": CLASS_NAMES,
        "info": info,
        "model_index": {c: i for i, c in enumerate(CLASS_NAMES)},
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "test_acc": json.load(open(MODELS / "training_report.json", encoding="utf-8"))["test_acc"],
    }
    json.dump(payload, open(WEB / "knowledge.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("知识库导出:", WEB / "knowledge.json")


if __name__ == "__main__":
    main()
