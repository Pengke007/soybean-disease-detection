# -*- coding: utf-8 -*-
"""大豆病害识别 API 服务（Flask）。
提供：/api/predict 图片识别、/api/diseases 病害知识库、/api/health 健康检查、/ 网页前端。
"""
import os
import io
import json
import base64
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from diseases import CLASS_INFO, get_info

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
FRONTEND = BASE / "frontend"

app = Flask(__name__, static_folder=None)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# ---------- 加载模型 ----------
CLASS_NAMES = []
model = None
tf_eval = None


def load_model():
    global model, CLASS_NAMES, tf_eval
    idx_path = MODELS / "class_indices.json"
    weight_path = MODELS / "soybean_model.pth"
    if not (idx_path.exists() and weight_path.exists()):
        print("警告：模型文件不存在，服务将以【演示模式】运行（返回模拟结果）")
        CLASS_NAMES = ["Bacterial_blight", "Frogeye", "Healthy", "Soyabean_rust"]
        return False
    CLASS_NAMES = json.load(open(idx_path, encoding="utf-8"))
    from torchvision import models as tmodels
    import torch.nn as nn
    m = tmodels.mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(CLASS_NAMES))
    m.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    m.to(DEVICE).eval()
    model = m
    tf_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print(f"模型加载成功，类别: {CLASS_NAMES}")
    return True


HAS_MODEL = load_model()


def preprocess(image: Image.Image):
    return tf_eval(image.convert("RGB")).unsqueeze(0).to(DEVICE)


def predict_image(image: Image.Image):
    if HAS_MODEL:
        x = preprocess(image)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        order = probs.argsort()[::-1]
        results = []
        for i in order:
            cls = CLASS_NAMES[i]
            info = get_info(cls) or {}
            results.append({
                "class": cls,
                "name_zh": info.get("name_zh", cls),
                "confidence": round(float(probs[i]) * 100, 2),
                "is_disease": info.get("is_disease", True),
            })
        return {"mode": "model", "top": results[0], "candidates": results[:4]}
    else:
        # 演示模式：基于图像主色调的粗略推测，仅用于界面联调演示
        import random
        small = image.convert("RGB").resize((64, 64))
        px = list(small.getdata())
        r = sum(p[0] for p in px) / len(px)
        g = sum(p[1] for p in px) / len(px)
        b = sum(p[2] for p in px) / len(px)
        if r > g * 1.25 and r > b * 1.3:
            guess = "Soyabean_rust"
        elif g > r * 1.15:
            guess = "Healthy"
        else:
            guess = "Bacterial_blight"
        info = get_info(guess) or {}
        return {
            "mode": "demo",
            "top": {"class": guess, "name_zh": info.get("name_zh", guess),
                    "confidence": 72.0, "is_disease": info.get("is_disease", True)},
            "candidates": [],
            "note": "当前为演示模式（未加载训练模型），结果仅供界面联调，不代表真实诊断。"
        }


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": HAS_MODEL, "device": str(DEVICE),
                    "classes": CLASS_NAMES})


@app.route("/api/diseases")
def diseases():
    return jsonify(CLASS_INFO)


def build_response(result):
    top_cls = result["top"]["class"]
    info = get_info(top_cls) or {}
    result["detail"] = {
        "name_zh": info.get("name_zh", top_cls),
        "name_en": info.get("name_en", top_cls),
        "pathogen": info.get("pathogen", ""),
        "type": info.get("type", ""),
        "symptoms": info.get("symptoms", ""),
        "confusion": info.get("confusion", ""),
        "control": info.get("control", []),
        "color": info.get("color", "#2e9e5b"),
        "emoji": info.get("emoji", "🌿"),
    }
    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def predict():
    """兼容两种调用：JSON {image: base64}（网页）与 multipart 文件上传（小程序）。"""
    try:
        if request.content_type and "multipart/form-data" in request.content_type:
            f = request.files.get("image")
            if f is None:
                return jsonify({"error": "缺少 image 文件"}), 400
            image = Image.open(io.BytesIO(f.read())).convert("RGB")
        else:
            data = request.get_json(force=True, silent=True) or {}
            img_b64 = data.get("image", "")
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            if not img_b64:
                return jsonify({"error": "缺少 image 参数"}), 400
            raw = base64.b64decode(img_b64)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        result = predict_image(image)
        return build_response(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/base64", methods=["POST"])
def predict_base64():
    try:
        data = request.get_json(force=True, silent=True) or {}
        img_b64 = data.get("image", "")
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        if not img_b64:
            return jsonify({"error": "缺少 image 参数"}), 400
        image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        return build_response(predict_image(image))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
@app.route("/<path:path>")
def frontend(path=""):
    if path == "" or path.endswith("/"):
        path = "index.html"
    return send_from_directory(str(FRONTEND), path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
