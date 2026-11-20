# -*- coding: utf-8 -*-
"""将 HF parquet 数据集导出为图片文件夹，并按类别划分 train/val/test。
仅依赖 pyarrow + pillow，无需 pandas。
数据集来源：anandvermagmailcom/soybean-leaf-diseases (CC BY 4.0)
类别：Bacterial_blight / Frogeye / Healthy / Soyabean_rust
"""
import json
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"

LABEL_MAP = {
    0: "Bacterial_blight",
    1: "Frogeye",
    2: "Healthy",
    3: "Soyabean_rust",
}
SPLITS = {"train": "train.parquet", "val": "val.parquet", "test": "test.parquet"}


def get_image(obj):
    """parquet 中 image 列可能是 struct{bytes,path} 或直接二进制。"""
    if isinstance(obj, dict):
        b = obj.get("bytes")
    elif isinstance(obj, (bytes, bytearray)):
        b = obj
    else:
        b = None
    if not b:
        return None
    try:
        return Image.open(BytesIO(b)).convert("RGB")
    except Exception:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {c: {} for c in LABEL_MAP.values()}

    for split, fname in SPLITS.items():
        p = RAW / fname
        if not p.exists():
            raise FileNotFoundError(f"缺少数据文件: {p}")
        table = pq.read_table(p)
        print(f"[{split}] {table.num_rows} 行, 列: {table.column_names}")
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()

        for img_obj, label in zip(images, labels):
            pil = get_image(img_obj)
            if pil is None:
                continue
            cls = LABEL_MAP.get(int(label))
            if cls is None:
                continue
            target_dir = OUT / split / cls
            target_dir.mkdir(parents=True, exist_ok=True)
            idx = counts[cls].get(split, 0)
            pil.save(target_dir / f"{cls}_{idx:04d}.jpg", quality=92)
            counts[cls][split] = idx + 1

    print("\n=== 类别分布 ===")
    for cls, d in counts.items():
        print(f"{cls:18s} train={d.get('train',0):4d}  val={d.get('val',0):4d}  test={d.get('test',0):4d}")

    classes = list(LABEL_MAP.values())
    with open(BASE / "data" / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print("\n类别顺序已写入 data/classes.json:", classes)


if __name__ == "__main__":
    main()
