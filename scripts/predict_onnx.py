#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不依赖 ROS，直接用 ONNX 模型对图像做离线推理，便于快速检查部署模型。

用法:
    python scripts/predict_onnx.py                      # 跑 data/sample 下的示例帧
    python scripts/predict_onnx.py path/to/a.jpg ...    # 指定图像
"""

import csv
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASS_NAMES = ["Forward", "Left", "Right", "Stop"]


def action_to_class(linear_x, angular_z):
    """与 train_yeahbot.py 中的映射保持一致"""
    if abs(linear_x) > 0.05:
        return 0
    if angular_z > 0.05:
        return 1
    if angular_z < -0.05:
        return 2
    return 3


def main():
    with open(os.path.join(ROOT, "models", "model_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    size = meta["img_size"]
    mean = np.array(meta["normalize_mean"], dtype=np.float32)
    std = np.array(meta["normalize_std"], dtype=np.float32)

    session = ort.InferenceSession(os.path.join(ROOT, "models", "yeahbot_model.onnx"))
    input_name = session.get_inputs()[0].name

    if len(sys.argv) > 1:
        items = [(path, None) for path in sys.argv[1:]]
    else:
        sample_dir = os.path.join(ROOT, "data", "sample")
        with open(os.path.join(sample_dir, "data.csv")) as f:
            items = [(os.path.join(sample_dir, row["image_path"]),
                      action_to_class(float(row["linear_x"]), float(row["angular_z"])))
                     for row in csv.DictReader(f)]

    correct = 0
    for path, label in items:
        image = Image.open(path).convert("RGB").resize((size, size))
        x = (np.asarray(image, dtype=np.float32) / 255.0 - mean) / std
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        logits = session.run(None, {input_name: x})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        pred = int(np.argmax(probs))
        lx, az = meta["class_to_cmd"][str(pred)]
        line = f"{os.path.basename(path)}: {CLASS_NAMES[pred]:<7} conf={probs[pred]:.2f} cmd=({lx:.3f}, {az:.3f})"
        if label is not None:
            correct += pred == label
            line += f"  label={CLASS_NAMES[label]}"
        print(line)

    if items and items[0][1] is not None:
        print(f"\n{correct}/{len(items)} 与示教标签一致（示例帧来自训练数据，仅用于检查模型可用）")


if __name__ == "__main__":
    main()
