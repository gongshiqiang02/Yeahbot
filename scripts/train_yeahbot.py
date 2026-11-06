#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  YeahBot 小车 — 行为克隆(Behavioral Cloning)训练脚本
=============================================================================
  模型架构: MobileNetV2 (轻量级, 适合树莓派部署)
  任务类型: 4-类分类 (前进 / 左转 / 右转 / 停止)
  
  三大核心行为:
    1. 障碍物绕行: 遇到障碍物 → 转向绕过 → 回正直线
    2. 垃圾桶掉头: 识别垃圾桶 → 原地掉头(持续转向)
    3. 红色篮子停车: 识别红色篮子 → 小车停止
  
  输出: ONNX 格式模型, 可部署至树莓派 + ROS1
=============================================================================
"""

import os
import sys
import csv
import time
import copy
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

# ─────────────────────────────────────────────────────────────
# 0. 日志 & 参数解析
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("yeahbot_train")


def parse_args():
    p = argparse.ArgumentParser(description="YeahBot 行为克隆训练")
    p.add_argument("--data_dir",   type=str, default="./data/2026_0210",
                    help="数据根目录, 包含 data.csv 和 images/ 子目录")
    p.add_argument("--csv_name",   type=str, default="data.csv",
                    help="CSV 文件名")
    p.add_argument("--output_dir", type=str, default="./output",
                    help="模型输出目录")
    p.add_argument("--epochs",     type=int,   default=500,
                    help="训练轮数")
    p.add_argument("--batch_size", type=int,   default=32,
                    help="批大小")
    p.add_argument("--lr",         type=float, default=1e-3,
                    help="初始学习率")
    p.add_argument("--img_size",   type=int,   default=224,
                    help="输入图像尺寸 (MobileNetV2 默认 224)")
    p.add_argument("--num_workers",type=int,   default=4,
                    help="DataLoader 工作线程数")
    p.add_argument("--save_every", type=int,   default=100,
                    help="每隔多少轮保存一次检查点")
    p.add_argument("--val_split",  type=float, default=0.15,
                    help="验证集比例")
    p.add_argument("--pretrained", action="store_true", default=True,
                    help="使用 ImageNet 预训练权重")
    p.add_argument("--freeze_backbone_epochs", type=int, default=10,
                    help="前 N 轮冻结主干网络只训练分类头")
    p.add_argument("--device",     type=str, default="auto",
                    help="训练设备 (auto / cpu / cuda)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# 1. 动作映射 — 将连续值映射为离散类别
# ─────────────────────────────────────────────────────────────
# 根据数据分析, 共有 4 种离散动作:
#   类别 0: 前进 (linear_x ≈ 0.113, angular_z = 0)
#   类别 1: 左转 (linear_x = 0,     angular_z ≈ +0.206)  — 绕障/掉头
#   类别 2: 右转 (linear_x = 0,     angular_z ≈ -0.206)  — 绕障/掉头
#   类别 3: 停止 (linear_x = 0,     angular_z = 0)       — 红色篮子/待命

CLASS_NAMES = ["前进(Forward)", "左转(Left)", "右转(Right)", "停止(Stop)"]
NUM_CLASSES = len(CLASS_NAMES)

# 每个类别对应的 (linear_x, angular_z) 指令, 部署时使用
CLASS_TO_CMD = {
    0: (0.113240122652057, 0.0),
    1: (0.0,  0.205891132094649),
    2: (0.0, -0.205891132094649),
    3: (0.0,  0.0),
}


def action_to_class(linear_x: float, angular_z: float) -> int:
    """将 (linear_x, angular_z) 映射为类别标签"""
    lx, az = float(linear_x), float(angular_z)
    if abs(lx) > 0.05:          # 有前进速度 → 直行
        return 0
    elif az > 0.05:             # 正角速度 → 左转
        return 1
    elif az < -0.05:            # 负角速度 → 右转
        return 2
    else:                       # 都为0 → 停止
        return 3


# ─────────────────────────────────────────────────────────────
# 2. 数据集定义
# ─────────────────────────────────────────────────────────────
class YeahBotDataset(Dataset):
    """
    从 CSV + images/ 加载数据
    CSV 格式: timestamp, image_path, linear_x, angular_z
    """

    def __init__(self, samples: list, data_dir: str, transform=None):
        """
        Args:
            samples: [(image_path, class_label), ...]
            data_dir: 数据根目录
            transform: torchvision 图像变换
        """
        self.samples = samples
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel_path, label = self.samples[idx]
        img_path = os.path.join(self.data_dir, img_rel_path)

        # 加载图像
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label


def load_csv(csv_path: str) -> list:
    """读取 CSV, 返回 [(image_path, class_label), ...]"""
    samples = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row["image_path"]
            lx = float(row["linear_x"])
            az = float(row["angular_z"])
            label = action_to_class(lx, az)
            samples.append((img_path, label))
    return samples


# ─────────────────────────────────────────────────────────────
# 3. 数据增强 & 预处理
# ─────────────────────────────────────────────────────────────
def get_transforms(img_size: int):
    """返回训练/验证的图像变换"""
    # ImageNet 归一化参数
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),  # 轻微旋转平移
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # 随机遮挡模拟遮挡
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ─────────────────────────────────────────────────────────────
# 4. 模型定义 — MobileNetV2 + 自定义分类头
# ─────────────────────────────────────────────────────────────
class YeahBotModel(nn.Module):
    """
    基于 MobileNetV2 的行为克隆模型
    
    选择 MobileNetV2 的理由:
      ✓ 极其轻量 (~3.4M 参数), 适合树莓派实时推理
      ✓ 使用倒残差结构(Inverted Residuals), 计算效率高
      ✓ ImageNet 预训练, 迁移学习收敛快
      ✓ ONNX 导出兼容性好, onnxruntime 推理顺畅
      ✓ 在嵌入式设备上有大量成功部署案例
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        # 加载 MobileNetV2 骨干
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # 提取特征提取层 (去掉最后的分类器)
        self.features = backbone.features          # 输出 [B, 1280, 7, 7]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # → [B, 1280, 1, 1]

        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)       # 特征提取
        x = self.pool(x)           # 全局平均池化
        x = torch.flatten(x, 1)    # 展平
        x = self.classifier(x)     # 分类
        return x

    def freeze_backbone(self):
        """冻结骨干网络 (只训练分类头)"""
        for param in self.features.parameters():
            param.requires_grad = False
        logger.info("🔒 骨干网络已冻结, 仅训练分类头")

    def unfreeze_backbone(self):
        """解冻骨干网络 (全网络微调)"""
        for param in self.features.parameters():
            param.requires_grad = True
        logger.info("🔓 骨干网络已解冻, 全网络微调")


# ─────────────────────────────────────────────────────────────
# 5. 训练 & 验证循环
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    # 每类统计
    class_correct = [0] * NUM_CLASSES
    class_total   = [0] * NUM_CLASSES

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for i in range(labels.size(0)):
            lbl = labels[i].item()
            class_total[lbl] += 1
            if predicted[i] == lbl:
                class_correct[lbl] += 1

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    class_acc  = {
        CLASS_NAMES[i]: (class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0)
        for i in range(NUM_CLASSES)
    }
    return epoch_loss, epoch_acc, class_acc


# ─────────────────────────────────────────────────────────────
# 6. 模型保存 & ONNX 导出
# ─────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, output_dir, tag=""):
    """保存训练检查点"""
    ckpt_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}{tag}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_acc": best_acc,
    }, ckpt_path)
    logger.info(f"💾 检查点已保存: {ckpt_path}")
    return ckpt_path


def export_onnx(model, img_size, output_path, device):
    """导出为 ONNX 格式"""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,           # opset 12 兼容性好
        do_constant_folding=True,   # 常量折叠优化
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={              # 支持动态 batch
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    logger.info(f"📦 ONNX 模型已导出: {output_path}")

    # 打印模型大小
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"   模型大小: {size_mb:.2f} MB")


# ─────────────────────────────────────────────────────────────
# 7. 训练日志写入 CSV
# ─────────────────────────────────────────────────────────────
def init_log_csv(log_path):
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc",
            "acc_forward", "acc_left", "acc_right", "acc_stop",
            "lr"
        ])


def append_log_csv(log_path, epoch, tl, ta, vl, va, class_acc, lr):
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, f"{tl:.6f}", f"{ta:.4f}",
            f"{vl:.6f}", f"{va:.4f}",
            f"{class_acc[CLASS_NAMES[0]]:.2f}",
            f"{class_acc[CLASS_NAMES[1]]:.2f}",
            f"{class_acc[CLASS_NAMES[2]]:.2f}",
            f"{class_acc[CLASS_NAMES[3]]:.2f}",
            f"{lr:.8f}"
        ])


# ─────────────────────────────────────────────────────────────
# 8. 主流程
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── 设备 ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"🖥️  训练设备: {device}")

    # ── 输出目录 ──
    os.makedirs(args.output_dir, exist_ok=True)
    log_csv_path = os.path.join(args.output_dir, "training_log.csv")
    init_log_csv(log_csv_path)

    # ── 加载数据 ──
    csv_path = os.path.join(args.data_dir, args.csv_name)
    all_samples = load_csv(csv_path)
    logger.info(f"📊 共加载 {len(all_samples)} 条样本")

    # 统计类别分布
    class_counts = [0] * NUM_CLASSES
    for _, lbl in all_samples:
        class_counts[lbl] += 1
    for i in range(NUM_CLASSES):
        logger.info(f"   {CLASS_NAMES[i]}: {class_counts[i]} 样本 ({class_counts[i]/len(all_samples)*100:.1f}%)")

    # ── 划分训练/验证集 (按时间顺序, 保持序列结构) ──
    np.random.seed(42)
    indices = np.arange(len(all_samples))
    np.random.shuffle(indices)
    val_size = int(len(all_samples) * args.val_split)
    val_indices  = set(indices[:val_size])
    train_samples = [all_samples[i] for i in range(len(all_samples)) if i not in val_indices]
    val_samples   = [all_samples[i] for i in range(len(all_samples)) if i in val_indices]
    logger.info(f"   训练集: {len(train_samples)}, 验证集: {len(val_samples)}")

    # ── 数据增强 ──
    train_tf, val_tf = get_transforms(args.img_size)

    train_dataset = YeahBotDataset(train_samples, args.data_dir, transform=train_tf)
    val_dataset   = YeahBotDataset(val_samples,   args.data_dir, transform=val_tf)

    # ── 加权采样器 (解决类别不平衡) ──
    train_labels = [s[1] for s in train_samples]
    train_class_counts = [0] * NUM_CLASSES
    for lbl in train_labels:
        train_class_counts[lbl] += 1

    class_weights = [1.0 / c if c > 0 else 0 for c in train_class_counts]
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True
    )

    # ── 模型 ──
    model = YeahBotModel(num_classes=NUM_CLASSES, pretrained=args.pretrained).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🏗️  模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # ── 损失函数 (加权交叉熵, 进一步补偿不平衡) ──
    weight_tensor = torch.tensor(
        [1.0 / c if c > 0 else 0 for c in class_counts], dtype=torch.float32
    )
    weight_tensor = weight_tensor / weight_tensor.sum() * NUM_CLASSES  # 归一化
    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))

    # ── 优化器 & 学习率调度 ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    # ── 冻结骨干的预热训练 ──
    if args.freeze_backbone_epochs > 0:
        model.freeze_backbone()

    # ── 训练循环 ──
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    logger.info("=" * 60)
    logger.info(f"🚀 开始训练 — 共 {args.epochs} 轮")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # 解冻骨干
        if epoch == args.freeze_backbone_epochs + 1:
            model.unfreeze_backbone()
            # 解冻后降低学习率
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * 0.1
            logger.info(f"   学习率调整为 {args.lr * 0.1:.6f} 进行全网络微调")

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc, class_acc = validate(model, val_loader, criterion, device)

        # 学习率调度
        current_lr = optimizer.param_groups[0]["lr"]
        if epoch > args.freeze_backbone_epochs:
            scheduler.step()

        elapsed = time.time() - t0

        # 记录日志
        append_log_csv(log_csv_path, epoch, train_loss, train_acc, val_loss, val_acc, class_acc, current_lr)

        # 打印
        logger.info(
            f"Epoch [{epoch:>4d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s"
        )

        # 每类准确率 (每10轮打印一次)
        if epoch % 10 == 0 or epoch == 1:
            for name in CLASS_NAMES:
                logger.info(f"   {name}: {class_acc[name]:.1f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, args.output_dir, tag="_best")
            logger.info(f"   🏆 新的最佳验证准确率: {best_val_acc*100:.2f}%")

        # 每 save_every 轮保存检查点
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, args.output_dir)

    # ── 训练结束 ──
    logger.info("=" * 60)
    logger.info(f"✅ 训练完成! 最佳验证准确率: {best_val_acc*100:.2f}%")
    logger.info("=" * 60)

    # 加载最佳权重
    model.load_state_dict(best_model_wts)

    # 保存最终 PyTorch 模型
    final_pt_path = os.path.join(args.output_dir, "yeahbot_best.pth")
    torch.save(model.state_dict(), final_pt_path)
    logger.info(f"💾 最终 PyTorch 模型: {final_pt_path}")

    # 导出 ONNX
    onnx_path = os.path.join(args.output_dir, "yeahbot_model.onnx")
    export_onnx(model, args.img_size, onnx_path, device)

    # 保存类别映射 & 指令映射 (部署时用)
    import json
    meta = {
        "class_names": CLASS_NAMES,
        "class_to_cmd": {str(k): v for k, v in CLASS_TO_CMD.items()},
        "img_size": args.img_size,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "best_val_acc": best_val_acc,
    }
    meta_path = os.path.join(args.output_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"📋 模型元数据: {meta_path}")

    logger.info("\n🎉 所有文件已保存到: " + args.output_dir)
    logger.info("   ├── yeahbot_model.onnx     (部署模型)")
    logger.info("   ├── yeahbot_best.pth        (PyTorch 最佳权重)")
    logger.info("   ├── checkpoint_epoch*.pth   (训练检查点)")
    logger.info("   ├── model_meta.json         (元数据)")
    logger.info("   └── training_log.csv        (训练日志)")


if __name__ == "__main__":
    main()
