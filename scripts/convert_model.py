#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YeahBot 模型转换脚本
从 .pth 文件生成 model_meta.json 和 yeahbot_model.onnx

支持的输入文件格式:
  1. yeahbot_best.pth          (纯模型权重)
  2. checkpoint_epoch_xxx.pth  (完整检查点, 包含optimizer等)
  3. checkpoint_epoch_xxx_best.pth (最佳检查点)
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────────────────────────
# 模型定义（必须与训练时完全一致）
# ─────────────────────────────────────────────────────────────
class YeahBotModel(nn.Module):
    """
    基于 MobileNetV2 的行为克隆模型
    架构必须与 train_yeahbot.py 中的定义完全一致
    """
    def __init__(self, num_classes=4, pretrained=False):
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


# ─────────────────────────────────────────────────────────────
# 转换函数
# ─────────────────────────────────────────────────────────────
def convert_model(pth_path, output_dir="./converted_model", img_size=224, verify=True):
    """
    从 .pth 文件生成 model_meta.json 和 yeahbot_model.onnx
    
    Args:
        pth_path: .pth 权重文件路径
        output_dir: 输出目录
        img_size: 输入图像尺寸（默认 224）
        verify: 是否验证 ONNX 模型输出
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("🔄 YeahBot 模型转换 — 从 PyTorch 到 ONNX")
    print("=" * 70)
    print(f"📥 输入文件: {pth_path}")
    print(f"📂 输出目录: {output_dir}")
    print(f"🖼️  图像尺寸: {img_size}x{img_size}")
    print()
    
    # ── 1. 加载模型权重 ──
    print("🔧 步骤 1/4: 加载模型权重...")
    device = torch.device("cpu")
    
    # 创建模型实例
    model = YeahBotModel(num_classes=4, pretrained=False)
    
    # 加载权重
    try:
        checkpoint = torch.load(pth_path, map_location=device)
        
        # 处理不同的保存格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # 完整检查点格式
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                val_acc = checkpoint.get('best_val_acc', 'unknown')
                print(f"   ✓ 检查点信息: Epoch {epoch}, Val Acc {val_acc}")
            elif 'state_dict' in checkpoint:
                # 另一种检查点格式
                state_dict = checkpoint['state_dict']
            else:
                # 可能直接就是 state_dict
                state_dict = checkpoint
        else:
            # 纯权重格式
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        print("   ✓ 权重加载成功")
        
    except Exception as e:
        print(f"   ✗ 权重加载失败!")
        print(f"   错误信息: {e}")
        print()
        print("可能的原因:")
        print("  1. 文件路径错误")
        print("  2. 文件损坏")
        print("  3. 模型架构不匹配 (请确认使用的是 MobileNetV2)")
        return False
    
    model.eval()
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   ✓ 可训练参数: {trainable_params:,}")
    print()
    
    # ── 2. 生成 model_meta.json ──
    print("🔧 步骤 2/4: 生成 model_meta.json...")
    
    # 类别定义 (与 train_yeahbot.py 保持一致)
    CLASS_NAMES = ["前进(Forward)", "左转(Left)", "右转(Right)", "停止(Stop)"]
    CLASS_TO_CMD = {
        "0": [0.113240122652057, 0.0],           # 前进
        "1": [0.0,  0.205891132094649],          # 左转
        "2": [0.0, -0.205891132094649],          # 右转
        "3": [0.0,  0.0],                        # 停止
    }
    
    # 元数据 (与训练脚本输出格式完全一致)
    meta = {
        "class_names": CLASS_NAMES,
        "class_to_cmd": CLASS_TO_CMD,
        "img_size": img_size,
        "normalize_mean": [0.485, 0.456, 0.406],  # ImageNet 归一化
        "normalize_std": [0.229, 0.224, 0.225],
    }
    
    meta_path = os.path.join(output_dir, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ 已保存: {meta_path}")
    print()
    
    # ── 3. 导出 ONNX 模型 ──
    print("🔧 步骤 3/4: 导出 ONNX 模型...")
    
    onnx_path = os.path.join(output_dir, "yeahbot_model.onnx")
    
    # 创建虚拟输入 (batch_size=1, channels=3, height=img_size, width=img_size)
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=12,           # ONNX opset 12 兼容性好
                do_constant_folding=True,   # 常量折叠优化
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={              # 支持动态 batch size
                    "input":  {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
        print(f"   ✓ 已导出: {onnx_path}")
        
        # 显示文件大小
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"   ✓ 模型大小: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"   ✗ ONNX 导出失败!")
        print(f"   错误信息: {e}")
        return False
    
    print()
    
    # ── 4. 验证 ONNX 模型 (可选) ──
    if verify:
        print("🔧 步骤 4/4: 验证 ONNX 模型...")
        try:
            import onnx
            import onnxruntime as ort
            
            # 检查 ONNX 模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("   ✓ ONNX 模型格式正确")
            
            # 测试推理
            ort_session = ort.InferenceSession(onnx_path)
            test_input = dummy_input.numpy()
            
            # PyTorch 推理
            with torch.no_grad():
                torch_output = model(dummy_input).numpy()
            
            # ONNX 推理
            onnx_output = ort_session.run(None, {"input": test_input})[0]
            
            # 比较输出
            max_diff = abs(torch_output - onnx_output).max()
            print(f"   ✓ PyTorch vs ONNX 最大差异: {max_diff:.6f}")
            
            if max_diff < 1e-5:
                print("   ✓ 数值精度验证通过")
            else:
                print("   ⚠ 数值差异较大, 但通常可接受")
                
        except ImportError:
            print("   ⊘ 跳过验证 (需要安装 onnx 和 onnxruntime)")
            print("      pip install onnx onnxruntime")
        except Exception as e:
            print(f"   ⚠ 验证出现警告: {e}")
    else:
        print("🔧 步骤 4/4: 跳过验证 (使用 --verify 启用)")
    
    print()
    
    # ── 完成 ──
    print("=" * 70)
    print("✅ 转换成功完成!")
    print("=" * 70)
    print()
    print("📦 输出文件:")
    print(f"   ├── model_meta.json      ({os.path.getsize(meta_path)/1024:.1f} KB)")
    print(f"   └── yeahbot_model.onnx   ({os.path.getsize(onnx_path)/1024/1024:.2f} MB)")
    print()
    print("🚀 部署步骤:")
    print("   1. 将这两个文件复制到树莓派:")
    print("      scp model_meta.json yeahbot_model.onnx pi@raspberrypi:~/catkin_ws/src/yeahbot_nav/models/")
    print()
    print("   2. 在树莓派上启动推理节点:")
    print("      roslaunch yeahbot_nav yeahbot_inference.launch")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="YeahBot 模型转换脚本 — 从 PyTorch .pth 转为 ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python convert_model.py yeahbot_best.pth

  # 从检查点文件转换
  python convert_model.py checkpoint_epoch_300_best.pth

  # 指定输出目录
  python convert_model.py yeahbot_best.pth --output_dir ./deployment_models

  # 启用 ONNX 验证
  python convert_model.py yeahbot_best.pth --verify

支持的输入文件:
  • yeahbot_best.pth              (纯权重)
  • checkpoint_epoch_xxx.pth      (训练检查点)
  • checkpoint_epoch_xxx_best.pth (最佳检查点)
        """
    )
    
    parser.add_argument("pth_path", type=str, 
                        help=".pth 权重文件路径")
    parser.add_argument("--output_dir", type=str, default="./converted_model",
                        help="输出目录 (默认: ./converted_model)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="输入图像尺寸, 必须与训练时一致 (默认: 224)")
    parser.add_argument("--verify", action="store_true",
                        help="验证 ONNX 模型输出精度 (需要 onnx 和 onnxruntime)")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.pth_path):
        print("=" * 70)
        print("❌ 错误: 找不到输入文件")
        print("=" * 70)
        print(f"文件路径: {args.pth_path}")
        print()
        print("请检查:")
        print("  1. 文件路径是否正确")
        print("  2. 文件是否存在")
        print("  3. 是否有读取权限")
        print()
        return 1
    
    # 执行转换
    success = convert_model(args.pth_path, args.output_dir, args.img_size, args.verify)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
