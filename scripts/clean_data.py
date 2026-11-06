#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YeahBot 数据清洗脚本
解决"停止"标签歧义问题
"""

import csv
import os
from collections import defaultdict

def analyze_stop_sequences(csv_path):
    """分析数据集中的停止序列"""
    
    print("=" * 70)
    print("🔍 分析数据集中的'停止'样本")
    print("=" * 70)
    print()
    
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'timestamp': float(row['timestamp']),
                'image': row['image_path'],
                'linear_x': float(row['linear_x']),
                'angular_z': float(row['angular_z'])
            })
    
    # 识别停止序列
    stop_sequences = []
    in_stop = False
    current_seq = []
    
    for i, sample in enumerate(samples):
        is_stop = (abs(sample['linear_x']) < 0.05 and 
                   abs(sample['angular_z']) < 0.05)
        
        if is_stop:
            if not in_stop:
                # 开始新的停止序列
                in_stop = True
                current_seq = [i]
            else:
                current_seq.append(i)
        else:
            if in_stop:
                # 停止序列结束
                stop_sequences.append(current_seq)
                in_stop = False
                current_seq = []
    
    # 如果最后还在停止状态
    if in_stop and current_seq:
        stop_sequences.append(current_seq)
    
    print(f"总样本数: {len(samples)}")
    print(f"停止样本数: {sum(1 for s in samples if abs(s['linear_x']) < 0.05 and abs(s['angular_z']) < 0.05)}")
    print(f"停止序列数: {len(stop_sequences)}")
    print()
    
    # 分析停止序列长度
    lengths = [len(seq) for seq in stop_sequences]
    print("停止序列长度分布:")
    print(f"  最短: {min(lengths)} 帧")
    print(f"  最长: {max(lengths)} 帧")
    print(f"  平均: {sum(lengths)/len(lengths):.1f} 帧")
    print()
    
    # 分类停止序列
    print("停止序列分类:")
    very_short = sum(1 for l in lengths if l < 10)
    short = sum(1 for l in lengths if 10 <= l < 50)
    medium = sum(1 for l in lengths if 50 <= l < 200)
    long_seq = sum(1 for l in lengths if l >= 200)
    
    print(f"  极短 (<10帧):     {very_short:4d} 个 — 可能是转弯间隙")
    print(f"  短 (10-50帧):    {short:4d} 个 — 可能是短暂暂停")
    print(f"  中 (50-200帧):   {medium:4d} 个 — 可能是真正停止")
    print(f"  长 (>=200帧):    {long_seq:4d} 个 — 可能是待机/采集前")
    print()
    
    print("=" * 70)
    print()
    
    # 建议
    print("💡 建议:")
    print("-" * 70)
    print()
    print("1. 极短停止序列 (<10帧):")
    print("   这些很可能是转弯动作之间的短暂停顿")
    print("   建议: 删除这些样本或标注为前一个动作的延续")
    print()
    print("2. 长停止序列 (>=200帧):")
    print("   这些很可能是数据采集开始前的待机")
    print("   建议: 删除这些样本")
    print()
    print("3. 中等停止序列 (50-200帧):")
    print("   需要手动检查图像内容:")
    print("   - 如果图像中有红色篮子 → 保留为'停止'")
    print("   - 如果图像中无红色篮子 → 删除或重标注")
    print()
    print("=" * 70)
    print()
    
    return stop_sequences, samples


def filter_data(csv_path, output_path, 
                remove_very_short=True,  # 删除 <10 帧的停止
                remove_long=True,        # 删除 >=200 帧的停止
                remove_first_n=100):     # 删除前 N 个样本（通常是待机）
    """
    过滤数据集，移除有问题的停止样本
    """
    
    print("=" * 70)
    print("🧹 清洗数据集")
    print("=" * 70)
    print()
    
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'timestamp': row['timestamp'],
                'image': row['image_path'],
                'linear_x': float(row['linear_x']),
                'angular_z': float(row['angular_z'])
            })
    
    print(f"原始样本数: {len(samples)}")
    
    # 删除前 N 个样本
    if remove_first_n > 0:
        samples = samples[remove_first_n:]
        print(f"删除前 {remove_first_n} 个样本")
    
    # 识别要删除的停止样本
    to_remove = set()
    in_stop = False
    current_seq = []
    
    for i, sample in enumerate(samples):
        is_stop = (abs(sample['linear_x']) < 0.05 and 
                   abs(sample['angular_z']) < 0.05)
        
        if is_stop:
            if not in_stop:
                in_stop = True
                current_seq = [i]
            else:
                current_seq.append(i)
        else:
            if in_stop:
                # 停止序列结束，判断是否删除
                seq_len = len(current_seq)
                
                if remove_very_short and seq_len < 10:
                    to_remove.update(current_seq)
                    print(f"  删除极短停止序列 (索引 {current_seq[0]}-{current_seq[-1]}, {seq_len}帧)")
                
                if remove_long and seq_len >= 200:
                    to_remove.update(current_seq)
                    print(f"  删除长停止序列 (索引 {current_seq[0]}-{current_seq[-1]}, {seq_len}帧)")
                
                in_stop = False
                current_seq = []
    
    # 过滤样本
    filtered_samples = [s for i, s in enumerate(samples) if i not in to_remove]
    
    print()
    print(f"清洗后样本数: {len(filtered_samples)}")
    print(f"删除样本数: {len(samples) - len(filtered_samples)}")
    print()
    
    # 保存清洗后的数据
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'image_path', 'linear_x', 'angular_z'])
        writer.writeheader()
        for sample in filtered_samples:
            writer.writerow({
                'timestamp': sample['timestamp'],
                'image_path': sample['image'],
                'linear_x': sample['linear_x'],
                'angular_z': sample['angular_z']
            })
    
    print(f"✅ 清洗后的数据已保存到: {output_path}")
    print()
    print("=" * 70)
    print()
    
    # 统计清洗后的类别分布
    def action_to_class(lx, az):
        if abs(lx) > 0.05:
            return 0
        elif az > 0.05:
            return 1
        elif az < -0.05:
            return 2
        else:
            return 3
    
    class_counts = [0, 0, 0, 0]
    for sample in filtered_samples:
        cls = action_to_class(sample['linear_x'], sample['angular_z'])
        class_counts[cls] += 1
    
    total = len(filtered_samples)
    class_names = ["前进", "左转", "右转", "停止"]
    
    print("清洗后的类别分布:")
    print("-" * 70)
    for i, name in enumerate(class_names):
        pct = class_counts[i] / total * 100 if total > 0 else 0
        print(f"{name}: {class_counts[i]:>6} 样本 ({pct:>5.1f}%)")
    print("-" * 70)
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  分析数据: python clean_data.py analyze data.csv")
        print("  清洗数据: python clean_data.py filter data.csv data_cleaned.csv")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "analyze":
        if len(sys.argv) < 3:
            print("用法: python clean_data.py analyze <csv_file>")
            sys.exit(1)
        analyze_stop_sequences(sys.argv[2])
    
    elif mode == "filter":
        if len(sys.argv) < 4:
            print("用法: python clean_data.py filter <input_csv> <output_csv>")
            sys.exit(1)
        filter_data(sys.argv[2], sys.argv[3])
    
    else:
        print(f"未知模式: {mode}")
        print("支持的模式: analyze, filter")
