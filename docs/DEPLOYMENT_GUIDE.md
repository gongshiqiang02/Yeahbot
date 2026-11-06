# YeahBot 小车 — 模型训练 & 部署完整指南

## 一、项目概述

| 项目 | 说明 |
|------|------|
| 硬件 | YeahBot 小车, 树莓派主板 |
| 软件 | ROS1 (Noetic/Melodic), Python 3 |
| 模型 | MobileNetV2 (行为克隆 / 4类分类) |
| 部署格式 | ONNX (onnxruntime 推理) |

### 核心行为

1. **绕过障碍物**: 模型识别障碍物 → 预测左转/右转 → 通过后预测前进(自动回正)
2. **垃圾桶掉头**: 模型识别垃圾桶 → 连续预测左转(原地旋转180°)
3. **红色篮子停车**: 模型识别红色篮子 → 预测停止 → 连续确认后小车停止

### 文件清单

```
├── train_yeahbot.py          # 训练脚本
├── inference_ros_node.py     # ROS1 推理部署节点
├── yeahbot_inference.launch  # ROS1 启动文件
└── DEPLOYMENT_GUIDE.md       # 本文档
```

---

## 二、训练环境搭建 (在你的 PC / 服务器上)

### 2.1 Python 环境

```bash
# 建议使用 conda
conda create -n yeahbot python=3.9 -y
conda activate yeahbot

# 安装依赖
pip install torch torchvision    # GPU 版本参考 https://pytorch.org
pip install numpy pillow onnx onnxruntime
```

### 2.2 数据目录结构

确保你的数据目录结构如下:

```
data/2026_0210/
├── data.csv           # CSV 文件 (timestamp, image_path, linear_x, angular_z)
└── images/            # 图像文件夹
    ├── 1770701062854772316.jpg
    ├── 1770701062991438986.jpg
    └── ...
```

---

## 三、训练模型

### 3.1 基本训练命令

```bash
python train_yeahbot.py \
    --data_dir ./data/2026_0210 \
    --csv_name data.csv \
    --output_dir ./output \
    --epochs 500 \
    --batch_size 32 \
    --lr 1e-3 \
    --save_every 100
```

### 3.2 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data/2026_0210` | 数据根目录 |
| `--epochs` | 500 | 训练轮数 |
| `--batch_size` | 32 | 批大小 (显存不够可改为16) |
| `--lr` | 1e-3 | 初始学习率 |
| `--save_every` | 100 | 每100轮保存一次检查点 |
| `--val_split` | 0.15 | 验证集比例 |
| `--freeze_backbone_epochs` | 10 | 前10轮冻结主干, 只训练分类头 |
| `--img_size` | 224 | 输入图像尺寸 |

### 3.3 训练输出

训练完成后, `output/` 目录包含:

```
output/
├── yeahbot_model.onnx          # ★ 最终 ONNX 模型 (部署用)
├── yeahbot_best.pth            # PyTorch 最佳权重
├── checkpoint_epoch100.pth     # 第100轮检查点
├── checkpoint_epoch200.pth     # 第200轮检查点
├── checkpoint_epoch300.pth     # ...
├── checkpoint_epoch100_best.pth # 最佳模型检查点
├── model_meta.json             # 元数据 (类别映射、归一化参数等)
└── training_log.csv            # 训练日志 (loss, acc, 每类acc, lr)
```

### 3.4 训练策略说明

- **分阶段训练**: 前10轮冻结 MobileNetV2 主干, 只训练分类头快速收敛; 之后解冻全网络微调
- **类别平衡**: 使用加权采样器 + 加权交叉熵损失, 解决前进样本过多的问题
- **学习率调度**: 余弦退火带热重启 (CosineAnnealingWarmRestarts)
- **数据增强**: 颜色抖动、随机仿射变换、高斯模糊、随机擦除 (模拟遮挡)

---

## 四、模型部署 (树莓派)

### 4.1 树莓派环境准备

```bash
# 安装 onnxruntime (ARM 版本)
pip3 install onnxruntime

# 安装 OpenCV
pip3 install opencv-python-headless

# 确认 ROS1 已安装并能正常工作
roscore  # 测试 ROS 是否可用
```

### 4.2 创建 ROS 包

```bash
cd ~/catkin_ws/src
catkin_create_pkg yeahbot_nav rospy std_msgs sensor_msgs geometry_msgs cv_bridge

# 复制文件
mkdir -p yeahbot_nav/scripts yeahbot_nav/models yeahbot_nav/launch

cp inference_ros_node.py  yeahbot_nav/scripts/
cp yeahbot_inference.launch yeahbot_nav/launch/

# 复制训练好的模型
cp output/yeahbot_model.onnx yeahbot_nav/models/
cp output/model_meta.json    yeahbot_nav/models/

# 添加执行权限
chmod +x yeahbot_nav/scripts/inference_ros_node.py

# 编译
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 4.3 修改 launch 文件

编辑 `yeahbot_nav/launch/yeahbot_inference.launch`, 修改以下参数以匹配你的小车配置:

```xml
<!-- 根据你小车的实际话题名修改 -->
<param name="image_topic" value="/camera/image_raw"/>   <!-- 摄像头话题 -->
<param name="cmd_topic"   value="/cmd_vel"/>             <!-- 速度指令话题 -->

<!-- 推理参数调优 -->
<param name="confidence_threshold" value="0.5"/>         <!-- 置信度阈值 -->
<param name="stop_count_threshold" value="15"/>          <!-- 停止确认帧数 -->
```

### 4.4 启动推理

```bash
# 终端1: 启动 ROS
roscore

# 终端2: 启动小车底盘驱动 (根据你的小车实际情况)
roslaunch yeahbot_bringup bringup.launch

# 终端3: 启动摄像头
roslaunch usb_cam usb_cam-test.launch

# 终端4: 启动推理节点
roslaunch yeahbot_nav yeahbot_inference.launch
```

或者将所有节点写到一个总的 launch 文件中一键启动。

### 4.5 关键参数调优建议

| 参数 | 建议 | 说明 |
|------|------|------|
| `confidence_threshold` | 0.4~0.6 | 太低容易误判, 太高可能漏判 |
| `stop_count_threshold` | 10~20 | 连续多少帧预测"停止"才真正停车, 防止远处误判 |

---

## 五、调试 & 故障排除

### 5.1 查看推理结果

```bash
# 查看发布的速度指令
rostopic echo /cmd_vel

# 查看推理节点日志
rosnode info /yeahbot_inference
```

### 5.2 常见问题

**问题: 推理速度太慢**
- 检查图像分辨率, 考虑降低 `img_size` 到 160 或 128 (需要重新训练)
- 确保 onnxruntime 线程数合适 (`intra_op_num_threads`)

**问题: 模型不够准确**
- 采集更多数据, 尤其是少数类 (停止、掉头场景)
- 增加训练轮数
- 尝试调整学习率

**问题: 绕障碍物后不回正**
- 检查训练数据中绕障碍物→直行的过渡帧是否充足
- 可以补采一些障碍物侧面→前方空旷的过渡数据

**问题: 在垃圾桶前不掉头 / 掉头不完整**
- 确保训练数据中垃圾桶场景有足够的连续转向帧
- 掉头的完成依赖于模型看到不同角度后切换为前进指令

---

## 六、进阶优化建议

1. **量化加速**: 使用 ONNX Runtime 的 INT8 量化进一步提升树莓派推理速度
   ```bash
   python -m onnxruntime.quantization.preprocess --input yeahbot_model.onnx --output yeahbot_prep.onnx
   ```

2. **更多数据**: 在不同光照、不同位置多次采集数据, 提升泛化能力

3. **时序信息**: 当前模型仅使用单帧, 可考虑使用连续3~5帧堆叠作为输入, 提供时序上下文

4. **PID 平滑**: 在推理节点中增加 PID 控制器, 平滑速度指令, 避免急转
