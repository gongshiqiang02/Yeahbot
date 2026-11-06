#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  YeahBot 小车 — ROS1 推理节点 (ONNX Runtime)
=============================================================================
  功能: 订阅摄像头图像, 推理动作, 发布速度指令
  
  行为逻辑:
    模型输出 → 前进 → 发布 linear_x
    模型输出 → 左转/右转 → 发布 angular_z (绕障碍物 / 掉头)
    模型输出 → 停止 → 发布零速度并关闭节点
  
  部署环境: 树莓派 + ROS1 (Noetic/Melodic)
=============================================================================
"""

import os
import sys
import json
import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("请安装 onnxruntime: pip3 install onnxruntime")
    sys.exit(1)

try:
    import rospy
    from sensor_msgs.msg import Image as RosImage
    from geometry_msgs.msg import Twist
    from cv_bridge import CvBridge
except ImportError:
    print("请确保已安装 ROS1 (rospy, sensor_msgs, geometry_msgs, cv_bridge)")
    sys.exit(1)

import cv2


# ─────────────────────────────────────────────────────────────
# 类别定义 (与训练保持一致)
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = ["Forward", "Left", "Right", "Stop"]

# 默认指令 (可从 model_meta.json 读取覆盖)
DEFAULT_CMD = {
    0: (0.113240122652057, 0.0),       # 前进
    1: (0.0,  0.205891132094649),      # 左转
    2: (0.0, -0.205891132094649),      # 右转
    3: (0.0,  0.0),                    # 停止
}


class YeahBotInferenceNode:
    """ROS1 推理节点"""

    def __init__(self):
        rospy.init_node("yeahbot_inference", anonymous=True)

        # ── 参数 ──
        model_path = rospy.get_param("~model_path", "yeahbot_model.onnx")
        meta_path  = rospy.get_param("~meta_path",  "model_meta.json")
        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw")
        self.cmd_topic   = rospy.get_param("~cmd_topic",   "/cmd_vel")
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.5)
        self.stop_count_threshold = rospy.get_param("~stop_count_threshold", 15)

        # ── 加载元数据 ──
        self.img_size = 224
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.class_to_cmd = DEFAULT_CMD

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.img_size = meta.get("img_size", 224)
            self.mean = np.array(meta.get("normalize_mean", self.mean), dtype=np.float32)
            self.std  = np.array(meta.get("normalize_std", self.std), dtype=np.float32)
            if "class_to_cmd" in meta:
                self.class_to_cmd = {int(k): tuple(v) for k, v in meta["class_to_cmd"].items()}
            rospy.loginfo(f"已加载模型元数据: {meta_path}")

        # ── 加载 ONNX 模型 ──
        rospy.loginfo(f"正在加载模型: {model_path}")
        # 使用 CPU (树莓派)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        rospy.loginfo("模型加载完成!")

        # ── ROS 接口 ──
        self.bridge = CvBridge()
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.image_sub = rospy.Subscriber(self.image_topic, RosImage, self.image_callback, queue_size=1, buff_size=2**24)

        # ── 状态 ──
        self.consecutive_stop_count = 0  # 连续预测"停止"的计数
        self.is_running = True
        self.last_action = -1
        self.inference_count = 0

        rospy.loginfo("=" * 50)
        rospy.loginfo("YeahBot 推理节点已启动!")
        rospy.loginfo(f"  图像话题: {self.image_topic}")
        rospy.loginfo(f"  指令话题: {self.cmd_topic}")
        rospy.loginfo(f"  图像尺寸: {self.img_size}x{self.img_size}")
        rospy.loginfo(f"  置信度阈值: {self.confidence_threshold}")
        rospy.loginfo(f"  停止确认帧数: {self.stop_count_threshold}")
        rospy.loginfo("=" * 50)

    def preprocess(self, cv_image):
        """
        预处理图像: resize → normalize → CHW → add batch dim
        与训练时的 val_tf 保持一致
        """
        # BGR → RGB
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        # 归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0
        # ImageNet normalize
        img = (img - self.mean) / self.std
        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))
        # 添加 batch 维度
        img = np.expand_dims(img, axis=0)
        return img

    def image_callback(self, msg):
        """摄像头图像回调"""
        if not self.is_running:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"图像转换失败: {e}")
            return

        # ── 预处理 ──
        input_tensor = self.preprocess(cv_image)

        # ── 推理 ──
        t0 = time.time()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0][0]  # [num_classes]
        inference_time = (time.time() - t0) * 1000  # ms

        # Softmax 获取概率
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        self.inference_count += 1

        # ── 动作决策 ──
        twist = Twist()

        if predicted_class == 3 and confidence > self.confidence_threshold:
            # 预测"停止" — 需要连续多帧确认, 防止误判
            self.consecutive_stop_count += 1

            if self.consecutive_stop_count >= self.stop_count_threshold:
                # 确认停止: 发布零速度, 关闭节点
                rospy.loginfo("🛑 检测到红色篮子 — 小车停止!")
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                self.is_running = False
                rospy.signal_shutdown("到达终点 (红色篮子), 任务完成!")
                return
            else:
                # 还在确认中, 先减速
                twist.linear.x = 0.0
                twist.angular.z = 0.0
        else:
            # 非停止指令, 重置计数
            self.consecutive_stop_count = 0

            if confidence > self.confidence_threshold:
                lx, az = self.class_to_cmd[predicted_class]
                twist.linear.x = lx
                twist.angular.z = az
            else:
                # 置信度不够, 安全起见减速前进
                twist.linear.x = self.class_to_cmd[0][0] * 0.5
                twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

        # 日志 (每20帧打印一次, 避免刷屏)
        if self.inference_count % 20 == 0:
            rospy.loginfo(
                f"[{self.inference_count}] "
                f"Action: {CLASS_NAMES[predicted_class]} "
                f"(conf: {confidence:.2f}) | "
                f"cmd: ({twist.linear.x:.3f}, {twist.angular.z:.3f}) | "
                f"{inference_time:.1f}ms"
            )

    def run(self):
        """主循环"""
        rospy.spin()

    def shutdown(self):
        """安全关闭"""
        rospy.loginfo("正在关闭, 发送零速度...")
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)


if __name__ == "__main__":
    node = YeahBotInferenceNode()
    rospy.on_shutdown(node.shutdown)
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
