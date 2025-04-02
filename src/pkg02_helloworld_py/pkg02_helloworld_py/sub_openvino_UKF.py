import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
from copy import deepcopy
from math import log, exp, sqrt
from numpy import eye, zeros, dot, isscalar, outer
from scipy.linalg import cholesky, inv
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, unscented_transform
from filterpy.stats import logpdf
from filterpy.common import pretty_str
import hashlib
from openvino.runtime import Core


class UnscentedKalmanFilterWrapper:
    """UKF封装类，用于2D位置和速度的跟踪"""

    def __init__(self, dim_x, dim_z, dt, node=None):
        self.node = node  # 引用ROS节点用于日志记录

        # 定义sigma点生成方法
        points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2.0, kappa=-1)

        # 定义状态转移函数
        def fx(x, dt):
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0 ],
                [0, 0, 0, 1 ]
            ])
            return np.dot(F, x)

        # 定义测量函数
        def hx(x):
            return x[:2]

        # 初始化UKF
        self.ukf = UnscentedKalmanFilter(
            dim_x=dim_x,
            dim_z=dim_z,
            hx=hx,
            dt=dt,
            fx=fx,
            points=points
        )

        # 初始化状态向量和协方差矩阵
        self.ukf.x = np.zeros(dim_x)
        self.ukf.P = np.eye(dim_x) * 500.0  # 初始不确定性
        self.ukf.Q = np.eye(dim_x) * 0.1    # 过程噪声，根据实际情况调整
        self.ukf.R = np.eye(dim_z) * 0.1    # 测量噪声，根据实际情况调整

        # 记录先验和后验状态
        self.x_prior = self.ukf.x.copy()
        self.P_prior = self.ukf.P.copy()
        self.x_post = self.ukf.x.copy()
        self.P_post = self.ukf.P.copy()

    def predict(self):
        """执行预测步骤"""
        self.ukf.predict()
        self.x_prior = self.ukf.x.copy()
        self.P_prior = self.ukf.P.copy()

    def update(self, z):
        """执行更新步骤"""
        self.ukf.update(z)
        self.x_post = self.ukf.x.copy()
        self.P_post = self.ukf.P.copy()

    @property
    def log_likelihood(self):
        """计算最后一次测量的对数似然性"""
        return self.ukf.log_likelihood

    @property
    def likelihood(self):
        """计算最后一次测量的似然性"""
        return self.ukf.likelihood

    @property
    def mahalanobis(self):
        """计算创新的马氏距离"""
        return self.ukf.mahalanobis

    def __repr__(self):
        return self.ukf.__repr__()


# 定义类别
CLASSES = [
    'armor_blue', 'armor_red'
]


def name_to_color(name):
    # 使用哈希为每个类别生成唯一颜色
    hash_str = hashlib.md5(name.encode('utf-8')).hexdigest()
    r = int(hash_str[0:2], 16)
    g = int(hash_str[2:4], 16)
    b = int(hash_str[4:6], 16)
    return (r, g, b)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2]/2
    y[..., 1] = x[..., 1] - x[..., 3]/2
    y[..., 2] = x[..., 0] + x[..., 2]/2
    y[..., 3] = x[..., 1] + x[..., 3]/2
    return y


def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, xmax - xmin)
    inter_h = np.maximum(0, ymax - ymin)
    intersection = inter_w * inter_h

    box_area = (box[2]-box[0])*(box[3]-box[1])
    boxes_area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])

    union = box_area + boxes_area - intersection
    iou = intersection / union
    return iou


class YOLOv8_OpenVINO:
    def __init__(self, model_path, conf_threshold, iou_threshold, node, device='GPU'):
        """
        初始化 YOLOv8 模型使用 OpenVINO
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.node = node  # 引用 ROS 节点用于日志记录

        # 加载类别名称
        self.classes = CLASSES

        # 生成类别对应的颜色
        self.color_palette = [name_to_color(cls) for cls in self.classes]

        # 加载 OpenVINO 模型
        self.core = Core()
        try:
            self.model = self.core.read_model(model_path)
            self.compiled_model = self.core.compile_model(self.model, device_name=device)
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            self.input_shape = self.input_layer.shape
            self.input_width = self.input_shape[3]
            self.input_height = self.input_shape[2]
            self.node.get_logger().info(f"Successfully loaded OpenVINO model: {model_path} on device: {device}")
        except Exception as e:
            self.node.get_logger().error(f"Failed to load OpenVINO model: {e}")
            raise e

    def preprocess(self, img):
        """
        前处理：调整图像大小、颜色空间转换、归一化等
        """
        self.img = img
        self.img_height, self.img_width = self.img.shape[:2]
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (self.input_width, self.input_height))
        input_image = resized.astype(np.float32) / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_image, 0)
        self.node.get_logger().debug(f"Preprocessed image shape: {input_tensor.shape}")
        return input_tensor

    def draw_detections(self, img, box, score, class_id):
        """
        绘制检测框和标签（可选）
        """
        x1, y1, x2, y2 = box
        color = self.color_palette[class_id]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(
            img, (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)), color, cv2.FILLED
        )
        cv2.putText(img, label, (int(label_x), int(label_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # 返回检测框的左上角坐标和宽高
        x = max(int(x1), 0)
        y = max(int(y1), 0)
        w = int(x2 - x1)
        h = int(y2 - y1)

        return [x, y, w, h]
    
    def draw_predtions(self, img, box, score, class_id, predicted_point=None):
        """
        绘制检测框和标签，并可选绘制预测点
        """
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(
            img, (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)), color, cv2.FILLED
        )
        cv2.putText(img, label, (int(label_x), int(label_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        x = max(int(x1 + w / 2), 0)
        y = max(int(y1 + h / 2), 0)
        w = int(w)
        h = int(h)

        if predicted_point is not None:
            # 绘制预测点
            pred_x, pred_y = predicted_point
            cv2.circle(img, (int(pred_x), int(pred_y)), 5, (0, 255, 0), -1)
            cv2.putText(img, "Predicted", (int(pred_x) + 10, int(pred_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 返回修改后的图像
        return img
    
    def postprocess(self, output):
        """
        后处理：解析模型输出，应用NMS，提取检测结果
        """
        predictions = np.squeeze(output, axis=0).T

        self.node.get_logger().debug(f"总预测数量: {predictions.shape[0]}")

        boxes = predictions[:, :4]
        class_scores = sigmoid(predictions[:, 4:])
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        self.node.get_logger().debug(f"应用置信度阈值后: {boxes.shape[0]} 个框")
        if len(confidences) > 0:
            self.node.get_logger().debug(
                f"置信度分布: 最小={confidences.min():.4f}, 最大={confidences.max():.4f}, 平均={confidences.mean():.4f}"
            )

        if len(confidences) == 0:
            self.node.get_logger().warn("No detections after confidence thresholding.")
            return [], [], []

        boxes_xyxy = xywh2xyxy(boxes)
        scale_w = self.img_width / self.input_width
        scale_h = self.img_height / self.input_height
        boxes_xyxy[:, [0, 2]] *= scale_w
        boxes_xyxy[:, [1, 3]] *= scale_h
        boxes_xyxy = boxes_xyxy.astype(np.int32)

        boxes_list = boxes_xyxy.tolist()
        scores_list = confidences.tolist()

        final_boxes = []
        final_confidences = []
        final_class_ids = []

        unique_classes = np.unique(class_ids)
        for cls in unique_classes:
            cls_mask = (class_ids == cls)
            cls_boxes = [boxes_list[i] for i in range(len(class_ids)) if cls_mask[i]]
            cls_scores = [scores_list[i] for i in range(len(class_ids)) if cls_mask[i]]

            if len(cls_boxes) == 0:
                continue

            cls_boxes_xywh = []
            for box in cls_boxes:
                x1, y1, x2, y2 = box
                cls_boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

            indices = cv2.dnn.NMSBoxes(cls_boxes_xywh, cls_scores, self.conf_threshold, self.iou_threshold)

            if len(indices) > 0:
                for i in indices.flatten():
                    final_boxes.append(cls_boxes[i])
                    final_confidences.append(cls_scores[i])
                    final_class_ids.append(cls)

        self.node.get_logger().debug(f"应用NMS后: {len(final_boxes)} 个框")

        return final_boxes, final_confidences, final_class_ids

    def infer(self, cv_image):
        """
        执行前处理、推理和后处理
        """
        input_tensor = self.preprocess(cv_image)

        try:
            outputs = self.compiled_model([input_tensor])[self.output_layer]
            self.node.get_logger().debug(f"Model raw outputs: {outputs.shape}")
        except Exception as e:
            self.node.get_logger().error(f"Inference Error: {e}")
            return cv_image, []

        boxes, confidences, class_ids = self.postprocess(outputs)

        detections = []
        if len(boxes) > 0:
            for box, score, cls_id in zip(boxes, confidences, class_ids):
                rectangle = self.draw_detections(cv_image, box, score, cls_id)
                    # 创建检测字典并添加到 detections 列表
                detection = {
                    'box': rectangle,          # [x1, y1, x2, y2]
                    'class_id': cls_id,  # 类别ID
                    'score': score       # 置信度分数
                }
                detections.append(detection)

        return cv_image, detections

class YOLOROSSubscriber(Node):
    def __init__(self):
        super().__init__('subscriber_py')

        # 初始化 YOLO 推理使用 OpenVINO
        self.yolo = YOLOv8_OpenVINO('/home/shaobing/best_red.onnx', 0.55, 0.5, self, device='CPU')  # 请替换为您的 OpenVINO 模型路径

        # 设置订阅者和发布者
        self.image_sub = self.create_subscription(Image, 'image_topic', self.callback, 10)
        self.detections_pub = self.create_publisher(Point, 'yolo_detections', 10)
        self.predictions_pub = self.create_publisher(Point, 'yolo_predictions', 10)
        self.annotated_image_pub = self.create_publisher(Image, 'yolo_detections_image', 10)  # 新增发布者

        self.bridge = CvBridge()

        self.ukf = None

        self.get_logger().info("YOLOROSSubscriber initialized.")

    def callback(self, data):
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            self.get_logger().debug("Image converted from ROS message to OpenCV format.")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return
        except Exception as e:
            self.get_logger().error(f"Unexpected Error: {e}")
            return

        # 执行YOLO推理
        output_image, detections = self.yolo.infer(cv_image)

        if detections:
            # 假设只跟踪第一个检测到的目标
            detection = detections[0]
            rect = detection['box']
            class_id = detection['class_id']
            score = detection['score']
            x, y, w, h = rect
            center_x = float(x + w / 2)
            center_y = float(y + h / 2)

            # 创建Point消息发布检测坐标
            detection_point = Point()
            detection_point.x = center_x
            detection_point.y = center_y
            detection_point.z = 0.0  # 将 z 设置为浮点数

            width_true = 2.0
            highth_true = 2.0
            left_top = Point(x = center_x - float(rect[2])/width_true, y = center_y - float(rect[3])/highth_true, z=0.0)  # 左上角
            right_top = Point(x = center_x + float(rect[2])/width_true, y = center_y - float(rect[3])/highth_true, z=0.0)  # 右上角
            right_low = Point(x = center_x + float(rect[2])/width_true, y = center_y + float(rect[3])/highth_true, z=0.0)  # 右下角
            left_low = Point(x = center_x - float(rect[2])/width_true, y = center_y + float(rect[3])/highth_true, z=0.0)  # 左下角

            # 创建Float64MultiArray消息
            point = Float64MultiArray()
            point.data = [
                left_top.x, left_top.y, left_top.z,
                right_top.x, right_top.y, right_top.z,
                right_low.x, right_low.y, right_low.z,
                left_low.x, left_low.y, left_low.z,
            ]

            # 打印调试信息
            self.get_logger().info(
                f'Publishing ll - x: {left_low.x}, y: {left_low.y}, z: {left_low.z}\n'
                f'Publishing lt - x: {left_top.x}, y: {left_top.y}, z: {left_top.z}\n'
                f'Publishing rt - x: {right_top.x}, y: {right_top.y}, z: {right_top.z}\n'
                f'Publishing rl - x: {right_low.x}, y: {right_low.y}, z: {right_low.z}\n'
                f'Publishing center - x: {center_x}, y: {center_y}, z: {right_low.z}\n'
                f'Publishing center - x: {rect[0]}, y: {rect[1]}, w:{rect[2]}, h:{rect[3]}\n'

            )

            self.detections_pub.publish(point)

             # 初始化UKF如果尚未初始化
            if self.ukf is None:
                dt = 1/30.0  # 假设帧率为30 FPS，可根据实际情况调整
                self.ukf = UnscentedKalmanFilterWrapper(dim_x=4, dim_z=2, dt=dt, node=self)
                self.ukf.ukf.x = np.array([center_x, center_y, 0, 0])
                self.get_logger().info(f"UKF initial state set to: {self.ukf.ukf.x}")
        

            # 更新UKF
            measurement = np.array([detection_point.x, detection_point.y])
            self.ukf.predict()  # 执行预测步骤
            self.ukf.update(measurement)  # 执行更新步骤

            # 获取当前帧的预测位置
            predicted_x = self.ukf.ukf.x[0]
            predicted_y = self.ukf.ukf.x[1]

            # 创建Point消息发布当前预测坐标（用于绘制）
            prediction_point = Point()
            prediction_point.x = predicted_x
            prediction_point.y = predicted_y
            prediction_point.z = 0.0  # 将 z 设置为浮点数

            width_true = 2.0
            highth_true = 2.0

            left_top = Point(x = predicted_x- float(rect[2])/width_true, y = predicted_y - float(rect[3])/highth_true, z=0.0)  # 左上角
            right_top = Point(x = predicted_x + float(rect[2])/width_true, y = predicted_y - float(rect[3])/highth_true, z=0.0)  # 右上角
            right_low = Point(x = predicted_x + float(rect[2])/width_true, y = predicted_y + float(rect[3])/highth_true, z=0.0)  # 右下角
            left_low = Point(x = predicted_x - float(rect[2])/width_true, y = predicted_y + float(rect[3])/highth_true, z=0.0)  # 左下角

            # 创建Float64MultiArray消息
            point = Float64MultiArray()
            point.data = [
                left_top.x, left_top.y, left_top.z,
                right_top.x, right_top.y, right_top.z,
                right_low.x, right_low.y, right_low.z,
                left_low.x, left_low.y, left_low.z,
            ]

            # 打印调试信息
            self.get_logger().info(
                f'Publishing ll - x: {left_low.x}, y: {left_low.y}, z: {left_low.z}\n'
                f'Publishing lt - x: {left_top.x}, y: {left_top.y}, z: {left_top.z}\n'
                f'Publishing rt - x: {right_top.x}, y: {right_top.y}, z: {right_top.z}\n'
                f'Publishing rl - x: {right_low.x}, y: {right_low.y}, z: {right_low.z}\n'
                f'Publishing center - x: {center_x}, y: {center_y}, z: {right_low.z}\n'
                f'Publishing center - x: {rect[0]}, y: {rect[1]}, w:{rect[2]}, h:{rect[3]}\n'
            )

            self.predictions_pub.publish(point)
            self.get_logger().info(
                f'Publishing Prediction Point - x: {prediction_point.x}, y: {prediction_point.y}, z: {prediction_point.z}'
            )

            # 绘制检测框和当前预测点
            annotated_image = self.yolo.draw_predtions(
                output_image, rect, score, class_id, predicted_point=(predicted_x, predicted_y)
            )
        else:
            self.get_logger().info("No rectangles detected to publish.")
            point = Float64MultiArray()
            point.data = [
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                ]
            self.detections_pub.publish(point)
            annotated_image = output_image

        # 发布带检测框和预测点的图像
        try:
            annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            self.annotated_image_pub.publish(annotated_image_msg)
            self.get_logger().debug("Annotated image published.")
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert annotated image: {e}")

        # 可选：显示带检测框和预测点的图像
        cv2.imshow('YOLO Detection with UKF', annotated_image)
        cv2.waitKey(1)  # 启用窗口刷新

    def destroy_node(self):
        """销毁节点时释放资源"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    yolo_ros_subscriber = YOLOROSSubscriber()
    try:
        rclpy.spin(yolo_ros_subscriber)
    except KeyboardInterrupt:
        yolo_ros_subscriber.get_logger().info("Shutting down YOLO ROS Subscriber.")
    finally:
        yolo_ros_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()