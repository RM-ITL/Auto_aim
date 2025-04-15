"""  
    需求：订阅发布方发布的消息，并输出到终端。
    步骤：
        1.导包；
        2.初始化 ROS2 客户端；
        3.定义节点类；
            3-1.创建订阅方；
            3-2.处理订阅到的消息。
        4.调用spin函数，并传入节点对象；
        5.释放资源。
"""
 
# 1.导包；
import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
import threading
from rclpy.time import Time

# 云台角度消息状态(自瞄数据)
class PanTiltAngle:
    def __init__(self, yaw_abs = 0.0, pitch_abs = 0.0):
        self.yaw_abs = yaw_abs
        self.pitch_abs = pitch_abs 

# 3.
class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber_py')
        # 3-1.创建订阅方（自瞄数据）,数据来源自角度解算的发布者；
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/Vision/pitch_yaw',
            self.pitch_yaw_callback,
            10)

               # 创建一个订阅者，订阅`cmd_vel`主题，消息类型为Twist，回调函数为self.twist_callback(导航数据)
        self.subscription_twist = self.create_subscription(
            Twist,
            '/red_standard_robot1/cmd_vel',
            self.listener_callback,
            10)  

        # 串口配置
        self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        
        # 共享资源锁
        self.lock = threading.Lock()
        
        # 状态管理
        self.last_cmd_vel_time = self.get_clock().now()
        self.cmd_vel_timeout_threshold = 1.0  # 1秒超时
        self.send_yaw_pitch = False
        self.last_yaw = 0.0
        self.last_pitch = 0.0

        # 超时检测定时器
        self.create_timer(0.1, self.check_timeout)

    def check_timeout(self):
        """检测cmd_vel话题超时"""
        with self.lock:
            current_time = self.get_clock().now()
            time_diff = (current_time - self.last_cmd_vel_time).nanoseconds / 1e9
            is_timeout = time_diff > self.cmd_vel_timeout_threshold

            # 如果新进入超时状态且需要发送云台数据
            if is_timeout and not self.send_yaw_pitch:
                data = bytearray([0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0xFE])
                if data:
                    self.serial_port.write(data)
                    self.get_logger().info(f"话题超时，发送角度")
                self.send_yaw_pitch = True
                
    def listener_callback(self, msg):
        """速度回调处理"""
        with self.lock:
            self.last_cmd_vel_time = self.get_clock().now()
            is_stopped = abs(msg.linear.x) + abs(msg.linear.y) < 0.001

            # 计算有效控制状态
            active_control = not is_stopped and not self.send_yaw_pitch

            if active_control:
                # 发送速度指令
                self.get_logger().info(f"Received Twist message: \n"
                    f"Linear Velocity: x={msg.linear.x}, y={msg.linear.y}, z={msg.linear.z}\n"
                    f"Angular Velocity: x={msg.angular.x}, y={msg.angular.y}, z={msg.angular.z}")     

                vx_linear = msg.linear.x
                vy_linear = msg.linear.y
                vz_angular = msg.angular.z

                if vx_linear > 0 and vy_linear > 0 :
                    vel_true = 0x05
                elif vx_linear < 0 and vy_linear < 0:
                    vel_true = 0x02
                elif vx_linear > 0 and vy_linear < 0:
                    vel_true = 0x01
                elif vx_linear < 0 and vy_linear > 0:
                    vel_true = 0x06
                else:
                    vel_true = 0x10

                vx_abs = abs(vx_linear) * 10
                vy_abs = abs(vy_linear) * 10
                vz_abs = abs(vz_angular) * 10

                vxInt = int(vx_abs)
                vyInt = int(vy_abs)
                vzInt = int(vz_abs)
                
                if vx_linear < -0.90 and vx_linear > -0.99:
                    data = bytearray([0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0xFE])
                else:
                    data = bytearray([0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, vyInt, vxInt, 0x00, vel_true, 0xFE])
                self.serial_port.write(data)
                self.get_logger().info(f"发送速度:")
            else:
                self.get_logger().info(f"Received Twist message: \n"
                    f"Linear Velocity: x={msg.linear.x}, y={msg.linear.y}, z={msg.linear.z}\n"
                    f"Angular Velocity: x={msg.angular.x}, y={msg.angular.y}, z={msg.angular.z}")     

                vx_linear = msg.linear.x
                vy_linear = msg.linear.y
                vz_angular = msg.angular.z

                if vx_linear > 0 and vy_linear > 0 :
                    vel_true = 0x05
                elif vx_linear < 0 and vy_linear < 0:
                    vel_true = 0x02
                elif vx_linear > 0 and vy_linear < 0:
                    vel_true = 0x01
                elif vx_linear < 0 and vy_linear > 0:
                    vel_true = 0x06
                else:
                    vel_true = 0x10

                vx_abs = abs(vx_linear) * 10
                vy_abs = abs(vy_linear) * 10
                vz_abs = abs(vz_angular) * 10

                vxInt = int(vx_abs)
                vyInt = int(vy_abs)
                vzInt = int(vz_abs)

                if vx_linear < -0.90 and vx_linear > -0.99:
                    data = bytearray([0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0xFE])
                else:
                    data = bytearray([0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, vyInt, vxInt, 0x00, vel_true, 0xFE])
                self.serial_port.write(data)
                self.send_yaw_pitch = True

    def pitch_yaw_callback(self, msg):
        with self.lock:
            self.last_yaw = msg.data[0]
            self.last_pitch = msg.data[1]
            print(self.send_yaw_pitch)

            if self.send_yaw_pitch:
                self.get_logger().info('订阅的消息: "%s"' % msg.data)
                if len(msg.data) >= 3:
                    angle0 = msg.data[0]  # 第一个数值
                    angle1 = msg.data[1]  # 第二个数值
                    distance = msg.data[2]  # 第三个数值

                if angle0 < 0 and angle1 < 0:
                    angle_true = 0x00
                elif angle0 > 0 and angle1 < 0:
                    angle_true = 0x01
                elif angle0 < 0 and angle1 > 0:
                    angle_true = 0x02
                elif angle0 > 0 and angle1 > 0:
                    angle_true = 0x03
                else:
                    angle_true = 0x04

                # 计算 yaw_abs 和 pitch_abs
                yaw_abs = abs(angle0) * 1000
                pitch_abs = abs(angle1) * 1000

                # 计算 distance_abs
                distance_abs = int(distance)

                # 提取千位数
                thousands_distance = (distance_abs // 1000) % 10

                yawInt = int(yaw_abs)
                pitchInt = int(pitch_abs)
                # 提取高八位并存储到变量 y1, p1
                y_h8 = (yawInt >> 8) & 0xFF
                p_h8 = (pitchInt >> 8) & 0xFF
                # 提取低八位并存储到变量 y2, p2
                y_d8 = yawInt & 0xFF
                p_d8 = pitchInt & 0xFF    

                data = bytearray([0xFF, p_d8, p_h8, y_d8, y_h8, angle_true, thousands_distance, 0x00, 0x00, 0x00, 0x08, 0xFE])
                if data:
                    self.serial_port.write(data)
                    self.get_logger().debug(f"更新云台:")

def main(args=None):
    # 2.初始化 ROS2 客户端；
    rclpy.init(args=args)
 
    # 4.调用spin函数，并传入节点对象；
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node() 
    # 5.释放资源。
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
