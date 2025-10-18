/**
 * 需求：订阅发布方发布的消息，并通过串口发送控制信号。
 * 步骤：
 *   1. 包含必要的头文件；
 *   2. 初始化 ROS2 客户端；
 *   3. 定义节点类；
 *      3-1. 创建订阅方；
 *      3-2. 处理订阅到的消息。
 *   4. 调用spin函数，并传入节点对象；
 *   5. 释放资源。
 */

// 1. 包含必要的头文件
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <mutex>
#include <thread>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <cstring>
#include <filesystem>
#include <climits> 

// 云台角度消息状态(自瞄数据)
struct PanTiltAngle {
    double yaw_abs = 0.0;
    double pitch_abs = 0.0;
    
    PanTiltAngle(double yaw = 0.0, double pitch = 0.0)
        : yaw_abs(yaw), pitch_abs(pitch) {}
};

// 加载配置文件
YAML::Node load_config() {
    // 首先尝试在项目源代码目录中查找配置文件
    std::string project_root = "/home/guo/ITL_sentry_auto";
    std::string config_path = project_root + "/src/config/robomaster_vision_config.yaml";
    
    // 如果文件不存在，可以尝试其他可能的位置
    if (!std::filesystem::exists(config_path)) {
        // 尝试当前工作目录
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd)) != nullptr) {
            config_path = std::string(cwd) + "/config/robomaster_vision_config.yaml";
        }
    }
    
    // 打开并加载文件
    try {
        return YAML::LoadFile(config_path);
    } catch (const YAML::Exception& e) {
        std::cerr << "无法加载配置文件: " << e.what() << std::endl;
        throw;
    }
}

// 串口类，提供简单的串口通信接口
class SerialPort {
public:
    SerialPort(const std::string& port, int baudrate, int timeout = 1) 
        : port_(port), baudrate_(baudrate), timeout_(timeout), fd_(-1) {
        open();
    }
    
    ~SerialPort() {
        close();
    }
    
    bool open() {
        fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd_ == -1) {
            std::cerr << "无法打开串口: " << port_ << std::endl;
            return false;
        }
        
        // 配置串口参数
        struct termios options;
        tcgetattr(fd_, &options);
        
        // 设置波特率
        speed_t baud;
        switch (baudrate_) {
            case 9600:   baud = B9600;   break;
            case 19200:  baud = B19200;  break;
            case 38400:  baud = B38400;  break;
            case 57600:  baud = B57600;  break;
            case 115200: baud = B115200; break;
            default:     baud = B9600;   break;
        }
        
        cfsetispeed(&options, baud);
        cfsetospeed(&options, baud);
        
        options.c_cflag |= (CLOCAL | CREAD); // 启用接收器并忽略调制解调器控制线
        options.c_cflag &= ~PARENB;          // 无奇偶校验
        options.c_cflag &= ~CSTOPB;          // 1个停止位
        options.c_cflag &= ~CSIZE;           // 掩码字符大小位
        options.c_cflag |= CS8;              // 8位数据位
        options.c_cflag &= ~CRTSCTS;         // 无硬件流控制
        
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // 原始输入
        options.c_oflag &= ~OPOST;                          // 原始输出
        
        options.c_cc[VMIN] = 0;              // 最小字符数
        options.c_cc[VTIME] = timeout_ * 10; // 超时时间 (十分之一秒)
        
        tcsetattr(fd_, TCSANOW, &options);
        tcflush(fd_, TCIOFLUSH);
        
        return true;
    }
    
    void close() {
        if (fd_ != -1) {
            ::close(fd_);
            fd_ = -1;
        }
    }
    
    bool write(const std::vector<uint8_t>& data) {
        if (fd_ == -1) return false;
        
        ssize_t bytes_written = ::write(fd_, data.data(), data.size());
        return bytes_written == static_cast<ssize_t>(data.size());
    }
    
    bool write(const uint8_t* data, size_t size) {
        if (fd_ == -1) return false;
        
        ssize_t bytes_written = ::write(fd_, data, size);
        return bytes_written == static_cast<ssize_t>(size);
    }
    
private:
    std::string port_;
    int baudrate_;
    int timeout_;
    int fd_;
};

// 3. 定义节点类
class SerialSubscriber : public rclcpp::Node {
public:
    SerialSubscriber() : Node("serial_subscriber") {
        try {
            // 加载配置
            config_ = load_config();
            serial_config_ = config_["serial"];
            
            // 3-1. 创建订阅方（自瞄数据）
            std::string angle_sub_topic = serial_config_["topics"]["angle_sub"].as<std::string>();
            subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
                angle_sub_topic,
                10,
                std::bind(&SerialSubscriber::pitch_yaw_callback, this, std::placeholders::_1)
            );
            
            // 串口配置
            std::string port = serial_config_["port"].as<std::string>();
            int baudrate = serial_config_["baudrate"].as<int>();
            int timeout = serial_config_["timeout"].as<int>();
            
            // 初始化串口
            serial_port_ = std::make_unique<SerialPort>(port, baudrate, timeout);
            
            // 状态管理
            last_cmd_vel_time_ = this->now();
            cmd_vel_timeout_threshold_ = serial_config_["timeout_threshold"].as<double>();
            send_yaw_pitch_ = true;
            last_yaw_ = 0.0;
            last_pitch_ = 0.0;
            
            // 协议头尾
            header_ = static_cast<uint8_t>(serial_config_["protocol"]["header"].as<int>());
            footer_ = static_cast<uint8_t>(serial_config_["protocol"]["footer"].as<int>());
            
            RCLCPP_INFO(this->get_logger(), "串口订阅者已初始化: 端口=%s, 波特率=%d", port.c_str(), baudrate);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "初始化失败: %s", e.what());
            throw;
        }
    }
    
private:
    // 3-2. 处理订阅到的消息
    void pitch_yaw_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            last_yaw_ = msg->data[0];
            last_pitch_ = msg->data[1];
            RCLCPP_DEBUG(this->get_logger(), "发送状态: %s", send_yaw_pitch_ ? "true" : "false");
            
            if (send_yaw_pitch_) {
                RCLCPP_INFO(this->get_logger(), "订阅的消息: [%.2f, %.2f]", msg->data[0], msg->data[1]);
                
                if (msg->data.size() >= 3) {
                    double angle0 = msg->data[0];  // 第一个数值 (yaw)
                    double angle1 = msg->data[1];  // 第二个数值 (pitch)
                    double distance = msg->data[2];  // 第三个数值 (距离)
                    
                    // 根据角度正负确定象限
                    int angle_true;
                    if (angle0 < 0 && angle1 < 0) {
                        angle_true = 0x00;
                    } else if (angle0 > 0 && angle1 < 0) {
                        angle_true = 0x01;
                    } else if (angle0 < 0 && angle1 > 0) {
                        angle_true = 0x02;
                    } else if (angle0 > 0 && angle1 > 0) {
                        angle_true = 0x03;
                    } else {
                        angle_true = 0x04;
                    }
                    
                    // 计算 yaw_abs 和 pitch_abs (转换为毫弧度)
                    double yaw_abs = std::abs(angle0) * 1000;
                    double pitch_abs = std::abs(angle1) * 1000;
                    
                    // 计算 distance_abs
                    int distance_abs = static_cast<int>(distance);
                    
                    // 提取千位数
                    [[maybe_unused]] int thousands_distance = (distance_abs / 1000) % 10;
                    
                    // 转换为整数
                    int yawInt = static_cast<int>(yaw_abs);
                    int pitchInt = static_cast<int>(pitch_abs);
                    
                    // 提取高八位和低八位
                    uint8_t y_h8 = (yawInt >> 8) & 0xFF;
                    uint8_t p_h8 = (pitchInt >> 8) & 0xFF;
                    uint8_t y_d8 = yawInt & 0xFF;
                    uint8_t p_d8 = pitchInt & 0xFF;
                    
                    // 组装数据包
                    std::vector<uint8_t> data = {
                        header_,
                        p_d8,
                        p_h8,
                        y_d8,
                        y_h8,
                        footer_
                    };
                    
                    // 发送数据
                    if (!data.empty()) {
                        if (serial_port_->write(data)) {
                            RCLCPP_DEBUG(this->get_logger(), "已发送云台控制数据");
                        } else {
                            RCLCPP_ERROR(this->get_logger(), "串口发送失败");
                        }
                    }
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        RCLCPP_INFO(this->get_logger(), "串口发送延时: %.2f ms", processing_time);
    }
    
    // 成员变量
    YAML::Node config_;
    YAML::Node serial_config_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
    std::unique_ptr<SerialPort> serial_port_;
    std::mutex mutex_;
    rclcpp::Time last_cmd_vel_time_;
    double cmd_vel_timeout_threshold_;
    bool send_yaw_pitch_;
    double last_yaw_;
    double last_pitch_;
    uint8_t header_;
    uint8_t footer_;
};

int main(int argc, char * argv[]) {
    // 2. 初始化 ROS2 客户端
    rclcpp::init(argc, argv);
    
    // 创建节点
    std::shared_ptr<SerialSubscriber> serial_subscriber;
    try {
        serial_subscriber = std::make_shared<SerialSubscriber>();
        
        // 4. 调用spin函数，并传入节点对象
        rclcpp::spin(serial_subscriber);
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    // 5. 释放资源
    rclcpp::shutdown();
    return 0;
}
