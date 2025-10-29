// serial_node.cpp - 完整实现版本
#include "serial_node.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>

// 构造函数
SerialSenderNode::SerialSenderNode() 
    : Node("serial_sender_node"), serial_fd_(-1) {
    
    RCLCPP_INFO(this->get_logger(), "串口发送节点启动中...");
    
    // 声明并获取参数
    this->declare_parameter("serial_port", "/dev/ttyUSB0");
    this->declare_parameter("baud_rate", 115200);
    this->declare_parameter("debug_mode", false);
    
    config_.port_name = this->get_parameter("serial_port").as_string();
    int baud = this->get_parameter("baud_rate").as_int();
    debug_mode_ = this->get_parameter("debug_mode").as_bool();
    
    // 将波特率数值转换为termios常量
    switch(baud) {
        case 9600:   config_.baud_rate = B9600; break;
        case 19200:  config_.baud_rate = B19200; break;
        case 38400:  config_.baud_rate = B38400; break;
        case 57600:  config_.baud_rate = B57600; break;
        case 115200: config_.baud_rate = B115200; break;
        case 230400: config_.baud_rate = B230400; break;
        default:
            RCLCPP_WARN(this->get_logger(), 
                "不支持的波特率 %d，使用默认值 115200", baud);
            config_.baud_rate = B115200;
    }
    
    // 初始化串口
    if (!initializeSerial()) {
        RCLCPP_ERROR(this->get_logger(), "串口初始化失败");
        throw std::runtime_error("串口初始化失败");
    }
    
    // 创建订阅器
    serial_cmd_sub_ = this->create_subscription<autoaim_msgs::msg::Serialcmd>(
        "serial_cmd", 10,
        std::bind(&SerialSenderNode::serialCmdCallback, this, std::placeholders::_1)
    );
    
    // 创建统计信息定时器
    stats_timer_ = this->create_wall_timer(
        std::chrono::seconds(5),
        std::bind(&SerialSenderNode::printStatistics, this)
    );
    
    RCLCPP_INFO(this->get_logger(), 
        "串口发送节点初始化完成 - 端口: %s, 波特率: %d, 帧长度: %zu字节",
        config_.port_name.c_str(), baud, FRAME_SIZE);
}

// 析构函数
SerialSenderNode::~SerialSenderNode() {
    closeSerial();
    RCLCPP_INFO(this->get_logger(), "串口发送节点已关闭");
}

// 初始化串口
bool SerialSenderNode::initializeSerial() {
    // 打开串口
    serial_fd_ = open(config_.port_name.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (serial_fd_ == -1) {
        RCLCPP_ERROR(this->get_logger(), 
            "无法打开串口 %s: %s", 
            config_.port_name.c_str(), strerror(errno));
        return false;
    }
    
    // 设置为阻塞模式
    if (fcntl(serial_fd_, F_SETFL, 0) < 0) {
        RCLCPP_ERROR(this->get_logger(), "设置串口阻塞模式失败");
        close(serial_fd_);
        return false;
    }
    
    // 配置串口参数
    return configureSerial();
}

// 配置串口参数
bool SerialSenderNode::configureSerial() {
    struct termios options;
    
    // 获取当前串口配置
    if (tcgetattr(serial_fd_, &options) != 0) {
        RCLCPP_ERROR(this->get_logger(), "获取串口属性失败");
        return false;
    }
    
    // 设置波特率
    cfsetispeed(&options, config_.baud_rate);
    cfsetospeed(&options, config_.baud_rate);
    
    // 设置数据位、停止位、校验位
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;  // 8位数据位
    
    // 无校验
    options.c_cflag &= ~PARENB;
    options.c_iflag &= ~INPCK;
    
    // 1个停止位
    options.c_cflag &= ~CSTOPB;
    
    // 关闭硬件流控
    options.c_cflag &= ~CRTSCTS;
    
    // 使能接收和本地模式
    options.c_cflag |= (CLOCAL | CREAD);
    
    // 设置为原始模式
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_iflag &= ~(INLCR | IGNCR | ICRNL);
    
    // 设置读取超时
    options.c_cc[VTIME] = 1;  // 0.1秒超时
    options.c_cc[VMIN] = 0;   // 非阻塞读取
    
    // 清空输入输出缓冲区
    tcflush(serial_fd_, TCIFLUSH);
    
    // 应用配置
    if (tcsetattr(serial_fd_, TCSANOW, &options) != 0) {
        RCLCPP_ERROR(this->get_logger(), "设置串口属性失败");
        return false;
    }
    
    return true;
}

// 关闭串口
void SerialSenderNode::closeSerial() {
    if (serial_fd_ != -1) {
        close(serial_fd_);
        serial_fd_ = -1;
    }
}

// ROS2消息回调函数
void SerialSenderNode::serialCmdCallback(const autoaim_msgs::msg::Serialcmd::SharedPtr msg) {
    // 检查串口状态
    if (serial_fd_ == -1) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), 
            *this->get_clock(), 1000, "串口未打开");
        return;
    }
    
    // 输入数据验证
    if (std::isnan(msg->detla_yaw) || std::isnan(msg->detla_pitch)) {
        RCLCPP_WARN(this->get_logger(), 
            "接收到无效数据 - Yaw: %f, Pitch: %f", 
            msg->detla_yaw, msg->detla_pitch);
        error_count_++;
        return;
    }
    
    // 范围检查（防止数据溢出）
    if (std::abs(msg->detla_yaw) > 180.0f || std::abs(msg->detla_pitch) > 180.0f) {
        RCLCPP_WARN(this->get_logger(), 
            "角度增量超出范围 [-180°, 180°] - Yaw: %.2f°, Pitch: %.2f°",
            msg->detla_yaw, msg->detla_pitch);
        error_count_++;
        return;
    }
    
    // 打包数据
    auto frame = packData(msg->detla_yaw, msg->detla_pitch);
    
    // 发送数据
    if (sendData(frame)) {
        sent_count_++;
        
        // 调试输出
        if (debug_mode_) {
            RCLCPP_INFO(this->get_logger(), 
                "发送成功 - Yaw增量: %.3f°, Pitch增量: %.3f°",
                msg->detla_yaw, msg->detla_pitch);
        }
    } else {
        error_count_++;
        RCLCPP_WARN_THROTTLE(this->get_logger(), 
            *this->get_clock(), 1000, "串口发送失败");
    }
}

// 确定符号标志位
uint8_t SerialSenderNode::determineSignFlag(float yaw, float pitch) {
    // 匹配Python版本的逻辑
    if (yaw < 0 && pitch < 0) {
        return 0x00;  // 都为负
    } else if (yaw > 0 && pitch < 0) {
        return 0x01;  // yaw正，pitch负
    } else if (yaw < 0 && pitch > 0) {
        return 0x02;  // yaw负，pitch正
    } else if (yaw > 0 && pitch > 0) {
        return 0x03;  // 都为正
    } else {
        return 0x04;  // 其他情况（包括零值）
    }
}

// 数据打包函数
std::vector<uint8_t> SerialSenderNode::packData(float delta_yaw_deg, float delta_pitch_deg, float distance) {
    std::vector<uint8_t> frame(FRAME_SIZE, 0);
    
    // 1. 设置帧头
    frame[0] = FRAME_HEADER;  // 0xFF
    
    // 2. 角度数据转换 - 注意取绝对值后再转换
    int16_t pitch_int = static_cast<int16_t>(std::round(std::abs(delta_pitch_deg) * ANGLE_SCALE));
    int16_t yaw_int = static_cast<int16_t>(std::round(std::abs(delta_yaw_deg) * ANGLE_SCALE));
    
    
    // 3. 打包Pitch数据（小端序）
    frame[1] = pitch_int & 0xFF;          // Pitch低字节
    frame[2] = (pitch_int >> 8) & 0xFF;   // Pitch高字节
    
    // 4. 打包Yaw数据（小端序）
    frame[3] = yaw_int & 0xFF;            // Yaw低字节
    frame[4] = (yaw_int >> 8) & 0xFF;     // Yaw高字节
    
    // 5. 设置符号位 - 基于原始值的正负
    frame[5] = determineSignFlag(delta_yaw_deg, delta_pitch_deg);
    
    // 6. 设置目标距离
    int distance_int = static_cast<int>(distance);
    frame[6] = 0;
    
    // 7. 导航部分全部设为0（屏蔽导航功能）
    frame[7] = 0;  // Vx
    frame[8] = 0;  // Vy
    frame[9] = 0;  // Wz
    
    // 8. 控制模式字节 - 这是最关键的修改！
    frame[10] = CONTROL_MODE_AUTOAIM;  // 使用0x08而不是0x07
    
    // 9. 设置帧尾
    frame[11] = FRAME_TAIL;  // 0xFE
    
    // 调试输出
    if (debug_mode_) {
        printDataDetails(delta_yaw_deg, delta_pitch_deg, yaw_int, pitch_int);
        printFrameHex(frame);
    }
    
    return frame;
}

// 发送数据函数
bool SerialSenderNode::sendData(const std::vector<uint8_t>& data) {
    std::lock_guard<std::mutex> lock(serial_mutex_);
    
    // 发送数据
    ssize_t bytes_written = write(serial_fd_, data.data(), data.size());
    
    if (bytes_written != static_cast<ssize_t>(data.size())) {
        if (bytes_written == -1) {
            RCLCPP_ERROR(this->get_logger(), 
                "串口写入错误: %s", strerror(errno));
        } else {
            RCLCPP_WARN(this->get_logger(), 
                "串口写入不完整: 期望 %zu 字节, 实际 %zd 字节",
                data.size(), bytes_written);
        }
        return false;
    }
    
    // 确保数据被发送
    tcdrain(serial_fd_);
    
    return true;
}

// 打印数据详情
void SerialSenderNode::printDataDetails(float yaw_deg, float pitch_deg, 
                                       int16_t yaw_int, int16_t pitch_int) {
    RCLCPP_DEBUG(this->get_logger(), 
        "数据转换详情:");
    RCLCPP_DEBUG(this->get_logger(), 
        "  输入角度 - Yaw: %.3f°, Pitch: %.3f°", 
        yaw_deg, pitch_deg);
    RCLCPP_DEBUG(this->get_logger(), 
        "  转换后整数 - Yaw: %d, Pitch: %d", 
        yaw_int, pitch_int);
    RCLCPP_DEBUG(this->get_logger(), 
        "  十六进制 - Yaw: 0x%04X, Pitch: 0x%04X", 
        static_cast<uint16_t>(yaw_int), static_cast<uint16_t>(pitch_int));
}

// 调试输出函数
void SerialSenderNode::printFrameHex(const std::vector<uint8_t>& frame) {
    std::stringstream ss;
    ss << "发送帧数据 [" << frame.size() << "字节]: ";
    for (size_t i = 0; i < frame.size(); ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') 
           << static_cast<int>(frame[i]);
        if (i < frame.size() - 1) ss << " ";
        
        // 添加字段说明
        switch(i) {
            case 0: ss << "(帧头)"; break;
            case 2: ss << "(Pitch)"; break;
            case 4: ss << "(Yaw)"; break;
            case 5: ss << "(符号)"; break;
            case 6: ss << "(距离)"; break;
            case 10: ss << "(导航符号)"; break;
            case 11: ss << "(帧尾)"; break;
        }
        if (i < frame.size() - 1) ss << " ";
    }
    RCLCPP_DEBUG(this->get_logger(), "%s", ss.str().c_str());
}

// 统计信息输出
void SerialSenderNode::printStatistics() {
    uint64_t sent = sent_count_.load();
    uint64_t errors = error_count_.load();
    uint64_t total = sent + errors;
    
    if (total > 0) {
        double success_rate = (static_cast<double>(sent) / total) * 100.0;
        RCLCPP_INFO(this->get_logger(), 
            "串口统计 - 发送: %lu, 错误: %lu, 成功率: %.1f%%",
            sent, errors, success_rate);
    }
}

// 主函数
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<SerialSenderNode>();
        
        // 输出协议信息
        RCLCPP_INFO(rclcpp::get_logger("serial_sender"), 
            "串口通信协议: 帧长%zu字节, 角度单位:度, 缩放因子:%.0f",
            SerialSenderNode::FRAME_SIZE, SerialSenderNode::ANGLE_SCALE);
        
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("serial_sender"), 
            "节点运行异常: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}