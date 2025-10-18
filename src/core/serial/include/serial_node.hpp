// serial_node.hpp - 修正版本
#ifndef SERIAL_SENDER_NODE_HPP
#define SERIAL_SENDER_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <autoaim_msgs/msg/serialcmd.hpp>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>

class SerialSenderNode : public rclcpp::Node {
public:
    SerialSenderNode();
    ~SerialSenderNode();
    
    static constexpr size_t FRAME_SIZE = 12;
    static constexpr float ANGLE_SCALE = 1000.0f;

private:
    // 串口通信协议常量
    static constexpr uint8_t FRAME_HEADER = 0xFF;
    static constexpr uint8_t FRAME_TAIL = 0xFE;
    
    // 控制模式字节 - 这是关键修改点！
    static constexpr uint8_t CONTROL_MODE_AUTOAIM = 0x08;  // 匹配Python版本
    
    // 符号位定义
    enum SignFlag {
        SIGN_ALL_NEGATIVE = 0x00,    // pitch和yaw都为负
        SIGN_YAW_POSITIVE = 0x01,    // yaw为正，pitch为负
        SIGN_PITCH_POSITIVE = 0x02,  // pitch为正，yaw为负
        SIGN_ALL_POSITIVE = 0x03,    // 都为正
        SIGN_NONE = 0x04             // 无效/零值
    };
    
    struct SerialConfig {
        std::string port_name = "/dev/ttyUSB0";  // 修改为匹配Python版本
        int baud_rate = B115200;
        int data_bits = 8;
        int stop_bits = 1;
        char parity = 'N';
    };
    
    // 初始化函数
    bool initializeSerial();
    bool configureSerial();
    void closeSerial();
    
    // ROS2回调函数
    void serialCmdCallback(const autoaim_msgs::msg::Serialcmd::SharedPtr msg);
    
    // 数据打包与发送函数
    std::vector<uint8_t> packData(float delta_yaw_deg, float delta_pitch_deg, float distance = 0.0f);
    bool sendData(const std::vector<uint8_t>& data);
    
    // 辅助函数
    uint8_t determineSignFlag(float yaw, float pitch);
    void printFrameHex(const std::vector<uint8_t>& frame);
    void printDataDetails(float yaw_deg, float pitch_deg, int16_t yaw_int, int16_t pitch_int);
    
    // ROS2订阅器
    rclcpp::Subscription<autoaim_msgs::msg::Serialcmd>::SharedPtr serial_cmd_sub_;
    
    // 串口相关成员
    int serial_fd_;
    SerialConfig config_;
    std::mutex serial_mutex_;
    
    // 统计信息
    std::atomic<uint64_t> sent_count_{0};
    std::atomic<uint64_t> error_count_{0};
    
    // 定时器
    rclcpp::TimerBase::SharedPtr stats_timer_;
    void printStatistics();
    
    bool debug_mode_ = false;
};

#endif