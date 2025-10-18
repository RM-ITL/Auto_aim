#ifndef ROBOT_STATUS_UTILS_HPP
#define ROBOT_STATUS_UTILS_HPP

#include "autoaim_msgs/conversion/robot_status.hpp"  // 替换为你的实际包名

namespace robot_status_utils {

// 为了向后兼容，重新定义枚举类
enum class ShootMode : uint8_t {
    IDLE = your_package::msg::RobotStatus::SHOOT_MODE_IDLE,
    TRACKING = your_package::msg::RobotStatus::SHOOT_MODE_TRACKING,
    SHOOT_NOW = your_package::msg::RobotStatus::SHOOT_MODE_SHOOT_NOW
};

enum class ProgramMode : uint8_t {
    AUTOAIM = your_package::msg::RobotStatus::PROGRAM_MODE_AUTOAIM,
    MANUAL = your_package::msg::RobotStatus::PROGRAM_MODE_MANUAL,
    ENERGY_HIT = your_package::msg::RobotStatus::PROGRAM_MODE_ENERGY_HIT,
    NOT_RECEIVED = your_package::msg::RobotStatus::PROGRAM_MODE_NOT_RECEIVED,
    ENERGY_DISTURB = your_package::msg::RobotStatus::PROGRAM_MODE_ENERGY_DISTURB
};

enum class EnemyColor : uint8_t {
    BLUE = your_package::msg::RobotStatus::ENEMY_COLOR_BLUE,
    RED = your_package::msg::RobotStatus::ENEMY_COLOR_RED
};

enum class ArmorColor : uint8_t {
    BLUE = your_package::msg::RobotStatus::ARMOR_COLOR_BLUE,
    RED = your_package::msg::RobotStatus::ARMOR_COLOR_RED,
    GRAY = your_package::msg::RobotStatus::ARMOR_COLOR_GRAY,
    PURPLE = your_package::msg::RobotStatus::ARMOR_COLOR_PURPLE
};

// 类型转换函数
inline uint8_t to_msg_value(ShootMode mode) {
    return static_cast<uint8_t>(mode);
}

inline uint8_t to_msg_value(ProgramMode mode) {
    return static_cast<uint8_t>(mode);
}

inline uint8_t to_msg_value(EnemyColor color) {
    return static_cast<uint8_t>(color);
}

inline uint8_t to_msg_value(ArmorColor color) {
    return static_cast<uint8_t>(color);
}

inline ShootMode from_msg_value_shoot_mode(uint8_t value) {
    return static_cast<ShootMode>(value);
}

inline ProgramMode from_msg_value_program_mode(uint8_t value) {
    return static_cast<ProgramMode>(value);
}

inline EnemyColor from_msg_value_enemy_color(uint8_t value) {
    return static_cast<EnemyColor>(value);
}

inline ArmorColor from_msg_value_armor_color(uint8_t value) {
    return static_cast<ArmorColor>(value);
}

// 便利函数：创建默认的RobotStatus消息
inline your_package::msg::RobotStatus create_default_robot_status() {
    your_package::msg::RobotStatus status;
    status.program_mode = to_msg_value(ProgramMode::NOT_RECEIVED);
    status.enemy_color = to_msg_value(EnemyColor::BLUE);
    status.yaw_compensate = 0.0;
    status.pitch_compensate = 0.0;
    status.bullet_speed = 0.0;
    status.last_shoot_aim_id = 0;
    status.latency_cmd_to_fire = 0;
    return status;
}

// 便利函数：设置程序模式
inline void set_program_mode(your_package::msg::RobotStatus& status, ProgramMode mode) {
    status.program_mode = to_msg_value(mode);
}

// 便利函数：设置敌人颜色
inline void set_enemy_color(your_package::msg::RobotStatus& status, EnemyColor color) {
    status.enemy_color = to_msg_value(color);
}

// 便利函数：获取程序模式
inline ProgramMode get_program_mode(const your_package::msg::RobotStatus& status) {
    return from_msg_value_program_mode(status.program_mode);
}

// 便利函数：获取敌人颜色
inline EnemyColor get_enemy_color(const your_package::msg::RobotStatus& status) {
    return from_msg_value_enemy_color(status.enemy_color);
}

// 便利函数：转换为字符串（用于调试）
inline std::string to_string(ProgramMode mode) {
    switch (mode) {
        case ProgramMode::AUTOAIM: return "AUTOAIM";
        case ProgramMode::MANUAL: return "MANUAL";
        case ProgramMode::ENERGY_HIT: return "ENERGY_HIT";
        case ProgramMode::NOT_RECEIVED: return "NOT_RECEIVED";
        case ProgramMode::ENERGY_DISTURB: return "ENERGY_DISTURB";
        default: return "UNKNOWN";
    }
}

inline std::string to_string(EnemyColor color) {
    switch (color) {
        case EnemyColor::BLUE: return "BLUE";
        case EnemyColor::RED: return "RED";
        default: return "UNKNOWN";
    }
}

inline std::string to_string(ShootMode mode) {
    switch (mode) {
        case ShootMode::IDLE: return "IDLE";
        case ShootMode::TRACKING: return "TRACKING";
        case ShootMode::SHOOT_NOW: return "SHOOT_NOW";
        default: return "UNKNOWN";
    }
}

inline std::string to_string(ArmorColor color) {
    switch (color) {
        case ArmorColor::BLUE: return "BLUE";
        case ArmorColor::RED: return "RED";
        case ArmorColor::GRAY: return "GRAY";
        case ArmorColor::PURPLE: return "PURPLE";
        default: return "UNKNOWN";
    }
}

} // namespace robot_status_utils

#endif /* ROBOT_STATUS_UTILS_HPP */