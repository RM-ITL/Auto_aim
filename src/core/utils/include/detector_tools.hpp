#ifndef ARMOR_UTILS_HPP
#define ARMOR_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <filesystem>

namespace utils {

// 加载配置文件
inline YAML::Node load_config(const std::string& config_path = "") {
    std::string path = config_path;
    
    if (path.empty()) {
        // 默认配置文件路径
        std::string project_root = "/home/guo/ITL_sentry_auto";
        path = project_root + "/src/config/robomaster_vision_config.yaml";
        
        // 如果默认路径不存在，尝试当前目录
        if (!std::filesystem::exists(path)) {
            path = std::filesystem::current_path().string() + "/config/robomaster_vision_config.yaml";
        }
    }
    
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Config file not found: " + path);
    }
    
    return YAML::LoadFile(path);
}

// 根据类名生成颜色
inline cv::Scalar name_to_color(const std::string& name) {
    unsigned int hash = 0;
    for (char c : name) {
        hash = hash * 31 + c;
    }
    return cv::Scalar(hash & 0xFF, (hash >> 8) & 0xFF, (hash >> 16) & 0xFF);
}

// Sigmoid激活函数
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// 计算IoU (Intersection over Union)
inline float compute_iou(const cv::Rect& box1, const cv::Rect& box2) {
    int x_left = std::max(box1.x, box2.x);
    int y_top = std::max(box1.y, box2.y);
    int x_right = std::min(box1.x + box1.width, box2.x + box2.width);
    int y_bottom = std::min(box1.y + box1.height, box2.y + box2.height);
    
    if (x_right <= x_left || y_bottom <= y_top) {
        return 0.0f;
    }
    
    float intersection_area = static_cast<float>((x_right - x_left) * (y_bottom - y_top));
    float box1_area = static_cast<float>(box1.width * box1.height);
    float box2_area = static_cast<float>(box2.width * box2.height);
    float union_area = box1_area + box2_area - intersection_area;
    
    return union_area > 0 ? intersection_area / union_area : 0.0f;
}

// 字符串类名转数字
inline int string_to_armor_number(const std::string& class_name) {
    if (class_name == "1") return 1;
    else if (class_name == "2") return 2;
    else if (class_name == "3") return 3;
    else if (class_name == "4") return 4;
    else if (class_name == "Sentry") return -1;
    else return 0;  // Unknown
}

} // namespace utils

#endif // ARMOR_UTILS_HPP
