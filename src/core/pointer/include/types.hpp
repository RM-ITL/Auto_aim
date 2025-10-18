#ifndef ARMOR_POINTER_TYPES_HPP
#define ARMOR_POINTER_TYPES_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace armor_pointer {

// 使用原有的枚举定义
enum ArmorColor {
    ARMOR_COLOR_NONE = 0,
    ARMOR_COLOR_BLUE = 1,
    ARMOR_COLOR_RED = 2,
    ARMOR_COLOR_PURPLE = 3
};

// 灯条结构 - 保持原有设计
struct Lightbar {
    cv::RotatedRect rect;
    std::vector<cv::Point> contour;
    double angle;
    double length;
};

// 点对 - 添加构造函数
struct PointPair {
    cv::Point2f point_up;
    cv::Point2f point_down;
    
    // 默认构造函数
    PointPair() = default;
    
    // 带参数的构造函数
    PointPair(const cv::Point2f& up, const cv::Point2f& down) 
        : point_up(up), point_down(down) {}
};

// 处理结果
struct PointerResult {
    std::vector<cv::Point2f> four_points;  // 四个角点
    ArmorColor detected_color;             // 检测到的颜色
    bool success;                          // 是否成功
    double confidence;                     // 置信度
    cv::Mat binary_image;                  // 新增：二值化图像（用于调试）
};

} // namespace armor_pointer

#endif