#ifndef UTILS__DRAW_TOOLS_HPP
#define UTILS__DRAW_TOOLS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

namespace utils
{

// 定义标准颜色方案
namespace colors {
    const cv::Scalar GREEN = cv::Scalar(0, 255, 0);       // 主要检测框颜色
    const cv::Scalar BLUE = cv::Scalar(255, 0, 0);        // 蓝色装甲板
    const cv::Scalar RED = cv::Scalar(0, 0, 255);         // 红色装甲板
    const cv::Scalar YELLOW = cv::Scalar(0, 255, 255);    // 警告或特殊标记
    const cv::Scalar WHITE = cv::Scalar(255, 255, 255);   // 文字颜色
    const cv::Scalar BLACK = cv::Scalar(0, 0, 0);         // 背景色
    const cv::Scalar CYAN = cv::Scalar(255, 255, 0);      // 辅助信息
}

inline void draw_label(cv::Mat & img, const std::string & text, const cv::Point & position,
                      const cv::Scalar & color = {0, 0, 0}, double scale = 0.5)
{
    int baseline;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    
    cv::rectangle(img, 
                  position + cv::Point(0, -text_size.height - 2),
                  position + cv::Point(text_size.width, baseline + 2),
                  color, cv::FILLED);
    
    cv::putText(img, text, position, cv::FONT_HERSHEY_SIMPLEX, scale, {255, 255, 255}, 1);
}

// 绘制精确的四边形轮廓（连接四个角点）
inline void draw_quadrangle(cv::Mat& img, const std::vector<cv::Point2f>& points,
                           const cv::Scalar& color = colors::GREEN, int thickness = 2)
{
    if (points.size() != 4) return;
    
    // 将浮点坐标转换为整数坐标
    std::vector<cv::Point> int_points;
    for (const auto& p : points) {
        int_points.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
    }
    
    // 绘制四条边
    for (size_t i = 0; i < 4; ++i) {
        cv::line(img, int_points[i], int_points[(i + 1) % 4], color, thickness, cv::LINE_AA);
    }
}

// 绘制带角点标记的四边形
inline void draw_quadrangle_with_corners(cv::Mat& img, const std::vector<cv::Point2f>& points,
                                        const cv::Scalar& color = colors::GREEN, 
                                        int line_thickness = 2, int corner_radius = 4)
{
    if (points.size() != 4) return;
    
    // 绘制四边形轮廓
    draw_quadrangle(img, points, color, line_thickness);
    
    // 在每个角点绘制实心圆
    for (const auto& p : points) {
        cv::circle(img, cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)), 
                  corner_radius, color, -1, cv::LINE_AA);
    }
}

// 专业的标签绘制（类似第二张图的效果）
inline void draw_detection_label(cv::Mat& img, const std::string& text, 
                                const cv::Point2f& position,
                                const cv::Scalar& color = colors::GREEN,
                                double font_scale = 0.7, int thickness = 2)
{
    // 使用更清晰的字体
    int font_face = cv::FONT_HERSHEY_DUPLEX;
    
    // 获取文字大小
    int baseline;
    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
    
    // 计算文字位置（稍微偏移以避免遮挡）
    cv::Point text_org(static_cast<int>(position.x), 
                      static_cast<int>(position.y - 5));
    
    // 如果文字会超出图像顶部，放到下方
    if (text_org.y - text_size.height < 0) {
        text_org.y = static_cast<int>(position.y + text_size.height + 5);
    }
    
    // 绘制半透明背景（可选，提高可读性）
    cv::Rect bg_rect(text_org.x - 2, text_org.y - text_size.height - 2,
                    text_size.width + 4, text_size.height + baseline + 4);
    
    // 确保背景矩形在图像范围内
    bg_rect &= cv::Rect(0, 0, img.cols, img.rows);
    if (bg_rect.area() > 0) {
        cv::Mat roi = img(bg_rect);
        cv::Mat bg_color(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
        cv::addWeighted(roi, 0.3, bg_color, 0.7, 0, roi);
    }
    
    // 绘制文字
    cv::putText(img, text, text_org, font_face, font_scale, color, thickness, cv::LINE_AA);
}

// 绘制装甲板检测结果（模仿第二张图的风格）
inline void draw_armor_detection(cv::Mat& img, const std::vector<cv::Point2f>& corners,
                                const std::string& class_name, int class_id,
                                float confidence, const std::string& color_str = "blue")
{
    // 根据颜色字符串选择绘制颜色
    cv::Scalar draw_color = colors::GREEN;  // 默认绿色
    
    // 绘制四边形轮廓和角点
    draw_quadrangle_with_corners(img, corners, draw_color, 2, 3);
    
    // 计算标签位置（四边形的顶部中心）
    cv::Point2f top_center;
    if (corners.size() == 4) {
        top_center.x = (corners[0].x + corners[1].x) / 2;
        top_center.y = std::min(corners[0].y, corners[1].y);
    }
    
    // 构建标签文本
    std::stringstream label;
    label << std::fixed << std::setprecision(2);
    label << confidence << " " << color_str << ", " << class_name << " ,small";
    
    // 绘制标签
    draw_detection_label(img, label.str(), top_center, draw_color);
}

// 在图像左下角添加帧号标记（类似[494]）
inline void draw_frame_number(cv::Mat& img, int frame_num, 
                             const cv::Point& pos = cv::Point(10, -1))
{
    std::stringstream ss;
    ss << "[" << frame_num << "]";
    
    // 如果y坐标为-1，自动放在左下角
    cv::Point actual_pos = pos;
    if (pos.y == -1) {
        actual_pos.y = img.rows - 10;
    }
    
    cv::putText(img, ss.str(), actual_pos, cv::FONT_HERSHEY_SIMPLEX, 
               0.6, colors::WHITE, 1, cv::LINE_AA);
}

// 主要的可视化函数 - 用于节点中
inline void draw_armor_with_corners(cv::Mat& img, const cv::Rect& box, 
                                   const std::string& label,
                                   const std::vector<cv::Point2f>& corners,
                                   const cv::Scalar& color = colors::GREEN)
{
    if (corners.size() == 4) {
        // 绘制精确的四边形
        draw_quadrangle_with_corners(img, corners, color, 2, 3);
        
        // 计算标签位置
        cv::Point2f label_pos;
        label_pos.x = (corners[0].x + corners[1].x) / 2;
        label_pos.y = std::min(corners[0].y, corners[1].y);
        
        // 绘制标签
        draw_detection_label(img, label, label_pos, color, 0.6, 2);
    } else {
        // 降级到矩形框
        cv::rectangle(img, box, color, 2, cv::LINE_AA);
        cv::Point label_pos(box.x, box.y - 5);
        draw_detection_label(img, label, cv::Point2f(label_pos.x, label_pos.y), color);
    }
}

// 简化版本的装甲板绘制（仅矩形框）
inline void draw_armor(cv::Mat& img, const cv::Rect& box, const std::string& label,
                      const cv::Scalar& color = colors::GREEN)
{
    cv::rectangle(img, box, color, 2, cv::LINE_AA);
    draw_detection_label(img, label, cv::Point2f(box.x, box.y), color);
}

// 绘制十字准星
inline void draw_crosshair(cv::Mat& img, const cv::Point& center, 
                          int size = 15, const cv::Scalar& color = colors::YELLOW,
                          int thickness = 1)
{
    // 绘制十字线
    cv::line(img, center - cv::Point(size, 0), center + cv::Point(size, 0), 
            color, thickness, cv::LINE_AA);
    cv::line(img, center - cv::Point(0, size), center + cv::Point(0, size), 
            color, thickness, cv::LINE_AA);
    
    // 中心点
    cv::circle(img, center, 2, color, -1, cv::LINE_AA);
}

// 绘制FPS和检测信息
inline void draw_fps(cv::Mat& img, double fps, int detections = -1,
                    const cv::Point& pos = cv::Point(10, 30))
{
    std::stringstream ss;
    ss << "FPS: " << std::fixed << std::setprecision(1) << fps;
    
    if (detections >= 0) {
        ss << " | Detections: " << detections;
    }
    
    // 使用更醒目的颜色
    cv::putText(img, ss.str(), pos, cv::FONT_HERSHEY_SIMPLEX, 
               0.7, colors::CYAN, 2, cv::LINE_AA);
}

// 保留原有的辅助函数
inline void draw_corners(cv::Mat& img, const std::vector<cv::Point2f>& points,
                        const cv::Scalar& color = colors::GREEN, int radius = 3)
{
    for (const auto& p : points) {
        cv::circle(img, cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)), 
                  radius, color, -1, cv::LINE_AA);
    }
}

inline void draw_polygon(cv::Mat& img, const std::vector<cv::Point2f>& points,
                        const cv::Scalar& color = colors::GREEN, int thickness = 2)
{
    if (points.size() < 2) return;
    
    for (size_t i = 0; i < points.size(); ++i) {
        size_t next = (i + 1) % points.size();
        cv::line(img, 
                cv::Point(static_cast<int>(points[i].x), static_cast<int>(points[i].y)),
                cv::Point(static_cast<int>(points[next].x), static_cast<int>(points[next].y)),
                color, thickness, cv::LINE_AA);
    }
}

inline cv::Mat create_grid(const std::vector<cv::Mat>& images, int cols = 4, int padding = 2)
{
    if (images.empty()) {
        return cv::Mat::zeros(200, 400, CV_8UC3);
    }
    
    if (images.size() == 1) {
        return images[0];
    }
    
    int rows = (images.size() + cols - 1) / cols;
    
    int max_w = 0, max_h = 0;
    for (const auto& img : images) {
        max_w = std::max(max_w, img.cols);
        max_h = std::max(max_h, img.rows);
    }
    
    cv::Mat grid(rows * (max_h + padding), cols * (max_w + padding), 
                 CV_8UC3, cv::Scalar(0, 0, 0));
    
    for (size_t i = 0; i < images.size(); ++i) {
        int r = i / cols;
        int c = i % cols;
        
        cv::Rect roi(c * (max_w + padding) + padding/2, 
                     r * (max_h + padding) + padding/2,
                     images[i].cols, images[i].rows);
        
        if (roi.x + roi.width <= grid.cols && roi.y + roi.height <= grid.rows) {
            images[i].copyTo(grid(roi));
        }
    }
    
    return grid;
}

}  // namespace utils

#endif  // UTILS__DRAW_TOOLS_HPP