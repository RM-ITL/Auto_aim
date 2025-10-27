#ifndef ARMOR_POINTER_LIGHTBAR_DETECTOR_HPP
#define ARMOR_POINTER_LIGHTBAR_DETECTOR_HPP

#include "types.hpp"
#include "image_processor.hpp"

namespace armor_pointer {

class LightbarDetector {
public:
    struct Config {
        // ROI扩展参数（减小默认值）
        double roi_extend_width_ratio = 1.5;   // 从1.4减小到1.2
        double roi_extend_height_ratio = 1.5;  // 从1.8减小到1.3
        
        // 灯条筛选参数
        double lightbar_min_ratio = 2.0;
        double lightbar_max_ratio = 8.0;
        double lightbar_min_area = 50.0;
        double min_overlap_ratio = 0.3;  // 新增：灯条与检测框的最小重叠比例
        
        // 灯条配对参数
        double max_angle_diff = 20.0;
        double max_length_ratio = 2.0;
        bool require_center_in_box = true;  // 新增：要求灯条对中心在检测框内
        
        // 端点定位参数
        double point_refine_radius_ratio = 0.1;
        
        void load_from_yaml(const std::string& yaml_path);
    };
    
    // 构造函数
    explicit LightbarDetector(const Config& config);
    LightbarDetector();
    
    // 主要接口 - 从检测框生成四点
    PointerResult process(const cv::Mat& image, 
                         const cv::Rect& detection_box,
                         ArmorColor enemy_color);
    
private:
    Config config_;
    ImageProcessor image_processor_;
    
    // 新增：用于存储检测框相关信息
    cv::Rect original_detection_box_;     // 原始检测框（相对于整图）
    cv::Rect detection_box_in_roi_;       // 检测框在ROI中的位置
    cv::Point roi_offset_;                // ROI相对于原图的偏移
    
    // 内部处理步骤
    cv::Rect expand_roi(const cv::Rect& box, const cv::Size& image_size);
    
    // 新增：带约束的灯条查找
    std::vector<Lightbar> find_lightbars_with_constraint(const cv::Mat& binary);
    bool is_lightbar_valid_for_detection(const Lightbar& lightbar);
    
    // 新增：带约束的灯条匹配
    std::pair<Lightbar, Lightbar> match_lightbars_with_constraint(const std::vector<Lightbar>& lightbars);
    
    // 新增：计算置信度
    double calculate_confidence(const std::pair<Lightbar, Lightbar>& matched_pair);
    
    PointPair refine_endpoints(const Lightbar& lightbar, const cv::Mat& gray);
    std::vector<cv::Point2f> generate_four_points(const PointPair& pp1, const PointPair& pp2);
};

} // namespace armor_pointer

#endif