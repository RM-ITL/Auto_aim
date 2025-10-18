#ifndef ARMOR_POINTER_IMAGE_PROCESSOR_HPP
#define ARMOR_POINTER_IMAGE_PROCESSOR_HPP

#include "types.hpp"
#include <string>

namespace armor_pointer {

// 灰度化方法枚举
enum GrayMethod {
    GRAY_CVT = 0,        // 标准BGR2GRAY转换
    GRAY_CHANNEL = 1,    // 单通道提取
    GRAY_SUBTRACT = 2,   // 通道差分
    GRAY_HSV = 3,        // HSV处理
    GRAY_MIX = 4         // 混合方法
};

// 二值化方法枚举
enum BinaryMethod {
    BINARY_FIXED = 0,     // 固定阈值
    BINARY_HISTOGRAM = 1, // 直方图分析
    BINARY_OTSU = 2,      // OTSU算法
    BINARY_MEAN = 3       // 均值阈值
};

class ImageProcessor {
public:
    struct Config {
        // 二值化参数
        double binary_threshold_ratio = 1.0;
        int histogram_cut_threshold = 8;
        int fixed_threshold = 50;           // 固定阈值默认值
        
        // 颜色判断参数
        double enemy_color_threshold = 30.0;
        
        // 通道处理参数
        double channel_weight_r = 0.7;      // 红色通道权重
        double channel_weight_g = 0.7;      // 绿色通道权重
        double channel_weight_b = 0.7;      // 蓝色通道权重
        
        // 默认处理方法
        GrayMethod default_gray_method = GRAY_CVT;
        BinaryMethod default_binary_method = BINARY_HISTOGRAM;
        
        void load_from_yaml(const std::string& yaml_path);
    };
    
    explicit ImageProcessor(const Config& config);
    ImageProcessor();
    
    // 主处理接口 - 使用指定的方法
    cv::Mat process_roi(const cv::Mat& roi, 
                        ArmorColor enemy_color,
                        GrayMethod gray_method = GRAY_CVT,
                        BinaryMethod binary_method = BINARY_HISTOGRAM);
    
    // 独立的灰度化接口
    cv::Mat get_gray_image(const cv::Mat& input, 
                           ArmorColor enemy_color, 
                           GrayMethod method);
    
    // 独立的二值化接口
    cv::Mat get_binary_image(const cv::Mat& gray, 
                             BinaryMethod method);
    
    // 颜色检测
    ArmorColor detect_armor_color(const cv::Mat& roi, 
                                 const std::pair<Lightbar, Lightbar>& lightbar_pair);
    
private:
    Config config_;
    
    // 各种灰度化实现
    cv::Mat gray_cvt(const cv::Mat& input);
    cv::Mat gray_channel(const cv::Mat& input, ArmorColor color);
    cv::Mat gray_subtract(const cv::Mat& input, ArmorColor color);
    cv::Mat gray_hsv(const cv::Mat& input, ArmorColor color);
    cv::Mat gray_mix(const cv::Mat& input, ArmorColor color);
    
    // 各种二值化实现
    cv::Mat binary_fixed(const cv::Mat& gray);
    cv::Mat binary_histogram(const cv::Mat& gray);
    cv::Mat binary_otsu(const cv::Mat& gray);
    cv::Mat binary_mean(const cv::Mat& gray);
    
    // 辅助函数
    int calculate_histogram_threshold(const cv::Mat& gray);
};

} // namespace armor_pointer

#endif