#include "image_processor.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <iostream>

namespace armor_pointer {

// 构造函数
ImageProcessor::ImageProcessor() : config_(Config()) {}
ImageProcessor::ImageProcessor(const Config& config) : config_(config) {}

// 配置加载
void ImageProcessor::Config::load_from_yaml(const std::string& yaml_path) {
    try {
        YAML::Node root = YAML::LoadFile(yaml_path);
        
        if (root["pointer"] && root["pointer"]["binary"]) {
            const auto& binary = root["pointer"]["binary"];
            binary_threshold_ratio = binary["threshold_ratio"].as<double>(1.0);
            histogram_cut_threshold = binary["histogram_cut"].as<int>(8);
            fixed_threshold = binary["fixed_threshold"].as<int>(50);
        }
        
        if (root["pointer"] && root["pointer"]["color"]) {
            enemy_color_threshold = root["pointer"]["color"]["enemy_threshold"].as<double>(30.0);
        }
        
        if (root["pointer"] && root["pointer"]["channel"]) {
            const auto& channel = root["pointer"]["channel"];
            channel_weight_r = channel["weight_r"].as<double>(0.7);
            channel_weight_g = channel["weight_g"].as<double>(0.7);
            channel_weight_b = channel["weight_b"].as<double>(0.7);
        }
        
        if (root["pointer"] && root["pointer"]["method"]) {
            const auto& method = root["pointer"]["method"];
            default_gray_method = static_cast<GrayMethod>(method["gray"].as<int>(0));
            default_binary_method = static_cast<BinaryMethod>(method["binary"].as<int>(1));
        }
    } catch (const YAML::Exception& e) {
        std::cout << "YAML加载失败，使用默认配置" << std::endl;
    }
}

// 主处理函数 - 明确指定使用的方法
cv::Mat ImageProcessor::process_roi(const cv::Mat& roi, 
                                   ArmorColor enemy_color,
                                   GrayMethod gray_method,
                                   BinaryMethod binary_method) {
    // 步骤1：灰度化
    cv::Mat gray = get_gray_image(roi, enemy_color, gray_method);
    
    // 步骤2：二值化
    cv::Mat binary = get_binary_image(gray, binary_method);
    
    return binary;
}

// 统一的灰度化接口
cv::Mat ImageProcessor::get_gray_image(const cv::Mat& input, 
                                       ArmorColor enemy_color, 
                                       GrayMethod method) {
    switch (method) {
        case GRAY_CVT:
            return gray_cvt(input);
        case GRAY_CHANNEL:
            return gray_channel(input, enemy_color);
        case GRAY_SUBTRACT:
            return gray_subtract(input, enemy_color);
        case GRAY_HSV:
            return gray_hsv(input, enemy_color);
        case GRAY_MIX:
            return gray_mix(input, enemy_color);
        default:
            return gray_cvt(input);
    }
}

// 统一的二值化接口
cv::Mat ImageProcessor::get_binary_image(const cv::Mat& gray, BinaryMethod method) {
    switch (method) {
        case BINARY_FIXED:
            return binary_fixed(gray);
        case BINARY_HISTOGRAM:
            return binary_histogram(gray);
        case BINARY_OTSU:
            return binary_otsu(gray);
        case BINARY_MEAN:
            return binary_mean(gray);
        default:
            return binary_histogram(gray);
    }
}

// ========== 灰度化方法实现 ==========

// 方法1：标准BGR转灰度
cv::Mat ImageProcessor::gray_cvt(const cv::Mat& input) {
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

// 方法2：单通道提取
cv::Mat ImageProcessor::gray_channel(const cv::Mat& input, ArmorColor color) {
    std::vector<cv::Mat> channels;
    cv::split(input, channels);
    
    switch (color) {
        case ARMOR_COLOR_BLUE:
            return channels[0];  // B通道
        case ARMOR_COLOR_RED:
            return channels[2];  // R通道
        case ARMOR_COLOR_PURPLE:
            return (channels[0] + channels[2]) / 2;  // B和R的平均
        case ARMOR_COLOR_NONE:
        default:
            return channels[1];  // G通道作为默认
    }
}

// 方法3：通道差分（专门处理过曝）
cv::Mat ImageProcessor::gray_subtract(const cv::Mat& input, ArmorColor color) {
    std::vector<cv::Mat> channels;
    cv::split(input, channels);
    
    cv::Mat result;
    switch (color) {
        case ARMOR_COLOR_BLUE: {
            // 蓝色LED：B通道 - R通道*权重
            cv::Mat weighted_r = channels[2] * config_.channel_weight_r;
            result = channels[0] - weighted_r;
            break;
        }
        case ARMOR_COLOR_RED: {
            // 红色LED：R通道 - B通道*权重
            cv::Mat weighted_b = channels[0] * config_.channel_weight_b;
            result = channels[2] - weighted_b;
            break;
        }
        case ARMOR_COLOR_PURPLE: {
            // 紫色LED：(R+B)/2 - G通道*权重
            cv::Mat rb_avg = (channels[0] + channels[2]) / 2;
            cv::Mat weighted_g = channels[1] * config_.channel_weight_g;
            result = rb_avg - weighted_g;
            break;
        }
        case ARMOR_COLOR_NONE:
        default:
            return gray_cvt(input);
    }
    
    // 处理负值并归一化
    result = cv::max(result, 0);
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX, CV_8U);
    return result;
}


// 方法4：HSV色相过滤
cv::Mat ImageProcessor::gray_hsv(const cv::Mat& input, ArmorColor color) {
    cv::Mat hsv;
    cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);
    
    cv::Mat mask;
    switch (color) {
        case ARMOR_COLOR_BLUE: {
            // 蓝色色相范围：100-124
            cv::inRange(hsv, cv::Scalar(100, 43, 46), cv::Scalar(124, 255, 255), mask);
            break;
        }
        case ARMOR_COLOR_RED: {
            // 注意这里加了花括号创建独立作用域
            // 红色色相范围：0-10 或 156-180
            cv::Mat mask1, mask2;
            cv::inRange(hsv, cv::Scalar(0, 43, 46), cv::Scalar(10, 255, 255), mask1);
            cv::inRange(hsv, cv::Scalar(156, 43, 46), cv::Scalar(180, 255, 255), mask2);
            mask = mask1 | mask2;
            break;
        }
        case ARMOR_COLOR_PURPLE: {
            // 紫色色相范围：125-155
            cv::inRange(hsv, cv::Scalar(125, 43, 46), cv::Scalar(155, 255, 255), mask);
            break;
        }
        case ARMOR_COLOR_NONE:
        default: {
            // 处理所有其他情况，返回标准灰度图
            return gray_cvt(input);
        }
    }
    
    // 提取V通道（亮度）并应用掩码
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);
    cv::Mat gray;
    cv::bitwise_and(hsv_channels[2], mask, gray);
    
    return gray;
}

// 方法5：混合方法（根据颜色选择最佳方法）
cv::Mat ImageProcessor::gray_mix(const cv::Mat& input, ArmorColor color) {
    // 紫色用HSV效果好，红蓝用通道提取效果好
    switch (color) {
        case ARMOR_COLOR_PURPLE:
            return gray_hsv(input, color);
        case ARMOR_COLOR_BLUE:
        case ARMOR_COLOR_RED:
            return gray_channel(input, color);
        case ARMOR_COLOR_NONE:
        default:
            return gray_cvt(input);
    }
}

// ========== 二值化方法实现 ==========

// 方法1：固定阈值
cv::Mat ImageProcessor::binary_fixed(const cv::Mat& gray) {
    cv::Mat binary;
    cv::threshold(gray, binary, config_.fixed_threshold, 255, cv::THRESH_BINARY);
    return binary;
}

// 方法2：直方图分析
cv::Mat ImageProcessor::binary_histogram(const cv::Mat& gray) {
    cv::Mat binary;
    int threshold = calculate_histogram_threshold(gray);
    cv::threshold(gray, binary, threshold, 255, cv::THRESH_BINARY);
    return binary;
}

// 方法3：OTSU自动阈值
cv::Mat ImageProcessor::binary_otsu(const cv::Mat& gray) {
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return binary;
}

// 方法4：均值阈值
cv::Mat ImageProcessor::binary_mean(const cv::Mat& gray) {
    cv::Mat binary;
    cv::Scalar mean = cv::mean(gray);
    int threshold = static_cast<int>(mean[0] + config_.binary_threshold_ratio);
    threshold = std::clamp(threshold, 0, 255);
    cv::threshold(gray, binary, threshold, 255, cv::THRESH_BINARY);
    return binary;
}

// 直方图阈值计算辅助函数
int ImageProcessor::calculate_histogram_threshold(const cv::Mat& gray) {
    // 计算直方图
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    
    // 归一化直方图
    cv::normalize(hist, hist, 0, 512, cv::NORM_MINMAX);
    
    // 在特定范围内查找合适的阈值
    int threshold = 10;  // 默认值
    int search_start = 80;
    int search_end = 10;
    
    for (int i = search_start; i > search_end; i--) {
        if (hist.at<float>(i) >= config_.histogram_cut_threshold) {
            threshold = i;
            break;
        }
    }
    
    // 应用偏移量
    threshold += static_cast<int>(config_.binary_threshold_ratio);
    
    // 限制在合理范围内
    return std::clamp(threshold, 10, 100);
}

// 颜色检测函数保持不变
ArmorColor ImageProcessor::detect_armor_color(const cv::Mat& roi, 
                                             const std::pair<Lightbar, Lightbar>& lightbar_pair) {
    // 转换到HSV色彩空间进行颜色识别
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    
    // 获取两个灯条的区域
    cv::Rect rect1 = lightbar_pair.first.rect.boundingRect();
    cv::Rect rect2 = lightbar_pair.second.rect.boundingRect();
    
    // 扩展检测区域
    rect1.x = std::max(0, rect1.x - static_cast<int>(rect1.width * 0.5));
    rect1.y = std::max(0, rect1.y - static_cast<int>(rect1.height * 0.25));
    rect1.width = std::min(static_cast<int>(rect1.width * 2), roi.cols - rect1.x);
    rect1.height = std::min(static_cast<int>(rect1.height * 1.5), roi.rows - rect1.y);
    
    rect2.x = std::max(0, rect2.x - static_cast<int>(rect2.width * 0.5));
    rect2.y = std::max(0, rect2.y - static_cast<int>(rect2.height * 0.25));
    rect2.width = std::min(static_cast<int>(rect2.width * 2), roi.cols - rect2.x);
    rect2.height = std::min(static_cast<int>(rect2.height * 1.5), roi.rows - rect2.y);
    
    // 统计颜色像素
    int red_count = 0, blue_count = 0, purple_count = 0;
    
    // 处理第一个灯条区域
    cv::Mat roi1 = hsv(rect1);
    for (int y = 0; y < roi1.rows; ++y) {
        for (int x = 0; x < roi1.cols; ++x) {
            cv::Vec3b pixel = roi1.at<cv::Vec3b>(y, x);
            int h = pixel[0];
            int s = pixel[1];
            int v = pixel[2];
            
            // 过滤低饱和度和极端亮度
            if (s < 43 || v < 46 || v > 240) continue;
            
            // 根据色调判断颜色
            if (h >= 100 && h <= 124) {
                blue_count++;
            } else if ((h >= 156 && h <= 180) || (h >= 0 && h <= 10)) {
                red_count++;
            } else if (h >= 125 && h <= 155) {
                purple_count++;
            }
        }
    }
    
    // 处理第二个灯条区域（代码相同）
    cv::Mat roi2 = hsv(rect2);
    for (int y = 0; y < roi2.rows; ++y) {
        for (int x = 0; x < roi2.cols; ++x) {
            cv::Vec3b pixel = roi2.at<cv::Vec3b>(y, x);
            int h = pixel[0];
            int s = pixel[1];
            int v = pixel[2];
            
            if (s < 43 || v < 46 || v > 240) continue;
            
            if (h >= 100 && h <= 124) {
                blue_count++;
            } else if ((h >= 156 && h <= 180) || (h >= 0 && h <= 10)) {
                red_count++;
            } else if (h >= 125 && h <= 155) {
                purple_count++;
            }
        }
    }
    
    // 判断主要颜色
    if (purple_count * 2 > red_count && purple_count * 2 > blue_count) {
        return ARMOR_COLOR_PURPLE;
    } else if (red_count > blue_count + purple_count && red_count > 0) {
        return ARMOR_COLOR_RED;
    } else if (blue_count > red_count + purple_count) {
        return ARMOR_COLOR_BLUE;
    } else {
        return ARMOR_COLOR_NONE;
    }
}

} // namespace armor_pointer