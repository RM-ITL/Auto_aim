#include "lightbar_detector.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <iostream>  // 添加用于调试输出

namespace armor_pointer {

// 默认构造函数实现
LightbarDetector::LightbarDetector() 
    : config_(Config()), image_processor_(ImageProcessor::Config()) {}

// 带参数构造函数实现
LightbarDetector::LightbarDetector(const Config& config) 
    : config_(config), image_processor_(ImageProcessor::Config()) {}

void LightbarDetector::Config::load_from_yaml(const std::string& yaml_path) {
    try {
        YAML::Node root = YAML::LoadFile(yaml_path);
        
        if (root["pointer"]) {
            const auto& pointer = root["pointer"];
            
            if (pointer["roi"]) {
                // 减小扩展比例，因为只需要包含灯条
                roi_extend_width_ratio = pointer["roi"]["extend_width"].as<double>(1.2);  // 从1.4减小到1.2
                roi_extend_height_ratio = pointer["roi"]["extend_height"].as<double>(1.3); // 从1.8减小到1.3
            }
            
            if (pointer["lightbar"]) {
                const auto& lb = pointer["lightbar"];
                lightbar_min_ratio = lb["min_ratio"].as<double>(2.0);
                lightbar_max_ratio = lb["max_ratio"].as<double>(8.0);
                lightbar_min_area = lb["min_area"].as<double>(50.0);
                // 新增：灯条必须与检测框的最小重叠比例
                min_overlap_ratio = lb["min_overlap_ratio"].as<double>(0.3);
            }
            
            if (pointer["match"]) {
                const auto& match = pointer["match"];
                max_angle_diff = match["max_angle_diff"].as<double>(20.0);
                max_length_ratio = match["max_length_ratio"].as<double>(2.0);
                // 新增：灯条对中心必须在检测框内的约束
                require_center_in_box = match["require_center_in_box"].as<bool>(true);
            }
            
            if (pointer["refine"]) {
                point_refine_radius_ratio = pointer["refine"]["radius_ratio"].as<double>(0.1);
            }
        }
    } catch (const YAML::Exception& e) {
        // 使用默认值
    }
}

PointerResult LightbarDetector::process(const cv::Mat& image, 
                                       const cv::Rect& detection_box,
                                       ArmorColor enemy_color) {
    std::cout << "\n========== Pointer处理开始 ==========" << std::endl;
    std::cout << "检测框: [" << detection_box.x << ", " << detection_box.y 
              << ", " << detection_box.width << ", " << detection_box.height << "]" << std::endl;
    
    PointerResult result;
    result.success = false;
    result.confidence = 0.0;
    result.detected_color = ARMOR_COLOR_NONE;
    
    // 保存原始检测框信息（相对于整图）
    original_detection_box_ = detection_box;
    
    // 步骤1：使用传入的检测框，不再额外扩展
    cv::Rect expanded_roi = detection_box;
    
    // 边界检查
    if (expanded_roi.x < 0 || expanded_roi.y < 0 || 
        expanded_roi.x + expanded_roi.width > image.cols ||
        expanded_roi.y + expanded_roi.height > image.rows) {
        std::cout << "[错误] ROI超出图像边界" << std::endl;
        return result;
    }
    
    // 计算检测框在ROI中的相对位置
    detection_box_in_roi_ = cv::Rect(
        detection_box.x - expanded_roi.x,
        detection_box.y - expanded_roi.y,
        detection_box.width,
        detection_box.height
    );
    
    // 提取ROI
    cv::Mat roi = image(expanded_roi);
    roi_offset_ = cv::Point(expanded_roi.x, expanded_roi.y);
    
    std::cout << "ROI大小: " << roi.cols << "x" << roi.rows << std::endl;
    std::cout << "相对检测框: [" << detection_box_in_roi_.x << ", " << detection_box_in_roi_.y 
              << ", " << detection_box_in_roi_.width << ", " << detection_box_in_roi_.height << "]" << std::endl;
    
    // 步骤2：图像预处理
    // cv::Mat binary = image_processor_.process_roi(roi, enemy_color);

    // 方式2：明确指定方法
    cv::Mat binary = image_processor_.process_roi(roi, enemy_color, 
                                           GrayMethod::GRAY_SUBTRACT,  // 使用通道差分处理过曝
                                           BinaryMethod::BINARY_HISTOGRAM); // 使用直方图阈值
    
    // 保存二值化图像到结果中（用于调试显示）
    result.binary_image = binary.clone();
    
    // 步骤3：查找灯条（带检测框约束）
    std::vector<Lightbar> lightbars = find_lightbars_with_constraint(binary);
    
    std::cout << "\n找到的有效灯条数量: " << lightbars.size() << std::endl;
    
    if (lightbars.size() < 2) {
        std::cout << "[警告] 灯条数量不足，需要至少2个，当前只有" << lightbars.size() << "个" << std::endl;
        return result;
    }
    
    // 后续处理保持不变...
    // 步骤4：匹配灯条对（带检测框约束）
    auto matched_pair = match_lightbars_with_constraint(lightbars);
    
    if (matched_pair.first.contour.empty() || matched_pair.second.contour.empty()) {
        std::cout << "[警告] 未找到匹配的灯条对" << std::endl;
        return result;
    }
    
    std::cout << "\n成功匹配灯条对!" << std::endl;
    
    // 步骤5：颜色检测
    result.detected_color = image_processor_.detect_armor_color(roi, matched_pair);
    std::cout << "检测到的颜色: " << static_cast<int>(result.detected_color) << std::endl;
    
    // 步骤6：精确定位端点
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    
    PointPair pp1 = refine_endpoints(matched_pair.first, gray);
    PointPair pp2 = refine_endpoints(matched_pair.second, gray);
    
    // 步骤7：生成四个角点
    result.four_points = generate_four_points(pp1, pp2);
    
    // 将相对坐标转换为绝对坐标
    for (auto& point : result.four_points) {
        point.x += expanded_roi.x;
        point.y += expanded_roi.y;
    }
    
    result.success = true;
    result.confidence = calculate_confidence(matched_pair);
    
    std::cout << "Pointer处理成功，置信度: " << result.confidence << std::endl;
    std::cout << "========== Pointer处理结束 ==========\n" << std::endl;
    
    return result;
}

cv::Rect LightbarDetector::expand_roi(const cv::Rect& box, const cv::Size& image_size) {
    // 适度扩展，主要是为了完整包含灯条
    int new_width = static_cast<int>(box.width * config_.roi_extend_width_ratio);
    int new_height = static_cast<int>(box.height * config_.roi_extend_height_ratio);
    
    int x = box.x - (new_width - box.width) / 2;
    int y = box.y - (new_height - box.height) / 2;
    
    // 确保不超出图像边界
    x = std::max(0, x);
    y = std::max(0, y);
    new_width = std::min(new_width, image_size.width - x);
    new_height = std::min(new_height, image_size.height - y);
    
    return cv::Rect(x, y, new_width, new_height);
}

std::vector<Lightbar> LightbarDetector::find_lightbars_with_constraint(const cv::Mat& binary) {
    std::vector<Lightbar> lightbars;
    
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    
    std::cout << "\n轮廓查找阶段：" << std::endl;
    std::cout << "总轮廓数: " << contours.size() << std::endl;
    
    // 筛选有效的灯条
    int contour_idx = 0;
    for (const auto& contour : contours) {
        Lightbar lightbar;
        lightbar.contour = contour;
        lightbar.rect = cv::minAreaRect(contour);
        
        // 计算灯条的角度和长度
        if (lightbar.rect.size.width > lightbar.rect.size.height) {
            lightbar.angle = lightbar.rect.angle - 90;
            lightbar.length = lightbar.rect.size.width;
        } else {
            lightbar.angle = lightbar.rect.angle;
            lightbar.length = lightbar.rect.size.height;
        }
        
        // 验证灯条是否有效
        double ratio = lightbar.length / std::min(lightbar.rect.size.width, lightbar.rect.size.height);
        double area = lightbar.rect.size.area();
        
        std::cout << "\n轮廓#" << contour_idx++ << ":" << std::endl;
        std::cout << "  中心: (" << lightbar.rect.center.x << ", " << lightbar.rect.center.y << ")" << std::endl;
        std::cout << "  尺寸: " << lightbar.rect.size.width << " x " << lightbar.rect.size.height << std::endl;
        std::cout << "  角度: " << lightbar.angle << "°" << std::endl;
        std::cout << "  长度: " << lightbar.length << std::endl;
        std::cout << "  长宽比: " << ratio << std::endl;
        std::cout << "  面积: " << area << std::endl;
        
        // 基本的形状和大小约束
        bool passed_basic_check = true;
        std::string rejection_reason = "";
        
        if (ratio < config_.lightbar_min_ratio) {
            passed_basic_check = false;
            rejection_reason += "长宽比过小(" + std::to_string(ratio) + " < " + std::to_string(config_.lightbar_min_ratio) + "); ";
        }
        if (ratio > config_.lightbar_max_ratio) {
            passed_basic_check = false;
            rejection_reason += "长宽比过大(" + std::to_string(ratio) + " > " + std::to_string(config_.lightbar_max_ratio) + "); ";
        }
        if (area < config_.lightbar_min_area) {
            passed_basic_check = false;
            rejection_reason += "面积过小(" + std::to_string(area) + " < " + std::to_string(config_.lightbar_min_area) + "); ";
        }
        if (std::abs(lightbar.angle) > 45.0) {
            passed_basic_check = false;
            rejection_reason += "角度过大(|" + std::to_string(lightbar.angle) + "| > 45°); ";
        }
        
        if (!passed_basic_check) {
            std::cout << "  [拒绝] 基础检查失败: " << rejection_reason << std::endl;
            continue;
        }
        
        // 新增：检查灯条是否与检测框有足够的重叠
        if (!is_lightbar_valid_for_detection(lightbar)) {
            std::cout << "  [拒绝] 与检测框重叠不足" << std::endl;
            continue;
        }
        
        std::cout << "  [通过] 有效灯条" << std::endl;
        lightbars.push_back(lightbar);
    }
    
    return lightbars;
}

bool LightbarDetector::is_lightbar_valid_for_detection(const Lightbar& lightbar) {
    // 获取灯条的边界框
    cv::Rect lightbar_bbox = lightbar.rect.boundingRect();
    
    // 计算与检测框的交集
    cv::Rect intersection = lightbar_bbox & detection_box_in_roi_;
    
    // 详细的重叠分析
    std::cout << "    重叠检查:" << std::endl;
    std::cout << "      灯条边界框: [" << lightbar_bbox.x << ", " << lightbar_bbox.y 
              << ", " << lightbar_bbox.width << ", " << lightbar_bbox.height << "]" << std::endl;
    std::cout << "      交集面积: " << intersection.area() << std::endl;
    
    // 如果没有交集，检查是否在检测框附近
    if (intersection.area() == 0) {
        // 计算灯条中心到检测框的距离
        cv::Point2f center = lightbar.rect.center;
        
        // 检查是否在检测框的合理扩展范围内（允许灯条稍微超出检测框）
        int margin = static_cast<int>(lightbar.length * 0.5);  // 允许半个灯条长度的余量
        cv::Rect extended_box(
            detection_box_in_roi_.x - margin,
            detection_box_in_roi_.y - margin,
            detection_box_in_roi_.width + 2 * margin,
            detection_box_in_roi_.height + 2 * margin
        );
        
        // 确保扩展框在ROI范围内
        extended_box &= cv::Rect(0, 0, detection_box_in_roi_.width + 100, detection_box_in_roi_.height + 100);
        
        bool in_extended = extended_box.contains(center);
        std::cout << "      无交集，扩展框检查: " << (in_extended ? "通过" : "失败") << std::endl;
        return in_extended;
    }
    
    // 计算重叠比例
    double overlap_ratio = static_cast<double>(intersection.area()) / lightbar_bbox.area();
    std::cout << "      重叠比例: " << overlap_ratio << " (阈值: " << config_.min_overlap_ratio << ")" << std::endl;
    
    return overlap_ratio >= config_.min_overlap_ratio;
}

std::pair<Lightbar, Lightbar> LightbarDetector::match_lightbars_with_constraint(
    const std::vector<Lightbar>& lightbars) {
    
    std::cout << "\n灯条配对阶段：" << std::endl;
    std::cout << "总共 " << lightbars.size() << " 个灯条进行配对" << std::endl;
    
    std::pair<Lightbar, Lightbar> best_pair;
    double best_score = std::numeric_limits<double>::max();
    
    int pair_count = 0;
    
    // 尝试所有可能的配对
    for (size_t i = 0; i < lightbars.size(); ++i) {
        for (size_t j = i + 1; j < lightbars.size(); ++j) {
            const Lightbar& lb1 = lightbars[i];
            const Lightbar& lb2 = lightbars[j];
            
            // 计算两个灯条的中心点
            cv::Point2f pair_center = (lb1.rect.center + lb2.rect.center) * 0.5f;
            
            std::cout << "\n  配对尝试#" << pair_count++ << " (灯条" << i << " + 灯条" << j << "):" << std::endl;
            std::cout << "    配对中心: (" << pair_center.x << ", " << pair_center.y << ")" << std::endl;
            
            // 新增：如果要求灯条对中心在检测框内，进行检查
            if (config_.require_center_in_box && 
                !detection_box_in_roi_.contains(pair_center)) {
                std::cout << "    [拒绝] 配对中心不在检测框内" << std::endl;
                continue;
            }
            
            // 计算各种差异指标
            double angle_diff = std::abs(lb1.angle - lb2.angle);
            double length_ratio = std::max(lb1.length, lb2.length) / std::min(lb1.length, lb2.length);
            
            // 计算中心点距离与平均长度的比值（装甲板的宽高比）
            cv::Point2f center_diff = lb1.rect.center - lb2.rect.center;
            double center_distance = std::sqrt(center_diff.x * center_diff.x + center_diff.y * center_diff.y);
            double avg_length = (lb1.length + lb2.length) / 2.0;
            double width_height_ratio = center_distance / avg_length;
            
            std::cout << "    角度差: " << angle_diff << "° (阈值: " << config_.max_angle_diff << "°)" << std::endl;
            std::cout << "    长度比: " << length_ratio << " (阈值: " << config_.max_length_ratio << ")" << std::endl;
            std::cout << "    宽高比: " << width_height_ratio << " (期望范围: 1.0-5.0)" << std::endl;
            
            // 检查是否满足基本条件
            bool meets_criteria = (angle_diff <= config_.max_angle_diff &&
                                 length_ratio <= config_.max_length_ratio &&
                                 width_height_ratio >= 1.0 && width_height_ratio <= 5.0);
            
            if (!meets_criteria) {
                std::cout << "    [拒绝] 不满足配对条件" << std::endl;
                continue;
            }
            
            // 计算综合得分（越小越好）
            double score = angle_diff + (length_ratio - 1.0) * 10.0 + 
                          std::abs(width_height_ratio - 2.2) * 20.0;
            
            // 新增：优先选择更靠近检测框中心的配对
            cv::Point2f detection_center(
                detection_box_in_roi_.x + detection_box_in_roi_.width / 2.0f,
                detection_box_in_roi_.y + detection_box_in_roi_.height / 2.0f
            );
            double distance_to_detection_center = cv::norm(pair_center - detection_center);
            score += distance_to_detection_center * 0.1;  // 加入距离惩罚
            
            std::cout << "    综合得分: " << score << " (当前最佳: " << best_score << ")" << std::endl;
            
            if (score < best_score) {
                best_score = score;
                best_pair = {lb1, lb2};
                std::cout << "    [接受] 新的最佳配对!" << std::endl;
            }
        }
    }
    
    if (best_score < std::numeric_limits<double>::max()) {
        std::cout << "\n最终选择的配对得分: " << best_score << std::endl;
    } else {
        std::cout << "\n未找到有效配对" << std::endl;
    }
    
    return best_pair;
}

double LightbarDetector::calculate_confidence(const std::pair<Lightbar, Lightbar>& matched_pair) {
    // 基础置信度
    double confidence = 0.8;
    
    // 根据角度差异调整
    double angle_diff = std::abs(matched_pair.first.angle - matched_pair.second.angle);
    confidence += (1.0 - angle_diff / config_.max_angle_diff) * 0.1;
    
    // 根据长度比例调整
    double length_ratio = std::max(matched_pair.first.length, matched_pair.second.length) / 
                         std::min(matched_pair.first.length, matched_pair.second.length);
    confidence += (1.0 - (length_ratio - 1.0) / (config_.max_length_ratio - 1.0)) * 0.1;
    
    return std::min(confidence, 1.0);
}

// 其余函数保持不变...
PointPair LightbarDetector::refine_endpoints(const Lightbar& lightbar, const cv::Mat& gray) {
    // 与原实现相同...
    cv::Vec4f line;
    cv::fitLine(lightbar.contour, line, cv::DIST_L2, 0, 0.01, 0.01);
    
    cv::Point2f center = lightbar.rect.center;
    cv::Point2f direction(line[0], line[1]);
    
    cv::Point2f endpoint1, endpoint2;
    double max_dist1 = 0, max_dist2 = 0;
    
    for (const auto& point : lightbar.contour) {
        cv::Point2f p(point.x, point.y);
        cv::Point2f diff = p - center;
        double proj = diff.dot(direction);
        
        if (proj > max_dist1) {
            max_dist1 = proj;
            endpoint1 = p;
        } else if (proj < max_dist2) {
            max_dist2 = proj;
            endpoint2 = p;
        }
    }
    
    int radius = static_cast<int>(cv::norm(endpoint1 - endpoint2) * config_.point_refine_radius_ratio);
    
    auto calculate_barycenter = [&gray, radius](cv::Point2f center) -> cv::Point2f {
        int x_min = std::max(static_cast<int>(center.x - radius), 0);
        int x_max = std::min(static_cast<int>(center.x + radius), gray.cols - 1);
        int y_min = std::max(static_cast<int>(center.y - radius), 0);
        int y_max = std::min(static_cast<int>(center.y + radius), gray.rows - 1);
        
        double weighted_x = 0, weighted_y = 0, total_weight = 0;
        
        for (int y = y_min; y <= y_max; ++y) {
            for (int x = x_min; x <= x_max; ++x) {
                cv::Point2f p(x, y);
                if (cv::norm(p - center) <= radius) {
                    double weight = gray.at<uchar>(y, x);
                    weighted_x += x * weight;
                    weighted_y += y * weight;
                    total_weight += weight;
                }
            }
        }
        
        if (total_weight > 0) {
            return cv::Point2f(weighted_x / total_weight, weighted_y / total_weight);
        }
        return center;
    };
    
    cv::Point2f refined_endpoint1 = calculate_barycenter(endpoint1);
    cv::Point2f refined_endpoint2 = calculate_barycenter(endpoint2);
    
    if (refined_endpoint1.y < refined_endpoint2.y) {
        return PointPair(refined_endpoint1, refined_endpoint2);
    } else {
        return PointPair(refined_endpoint2, refined_endpoint1);
    }
}

std::vector<cv::Point2f> LightbarDetector::generate_four_points(const PointPair& pp1, const PointPair& pp2) {
    std::vector<cv::Point2f> four_points;
    
    float pp1_center_x = (pp1.point_up.x + pp1.point_down.x) / 2.0f;
    float pp2_center_x = (pp2.point_up.x + pp2.point_down.x) / 2.0f;
    
    if (pp1_center_x < pp2_center_x) {
        four_points.push_back(pp1.point_up);
        four_points.push_back(pp2.point_up);
        four_points.push_back(pp2.point_down);
        four_points.push_back(pp1.point_down);
    } else {
        four_points.push_back(pp2.point_up);
        four_points.push_back(pp1.point_up);
        four_points.push_back(pp1.point_down);
        four_points.push_back(pp2.point_down);
    }
    
    return four_points;
}

} // namespace armor_pointer