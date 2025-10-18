#include "classifier/armor_classifier.hpp"
#include "detector_tools.hpp"
#include <filesystem>
#include <rclcpp/rclcpp.hpp>

namespace armor_auto_aim {

ArmorClassifier::Config ArmorClassifier::Config::from_yaml(const YAML::Node& node) {
    Config config;
    
    if (node["model"]) {
        const auto& model = node["model"];
        config.model_path = model["path"].as<std::string>();
        config.device = model["device"].as<std::string>("CPU");
        config.conf_threshold = model["conf_threshold"].as<float>(0.8f);
    }
    
    if (node["classes"]) {
        config.classes.clear();
        for (const auto& cls : node["classes"]) {
            config.classes.push_back(cls.as<std::string>());
        }
    }
    
    return config;
}

ArmorClassifier::ArmorClassifier(const std::string& config_path)
    : initialized_(false)
    , input_width_(32)
    , input_height_(32) {
    
    load_config(config_path);
    initialized_ = initialize_model();
    
    if (!initialized_) {
        throw std::runtime_error("分类器初始化失败: " + model_path_);
    }
    
    RCLCPP_INFO(rclcpp::get_logger("ArmorClassifier"), 
        "分类器初始化成功 | 模型: %s | 设备: %s | 阈值: %.2f", 
        model_path_.c_str(), device_.c_str(), conf_threshold_);
}

ArmorClassifier::~ArmorClassifier() = default;

void ArmorClassifier::load_config(const std::string& config_path) {
    try {
        YAML::Node root = YAML::LoadFile(config_path);
        
        if (!root["armor_classifier"]) {
            throw std::runtime_error("配置文件中未找到 'armor_classifier' 节点");
        }
        
        auto config = Config::from_yaml(root["armor_classifier"]);
        
        model_path_ = config.model_path;
        device_ = config.device;
        conf_threshold_ = config.conf_threshold;
        class_names_ = config.classes;
        
        if (model_path_.empty()) {
            throw std::runtime_error("模型路径不能为空");
        }
        
        if (!std::filesystem::exists(model_path_)) {
            throw std::runtime_error("模型文件不存在: " + model_path_);
        }
        
        if (class_names_.empty()) {
            throw std::runtime_error("类别列表不能为空");
        }
        
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML解析错误: " + std::string(e.what()));
    }
}

bool ArmorClassifier::initialize_model() {
    try {
        model_ = core_.read_model(model_path_);
        compiled_model_ = core_.compile_model(model_, device_);
        infer_request_ = compiled_model_.create_infer_request();
        
        auto input_info = compiled_model_.input();
        auto input_shape = input_info.get_shape();
        
        if (input_shape.size() >= 4) {
            input_height_ = static_cast<int>(input_shape[2]);
            input_width_ = static_cast<int>(input_shape[3]);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("ArmorClassifier"), 
            "模型加载失败: %s", e.what());
        return false;
    }
}

// 主要接口：直接操作Armor对象
void ArmorClassifier::classify(Armor& armor) {
    // 检查是否已经分类过
    if (armor.classify_confidence >= 0.0f) {
        RCLCPP_DEBUG(rclcpp::get_logger("ArmorClassifier"),
            "装甲板已分类，跳过重复分类");
        return;
    }
    
    // 检查ROI图像是否可用
    if (armor.roi_image.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("ArmorClassifier"),
            "装甲板ROI图像为空，无法分类");
        armor.classify_confidence = 0.0f;
        armor.is_valid = false;
        armor.name = not_armor;
        return;
    }
    
    // 执行分类
    auto [class_id, confidence] = classify_internal(armor.roi_image);
    
    // 更新Armor对象
    update_armor_with_classification(armor, class_id, confidence);
    
    RCLCPP_DEBUG(rclcpp::get_logger("ArmorClassifier"),
        "分类完成 | 数字: %s | 置信度: %.3f | 有效: %s",
        armor.getNameString().c_str(),
        armor.classify_confidence,
        armor.is_valid ? "是" : "否");
}

// 批量分类
void ArmorClassifier::classify_batch(std::vector<Armor>& armors) {
    int classified_count = 0;
    int skipped_count = 0;
    
    for (auto& armor : armors) {
        // 跳过已分类的装甲板
        if (armor.classify_confidence >= 0.0f) {
            skipped_count++;
            continue;
        }
        
        classify(armor);
        classified_count++;
    }
    
    RCLCPP_DEBUG(rclcpp::get_logger("ArmorClassifier"),
        "批量分类完成 | 总数: %zu | 分类: %d | 跳过: %d",
        armors.size(), classified_count, skipped_count);
}

// 内部分类实现
std::pair<int, float> ArmorClassifier::classify_internal(const cv::Mat& armor_image) {
    if (!validate_input(armor_image)) {
        return {-1, 0.0f};
    }
    
    if (!initialized_) {
        RCLCPP_ERROR(rclcpp::get_logger("ArmorClassifier"), "分类器未初始化");
        return {-1, 0.0f};
    }
    
    try {
        ov::Tensor input_tensor = preprocess(armor_image);
        if (!input_tensor) {
            return {-1, 0.0f};
        }
        
        infer_request_.set_input_tensor(input_tensor);
        infer_request_.infer();
        
        auto output = infer_request_.get_output_tensor();
        
        // 获取所有类别的置信度用于调试
        auto all_confidences = get_all_confidences(output);
        
        // 获取最高置信度的类别
        auto [class_id, confidence] = postprocess(output);
        
        // 调试输出：显示所有类别的置信度
        if (rclcpp::get_logger("ArmorClassifier").get_effective_level() <= rclcpp::Logger::Level::Debug) {
            std::stringstream ss;
            ss << "分类详细置信度: ";
            for (size_t i = 0; i < class_names_.size() && i < all_confidences.size(); ++i) {
                ss << class_names_[i] << ":" << std::fixed << std::setprecision(3) << all_confidences[i] << " ";
            }
            RCLCPP_DEBUG(rclcpp::get_logger("ArmorClassifier"), "%s", ss.str().c_str());
        }
        
        return {class_id, confidence};
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("ArmorClassifier"), 
            "分类失败: %s", e.what());
        return {-1, 0.0f};
    }
}

// 更新Armor对象的分类信息
void ArmorClassifier::update_armor_with_classification(Armor& armor, int class_id, float confidence) {
    // 保存分类器输出的原始索引
    armor.classify_class_id = class_id;
    armor.classify_confidence = confidence;
    
    // 判断分类是否有效
    bool classification_valid = (confidence >= conf_threshold_ && 
                                class_id >= 0 && 
                                class_id < static_cast<int>(class_names_.size()));
    
    if (classification_valid) {
        // 获取类名
        std::string class_name = class_names_[class_id];
        
        // 转换为ArmorName枚举
        armor.name = convert_class_id_to_armor_name(class_id, class_name);
        
        // 设置装甲板优先级（根据游戏规则）
        switch(armor.name) {
            case one:    // 英雄
            case two:
                armor.priority = first;
                break;
            case three:  // 步兵
            case four:
            case five:
                armor.priority = second;
                break;
            case sentry: // 哨兵
                armor.priority = third;
                break;
            case outpost: // 前哨站
                armor.priority = forth;
                break;
            default:
                armor.priority = fifth;
        }
        
        // 根据装甲板宽高比重新确认类型（大/小装甲）
        // 3号、4号、5号可能有大装甲板
        if (armor.name >= three && armor.name <= five) {
            float aspect_ratio = static_cast<float>(armor.box.width) / armor.box.height;
            armor.type = (aspect_ratio > 3.5f) ? big : small;
        }
        
        // 组合检测和分类的有效性
        armor.is_valid = true;
        
    } else {
        // 分类无效
        armor.name = not_armor;
        armor.priority = fifth;
        armor.is_valid = false;
    }
}

// 将分类器的class_id和class_name转换为ArmorName枚举
ArmorName ArmorClassifier::convert_class_id_to_armor_name(int class_id, const std::string& class_name) {
    // 根据类名字符串映射到枚举值
    if (class_name == "1") return one;
    if (class_name == "2") return two;
    if (class_name == "3") return three;
    if (class_name == "4") return four;
    if (class_name == "5") return five;
    if (class_name == "Sentry" || class_name == "sentry") return sentry;
    if (class_name == "Outpost" || class_name == "outpost") return outpost;
    if (class_name == "Base" || class_name == "base") return base;
    
    // 如果类名不匹配，尝试使用索引值
    // 这是为了兼容可能的不同配置
    if (class_id >= 0 && class_id <= 4) {
        // 假设前5个索引对应数字1-5
        return static_cast<ArmorName>(class_id + 1);
    }
    
    return not_armor;
}

ov::Tensor ArmorClassifier::preprocess(const cv::Mat& image) {
    try {
        if (image.empty()) {
            return ov::Tensor();
        }
        
        cv::Mat gray_image;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        } else if (image.channels() == 1) {
            gray_image = image.clone();
        } else {
            return ov::Tensor();
        }
        
        cv::Mat rgb_image;
        cv::cvtColor(gray_image, rgb_image, cv::COLOR_GRAY2RGB);
        
        cv::Mat resized;
        cv::resize(rgb_image, resized, cv::Size(input_width_, input_height_));
        
        cv::Mat normalized;
        resized.convertTo(normalized, CV_32F, 1.0/255.0);
        normalized = (normalized - 0.5f) / 0.5f;
        
        ov::Shape shape = {1, 3, 
                          static_cast<size_t>(input_height_), 
                          static_cast<size_t>(input_width_)};
        ov::Tensor tensor = ov::Tensor(ov::element::f32, shape);
        float* tensor_data = tensor.data<float>();
        
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < input_height_; ++h) {
                for (int w = 0; w < input_width_; ++w) {
                    int idx = c * input_height_ * input_width_ + h * input_width_ + w;
                    tensor_data[idx] = normalized.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        
        return tensor;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("ArmorClassifier"), 
            "预处理失败: %s", e.what());
        return ov::Tensor();
    }
}

std::pair<int, float> ArmorClassifier::postprocess(const ov::Tensor& output) {
    try {
        const float* output_data = output.data<const float>();
        const auto output_size = output.get_size();
        
        if (output_size < class_names_.size()) {
            return {-1, 0.0f};
        }
        
        int max_class_id = 0;
        float max_score = output_data[0];
        
        for (size_t i = 1; i < class_names_.size() && i < output_size; ++i) {
            if (output_data[i] > max_score) {
                max_score = output_data[i];
                max_class_id = static_cast<int>(i);
            }
        }
        
        return {max_class_id, max_score};
        
    } catch (...) {
        return {-1, 0.0f};
    }
}

std::vector<float> ArmorClassifier::get_all_confidences(const ov::Tensor& output) {
    std::vector<float> confidences;
    
    try {
        const float* output_data = output.data<const float>();
        const auto output_size = output.get_size();
        
        for (size_t i = 0; i < class_names_.size() && i < output_size; ++i) {
            confidences.push_back(output_data[i]);
        }
    } catch (...) {
        // 返回空vector
    }
    
    return confidences;
}

bool ArmorClassifier::validate_input(const cv::Mat& image) const {
    return !image.empty() && 
           image.rows >= 8 && 
           image.cols >= 8 && 
           (image.channels() == 1 || image.channels() == 3);
}

void ArmorClassifier::set_confidence_threshold(float threshold) {
    conf_threshold_ = std::max(0.0f, std::min(1.0f, threshold));
    RCLCPP_INFO(rclcpp::get_logger("ArmorClassifier"), 
        "置信度阈值更新: %.2f", conf_threshold_);
}


std::unique_ptr<ArmorClassifier> create_classifier(const std::string& config_path) {
    try {
        return std::make_unique<ArmorClassifier>(config_path);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("ArmorClassifier"), 
            "创建分类器失败: %s", e.what());
        return nullptr;
    }
}

} // namespace armor_auto_aim