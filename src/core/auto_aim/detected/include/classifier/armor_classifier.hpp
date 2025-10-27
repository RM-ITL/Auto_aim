#ifndef ARMOR_CLASSIFIER_HPP
#define ARMOR_CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <memory>
#include "common/armor.hpp"

namespace armor_auto_aim {


class ArmorClassifier {
public:
    struct Config {
        std::string model_path;
        std::string device = "CPU";
        float conf_threshold = 0.8f;
        std::vector<std::string> classes = {"1", "2", "3", "4", "Sentry"};
        
        static Config from_yaml(const YAML::Node& node);
    };
    
    // 统一使用配置文件初始化
    explicit ArmorClassifier(const std::string& config_path);
    ~ArmorClassifier();
    
    ArmorClassifier(const ArmorClassifier&) = delete;
    ArmorClassifier& operator=(const ArmorClassifier&) = delete;
    
    // 核心分类接口 - 现在直接操作Armor对象
    void classify(Armor& armor);
    void classify_batch(std::vector<Armor>& armors);
    
    
    // 状态查询
    bool is_initialized() const { return initialized_; }
    std::vector<std::string> get_class_names() const { return class_names_; }
    float get_confidence_threshold() const { return conf_threshold_; }
    void set_confidence_threshold(float threshold);

private:
    // OpenVINO组件
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    
    // 状态和参数
    bool initialized_;
    std::string model_path_;
    std::string device_;
    int input_width_;
    int input_height_;
    float conf_threshold_;
    std::vector<std::string> class_names_;
    
    // 内部方法
    void load_config(const std::string& config_path);
    bool initialize_model();
    ov::Tensor preprocess(const cv::Mat& image);
    std::pair<int, float> postprocess(const ov::Tensor& output);
    std::vector<float> get_all_confidences(const ov::Tensor& output);
    bool validate_input(const cv::Mat& image) const;
    
    // 新增：将分类结果更新到Armor对象
    void update_armor_with_classification(Armor& armor, int class_id, float confidence);
    ArmorName convert_class_id_to_armor_name(int class_id, const std::string& class_name);
    
    // 内部实现方法
    std::pair<int, float> classify_internal(const cv::Mat& armor_image);
};

// 工厂函数
std::unique_ptr<ArmorClassifier> create_classifier(const std::string& config_path);

} // namespace armor_auto_aim

#endif // ARMOR_CLASSIFIER_HPP