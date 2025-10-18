#ifndef ARMOR_DETECTOR_HPP
#define ARMOR_DETECTOR_HPP

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <future>
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "common/armor.hpp"

namespace armor_auto_aim {

class ArmorDetector {
public:
    struct Config {
        std::string model_path;
        std::string device = "CPU";
        float conf_threshold = 0.5f;
        float iou_threshold = 0.5f;
        int request_pool_size = 4;
        std::vector<std::string> classes;
        
        static Config from_yaml(const YAML::Node& node);
    };

    // 统一使用配置文件初始化
    explicit ArmorDetector(const std::string& config_path);
    ~ArmorDetector();

    ArmorDetector(const ArmorDetector&) = delete;
    ArmorDetector& operator=(const ArmorDetector&) = delete;

    // 核心检测接口 - 现在直接返回Armor对象
    std::vector<Armor> detect(const cv::Mat& image);
    std::future<std::vector<Armor>> detect_async(const cv::Mat& image);
    
    
    // 状态查询
    cv::Size get_input_size() const { return cv::Size(input_width_, input_height_); }
    bool is_initialized() const { return initialized_; }
    const std::vector<std::string>& get_classes() const { return classes_; }

private:
    struct InferenceTask {
        cv::Mat image;
        cv::Size original_size;
        std::shared_ptr<std::promise<std::vector<Armor>>> promise;  // 改为Armor
    };

    // OpenVINO组件
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
    ov::Output<const ov::Node> output_layer_;
    
    // 异步推理管理
    std::vector<ov::InferRequest> infer_requests_;
    std::vector<ov::Tensor> input_tensors_;
    std::queue<int> available_request_ids_;
    std::mutex requests_mutex_;
    std::condition_variable requests_cv_;
    std::vector<std::unique_ptr<InferenceTask>> pending_tasks_;
    std::atomic<bool> stop_processing_{false};
    
    // 模型参数
    int input_width_, input_height_;
    float conf_threshold_, iou_threshold_;
    std::vector<std::string> classes_;
    std::vector<cv::Scalar> color_palette_;
    bool initialized_;
    
    // 内部方法
    void load_config(const std::string& config_path);
    void init_model();
    void init_async_pipeline(int pool_size);
    void cleanup_async_pipeline();
    
    void preprocess(const cv::Mat& image, ov::Tensor& tensor);
    
    // 新的后处理方法，直接生成Armor对象
    std::vector<Armor> postprocess(const ov::Tensor& output, 
                                   const cv::Size& original_size,
                                   const cv::Mat& original_image);
    
    // 辅助方法：从检测结果创建Armor对象
    Armor create_armor_from_detection(const cv::Rect& box, 
                                      float confidence, 
                                      int class_id,
                                      const cv::Mat& image);
    
    int get_available_request_id();
    void return_request_id(int id);
    void on_inference_complete(int request_id, std::exception_ptr ex);
    
    // 成员变量保存配置
    std::string model_path_;
    std::string device_;
};

// 工厂函数
std::unique_ptr<ArmorDetector> create_detector(const std::string& config_path);

} // namespace armor_auto_aim

#endif // ARMOR_DETECTOR_HPP