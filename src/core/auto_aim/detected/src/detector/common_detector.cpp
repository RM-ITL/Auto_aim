#include "detector/common_detector.hpp"
#include "detector_tools.hpp"
#include <filesystem>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <utility>

namespace armor_auto_aim {

namespace {

template<typename... Args>
void log_info(const char* fmt, Args&&... args) {
    char buffer[512];
    std::snprintf(buffer, sizeof(buffer), fmt, std::forward<Args>(args)...);
    std::cout << "[ArmorDetector][INFO] " << buffer << std::endl;
}

template<typename... Args>
void log_error(const char* fmt, Args&&... args) {
    char buffer[512];
    std::snprintf(buffer, sizeof(buffer), fmt, std::forward<Args>(args)...);
    std::cerr << "[ArmorDetector][ERROR] " << buffer << std::endl;
}

template<typename... Args>
void log_debug(const char* fmt, Args&&... args) {
#ifndef NDEBUG
    char buffer[512];
    std::snprintf(buffer, sizeof(buffer), fmt, std::forward<Args>(args)...);
    std::cout << "[ArmorDetector][DEBUG] " << buffer << std::endl;
#else
    (void)fmt;
    (void)sizeof...(args);
#endif
}

}  // namespace

ArmorDetector::Config ArmorDetector::Config::from_yaml(const YAML::Node& node) {
    Config config;
    
    if (node["model"]) {
        const auto& model_node = node["model"];
        config.model_path = model_node["path"].as<std::string>();
        config.device = model_node["device"].as<std::string>("CPU");
        config.conf_threshold = model_node["conf_threshold"].as<float>(0.5f);
        config.iou_threshold = model_node["iou_threshold"].as<float>(0.5f);
        config.request_pool_size = model_node["request_pool_size"].as<int>(4);
    }
    
    if (node["classes"]) {
        config.classes = node["classes"].as<std::vector<std::string>>();
    }
    
    return config;
}

ArmorDetector::ArmorDetector(const std::string& config_path) 
    : initialized_(false) {
    
    load_config(config_path);
    
    try {
        for (const auto& cls : classes_) {
            color_palette_.push_back(utils::name_to_color(cls));
        }
        
        init_model();
        init_async_pipeline(4);  // 从配置中读取的值
        initialized_ = true;
        
        log_info("检测器初始化成功 | 模型: %s | 设备: %s | 输入尺寸: %dx%d",
                 model_path_.c_str(), device_.c_str(), input_width_, input_height_);
        
    } catch (const std::exception& e) {
        throw std::runtime_error("检测器初始化失败: " + std::string(e.what()));
    }
}

ArmorDetector::~ArmorDetector() {
    cleanup_async_pipeline();
}

void ArmorDetector::load_config(const std::string& config_path) {
    try {
        YAML::Node root = YAML::LoadFile(config_path);
        
        if (!root["yolo"]) {
            throw std::runtime_error("配置文件中未找到 'yolo' 节点");
        }
        
        auto config = Config::from_yaml(root["yolo"]);
        
        model_path_ = config.model_path;
        device_ = config.device;
        conf_threshold_ = config.conf_threshold;
        iou_threshold_ = config.iou_threshold;
        classes_ = config.classes;
        
        if (model_path_.empty()) {
            throw std::runtime_error("模型路径不能为空");
        }
        
        if (!std::filesystem::exists(model_path_)) {
            throw std::runtime_error("模型文件不存在: " + model_path_);
        }
        
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML解析错误: " + std::string(e.what()));
    }
}

void ArmorDetector::init_model() {
    model_ = core_.read_model(model_path_);
    compiled_model_ = core_.compile_model(model_, device_);
    
    auto inputs = compiled_model_.inputs();
    auto input_shape = inputs[0].get_shape();
    input_height_ = static_cast<int>(input_shape[2]);
    input_width_ = static_cast<int>(input_shape[3]);
    
    output_layer_ = compiled_model_.output();
}

void ArmorDetector::init_async_pipeline(int pool_size) {
    infer_requests_.reserve(pool_size);
    input_tensors_.reserve(pool_size);
    pending_tasks_.resize(pool_size);
    
    for (int i = 0; i < pool_size; ++i) {
        infer_requests_.emplace_back(compiled_model_.create_infer_request());
        
        input_tensors_.emplace_back(ov::Tensor(
            ov::element::f32, 
            {1, 3, static_cast<size_t>(input_height_), static_cast<size_t>(input_width_)}
        ));
        
        infer_requests_[i].set_callback([this, i](std::exception_ptr ex) {
            this->on_inference_complete(i, ex);
        });
        
        available_request_ids_.push(i);
    }
}

void ArmorDetector::cleanup_async_pipeline() {
    stop_processing_.store(true);
    requests_cv_.notify_all();
    
    for (auto& request : infer_requests_) {
        try {
            request.wait();
        } catch (...) {}
    }
}

// 主要检测接口 - 现在返回Armor对象
std::vector<Armor> ArmorDetector::detect(const cv::Mat& image) {
    if (!initialized_) {
        throw std::runtime_error("检测器未初始化");
    }
    
    int request_id = get_available_request_id();
    
    try {
        preprocess(image, input_tensors_[request_id]);
        
        infer_requests_[request_id].set_input_tensor(input_tensors_[request_id]);
        infer_requests_[request_id].infer();
        
        auto output = infer_requests_[request_id].get_output_tensor();
        auto results = postprocess(output, image.size(), image);
        
        log_debug("检测完成 | 发现 %zu 个装甲板", results.size());
        
        return_request_id(request_id);
        return results;
        
    } catch (...) {
        return_request_id(request_id);
        throw;
    }
}

// 异步检测接口
std::future<std::vector<Armor>> ArmorDetector::detect_async(const cv::Mat& image) {
    if (!initialized_) {
        auto promise = std::make_shared<std::promise<std::vector<Armor>>>();
        promise->set_exception(std::make_exception_ptr(
            std::runtime_error("检测器未初始化")));
        return promise->get_future();
    }
    
    int request_id = get_available_request_id();
    
    auto task = std::make_unique<InferenceTask>();
    task->image = image.clone();
    task->original_size = image.size();
    task->promise = std::make_shared<std::promise<std::vector<Armor>>>();
    
    auto future = task->promise->get_future();
    
    try {
        preprocess(task->image, input_tensors_[request_id]);
        
        {
            std::lock_guard<std::mutex> lock(requests_mutex_);
            pending_tasks_[request_id] = std::move(task);
        }
        
        infer_requests_[request_id].set_input_tensor(input_tensors_[request_id]);
        infer_requests_[request_id].start_async();
        
    } catch (...) {
        return_request_id(request_id);
        throw;
    }
    
    return future;
}

void ArmorDetector::preprocess(const cv::Mat& image, ov::Tensor& tensor) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_width_, input_height_));
    
    float* tensor_data = tensor.data<float>();
    
    // 将BGR图像数据转换为RGB并归一化到[0,1]
    for (int h = 0; h < input_height_; h++) {
        const cv::Vec3b* row = resized.ptr<cv::Vec3b>(h);
        for (int w = 0; w < input_width_; w++) {
            const cv::Vec3b& pixel = row[w];
            // 注意OpenCV使用BGR格式，需要转换为RGB
            tensor_data[0 * input_height_ * input_width_ + h * input_width_ + w] = pixel[2] / 255.0f; // R
            tensor_data[1 * input_height_ * input_width_ + h * input_width_ + w] = pixel[1] / 255.0f; // G
            tensor_data[2 * input_height_ * input_width_ + h * input_width_ + w] = pixel[0] / 255.0f; // B
        }
    }
}

// 新的后处理方法，直接生成Armor对象
std::vector<Armor> ArmorDetector::postprocess(const ov::Tensor& output, 
                                             const cv::Size& original_size,
                                             const cv::Mat& original_image) {
    const float* data = output.data<const float>();
    const auto& shape = output.get_shape();
    
    size_t num_features = shape[1];
    size_t num_detections = shape[2];
    int num_classes = static_cast<int>(num_features) - 4;
    
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    
    // 第一步：收集所有检测结果
    for (size_t i = 0; i < num_detections; i++) {
        float x = data[0 * num_detections + i];
        float y = data[1 * num_detections + i];
        float w = data[2 * num_detections + i];
        float h = data[3 * num_detections + i];
        
        float max_score = -1.0f;
        int max_class_id = -1;
        
        for (int c = 0; c < num_classes; c++) {
            float score = utils::sigmoid(data[(4 + c) * num_detections + i]);
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }
        
        if (max_score > conf_threshold_) {
            float scale_x = static_cast<float>(original_size.width) / input_width_;
            float scale_y = static_cast<float>(original_size.height) / input_height_;
            
            float x1 = std::max(0.0f, (x - w / 2) * scale_x);
            float y1 = std::max(0.0f, (y - h / 2) * scale_y);
            float x2 = std::min(static_cast<float>(original_size.width - 1), (x + w / 2) * scale_x);
            float y2 = std::min(static_cast<float>(original_size.height - 1), (y + h / 2) * scale_y);
            
            boxes.emplace_back(
                static_cast<int>(x1), 
                static_cast<int>(y1), 
                static_cast<int>(x2 - x1), 
                static_cast<int>(y2 - y1)
            );
            confidences.push_back(max_score);
            class_ids.push_back(max_class_id);
        }
    }
    
    // 第二步：应用NMS（非极大值抑制）
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, iou_threshold_, indices);
    
    // 第三步：创建Armor对象
    std::vector<Armor> results;
    results.reserve(indices.size());
    
    for (int idx : indices) {
        Armor armor = create_armor_from_detection(
            boxes[idx], 
            confidences[idx], 
            class_ids[idx],
            original_image
        );
        results.push_back(armor);
    }
    
    return results;
}

// 辅助方法：从检测结果创建Armor对象
Armor ArmorDetector::create_armor_from_detection(const cv::Rect& box, 
                                                float confidence, 
                                                int class_id,
                                                const cv::Mat& image) {
    // 使用Armor的构造函数创建对象
    Armor armor(box, confidence, class_id);
    
    // 计算中心点
    armor.center = cv::Point2f(
        box.x + box.width / 2.0f,
        box.y + box.height / 2.0f
    );
    
    // 计算归一化的中心点（相对于图像尺寸）
    armor.center_norm = cv::Point2f(
        armor.center.x / image.cols,
        armor.center.y / image.rows
    );
    
    // 计算四个角点
    armor.points.clear();
    armor.points.push_back(cv::Point2f(box.x, box.y));                          // 左上
    armor.points.push_back(cv::Point2f(box.x + box.width, box.y));            // 右上
    armor.points.push_back(cv::Point2f(box.x + box.width, box.y + box.height)); // 右下
    armor.points.push_back(cv::Point2f(box.x, box.y + box.height));           // 左下
    
    // 提取ROI图像（用于后续分类）
    // 确保边界框在图像范围内
    cv::Rect safe_box = box;
    safe_box.x = std::max(0, safe_box.x);
    safe_box.y = std::max(0, safe_box.y);
    safe_box.width = std::min(safe_box.width, image.cols - safe_box.x);
    safe_box.height = std::min(safe_box.height, image.rows - safe_box.y);
    
    if (safe_box.width > 0 && safe_box.height > 0) {
        armor.roi_image = image(safe_box).clone();
    }
    
    // 根据宽高比初步判断装甲板类型
    float aspect_ratio = static_cast<float>(box.width) / box.height;
    armor.type = (aspect_ratio > 3.5f) ? big : small;
    
    // 标记为已检测但未分类
    armor.classify_confidence = -1.0f;  // 使用-1表示未分类
    armor.is_valid = true;  // 检测有效，但分类结果待定
    
#ifndef NDEBUG
    log_debug("创建装甲板 | 位置: [%d,%d,%dx%d] | 置信度: %.3f | 颜色类别: %d | 类型: %s",
              box.x, box.y, box.width, box.height,
              confidence, class_id,
              armor.type == big ? "大装甲" : "小装甲");
#endif
    
    return armor;
}

int ArmorDetector::get_available_request_id() {
    std::unique_lock<std::mutex> lock(requests_mutex_);
    
    requests_cv_.wait(lock, [this] { 
        return !available_request_ids_.empty() || stop_processing_.load(); 
    });
    
    if (stop_processing_.load()) {
        throw std::runtime_error("检测器正在关闭");
    }
    
    int id = available_request_ids_.front();
    available_request_ids_.pop();
    return id;
}

void ArmorDetector::return_request_id(int id) {
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        available_request_ids_.push(id);
    }
    requests_cv_.notify_one();
}

void ArmorDetector::on_inference_complete(int request_id, std::exception_ptr ex) {
    std::unique_ptr<InferenceTask> task;
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        task = std::move(pending_tasks_[request_id]);
    }
    
    if (!task) {
        return_request_id(request_id);
        return;
    }
    
    try {
        if (ex) {
            std::rethrow_exception(ex);
        }
        
        auto output = infer_requests_[request_id].get_output_tensor();
        auto results = postprocess(output, task->original_size, task->image);
        
        task->promise->set_value(results);
        
    } catch (...) {
        task->promise->set_exception(std::current_exception());
    }
    
    return_request_id(request_id);
}

std::unique_ptr<ArmorDetector> create_detector(const std::string& config_path) {
    try {
        return std::make_unique<ArmorDetector>(config_path);
    } catch (const std::exception& e) {
        log_error("创建检测器失败: %s", e.what());
        return nullptr;
    }
}

} // namespace armor_auto_aim
