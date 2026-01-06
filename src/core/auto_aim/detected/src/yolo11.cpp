#include "yolo11.hpp"
#include "draw_tools.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <iostream>

namespace armor_auto_aim
{

YOLO11Detector::YOLO11Detector(const std::string& config_path, bool debug)
    : debug_(debug)
{
    // 读取配置文件
    auto yaml = YAML::LoadFile(config_path);
    
    // 从yolo11节点下读取配置参数
    if (!yaml["yolo"]) {
        throw std::runtime_error("配置文件中缺少'yolo11'节点");
    }
    
    const auto& yolo11_config = yaml["yolo"];
    
    // 读取模型路径和设备配置
    model_path_ = yolo11_config["yolo11_model_path"].as<std::string>();
    device_ = yolo11_config["device"].as<std::string>("CPU");
    
    // 读取检测阈值参数
    min_confidence_ = yolo11_config["min_confidence"].as<double>(0.8);
    score_threshold_ = yolo11_config["score_threshold"].as<float>(0.7f);
    nms_threshold_ = yolo11_config["nms_threshold"].as<float>(0.3f);
    
    // 读取其他配置（虽然暂时可能用不到，但预留接口）
    if (yolo11_config["enemy_color"]) {
        enemy_color_ = yolo11_config["enemy_color"].as<std::string>("red");
    }
    
    // 检查模型文件是否存在
    if (!std::filesystem::exists(model_path_)) {
        throw std::runtime_error("模型文件不存在: " + model_path_);
    }
    
    // 初始化OpenVINO模型
    auto model = core_.read_model(model_path_);
    ov::preprocess::PrePostProcessor ppp(model);
    auto& input = ppp.input();
    
    input.tensor()
        .set_element_type(ov::element::u8)
        .set_shape({1, 640, 640, 3})
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    
    input.model().set_layout("NCHW");
    
    input.preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale(255.0);
    
    model = ppp.build();
    compiled_model_ = core_.compile_model(
        model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

std::vector<Armor> YOLO11Detector::detect(const cv::Mat& raw_img, int frame_count)
{
    if (raw_img.empty()) {
        if (debug_) {
            std::cerr << "Empty img!, camera drop!" << std::endl;
        }
        return std::vector<Armor>();
    }

    // 预处理 - 与原始逻辑完全相同
    auto x_scale = static_cast<double>(640) / raw_img.rows;
    auto y_scale = static_cast<double>(640) / raw_img.cols;
    auto scale = std::min(x_scale, y_scale);
    auto h = static_cast<int>(raw_img.rows * scale);
    auto w = static_cast<int>(raw_img.cols * scale);

    auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    auto roi = cv::Rect(0, 0, w, h);
    cv::resize(raw_img, input(roi), {w, h});
    ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

    // 推理 - 与原始逻辑完全相同
    auto infer_request = compiled_model_.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    // 后处理 - 与原始逻辑完全相同
    auto output_tensor = infer_request.get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

    return parse(scale, output, raw_img, frame_count);
}

std::vector<Armor> YOLO11Detector::parse(
    double scale, cv::Mat& output, const cv::Mat& bgr_img, int frame_count)
{
    // 转置输出 - 原始逻辑
    cv::transpose(output, output);

    std::vector<int> ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point2f>> armors_key_points;

    for (int r = 0; r < output.rows; r++) {
        auto xywh = output.row(r).colRange(0, 4);
        auto scores = output.row(r).colRange(4, 4 + class_num_);
        auto one_key_points = output.row(r).colRange(4 + class_num_, 50);

        std::vector<cv::Point2f> armor_key_points;

        double score;
        cv::Point max_point;
        cv::minMaxLoc(scores, nullptr, &score, nullptr, &max_point);

        if (score < score_threshold_) continue;

        auto x = xywh.at<float>(0);
        auto y = xywh.at<float>(1);
        auto w = xywh.at<float>(2);
        auto h = xywh.at<float>(3);
        auto left = static_cast<int>((x - 0.5 * w) / scale);
        auto top = static_cast<int>((y - 0.5 * h) / scale);
        auto width = static_cast<int>(w / scale);
        auto height = static_cast<int>(h / scale);

        for (int i = 0; i < 4; i++) {
            float x = one_key_points.at<float>(0, i * 2 + 0) / scale;
            float y = one_key_points.at<float>(0, i * 2 + 1) / scale;
            cv::Point2f kp = {x, y};
            armor_key_points.push_back(kp);
        }

        ids.emplace_back(max_point.x);
        confidences.emplace_back(score);
        boxes.emplace_back(left, top, width, height);
        armors_key_points.emplace_back(armor_key_points);
    }

    // NMS - 原始逻辑
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

    std::vector<Armor> armors;
    for (const auto& i : indices) {
        sort_keypoints(armors_key_points[i]);
        // 使用新的Armor构造函数
        Armor armor(ids[i], confidences[i], boxes[i], armors_key_points[i]);
        armors.push_back(armor);
    }

    // 验证和过滤 - 原始逻辑
    auto it = armors.begin();
    while (it != armors.end()) {
        if (!check_name(*it)) {
            it = armors.erase(it);
            continue;
        }

        if (!check_type(*it)) {
            it = armors.erase(it);
            continue;
        }

        it->center_norm = get_center_norm(bgr_img, it->center);
        ++it;
    }

    if (debug_) {
        draw_detections(bgr_img, armors, frame_count);
    }

    return armors;
}

void YOLO11Detector::sort_keypoints(std::vector<cv::Point2f>& keypoints)
{
    // 原始的关键点排序逻辑
    if (keypoints.size() != 4) {
        std::cout << "beyond 4!!" << std::endl;
        return;
    }
    
    std::sort(keypoints.begin(), keypoints.end(), 
        [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y;
        });
    
    std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
    std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};
    
    std::sort(top_points.begin(), top_points.end(),
        [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
        });
    
    std::sort(bottom_points.begin(), bottom_points.end(),
        [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
        });
    
    keypoints[0] = top_points[0];     // top-left
    keypoints[1] = top_points[1];     // top-right  
    keypoints[2] = bottom_points[1];  // bottom-right
    keypoints[3] = bottom_points[0];  // bottom-left
}

bool YOLO11Detector::check_name(const Armor& armor) const
{
    // 原始的名称检查逻辑
    auto name_ok = armor.name != ArmorName::not_armor;
    auto confidence_ok = armor.confidence > min_confidence_;
    
    return name_ok && confidence_ok;
}

bool YOLO11Detector::check_type(const Armor& armor) const
{
    // 原始的类型检查逻辑
    auto name_ok = (armor.type == ArmorType::small)
                    ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                    : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                       armor.name != ArmorName::outpost);
    
    return name_ok;
}

cv::Point2f YOLO11Detector::get_center_norm(const cv::Mat& bgr_img, const cv::Point2f& center) const
{
    // 原始的归一化中心点计算
    auto h = bgr_img.rows;
    auto w = bgr_img.cols;
    return {center.x / w, center.y / h};
}

void YOLO11Detector::draw_detections(
    const cv::Mat& img, const std::vector<Armor>& armors, int frame_count) const
{
    // 使用新的绘图工具，但保持相同的可视化效果
    auto detection = img.clone();

    // 显示帧号
    if (frame_count >= 0) {
        utils::draw_text(detection, std::to_string(frame_count), {10, 30}, {255, 255, 255});
    }

    for (const auto& armor : armors) {
        // 根据颜色设置绘制颜色
        cv::Scalar color;
        switch (armor.color) {
            case red: color = cv::Scalar(0, 0, 255); break;
            case blue: color = cv::Scalar(255, 0, 0); break;
            case purple: color = cv::Scalar(255, 0, 255); break;
            default: color = cv::Scalar(128, 128, 128); break;
        }

        // 构建标签
        std::string info = std::to_string(armor.confidence) + " " +
                          COLORS[armor.color] + " " +
                          armor.getNameString() + " " +
                          ARMOR_TYPES[armor.type];

        utils::draw_points(detection, armor.points, color);
        utils::draw_label(detection, info, cv::Point(armor.center.x, armor.center.y), color);
    }

}

} // namespace armor_auto_aim