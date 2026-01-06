#include "detect_node.hpp"
#include "logger.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace armor_auto_aim {

Detector::Detector(const std::string& config_path){
    try {
        // 读取配置文件获取检测器类型
        auto yaml = YAML::LoadFile(config_path);

        // 默认使用yolo11，如果配置文件中有yolo_name则使用配置的值
        detector_type_ = "yolo11";
        if (yaml["yolo"] && yaml["yolo"]["yolo_name"]) {
            detector_type_ = yaml["yolo"]["yolo_name"].as<std::string>();
        }

        // 根据配置创建对应的检测器
        if (detector_type_ == "yolov5") {
            detector_ = std::make_unique<YOLOV5Detector>(config_path, debug_);
            utils::logger()->info("使用YOLOv5检测器");
        } else {
            // 默认使用YOLO11
            detector_ = std::make_unique<YOLO11Detector>(config_path, debug_);
            utils::logger()->info("使用YOLO11检测器");
        }

        initialized_ = true;
    } catch (const std::exception& e) {
        initialized_ = false;
        throw std::runtime_error("初始化装甲检测模块失败: " + std::string(e.what()));
    }
}

Detector::~Detector() = default;

void Detector::set_debug(bool debug) {
    debug_ = debug;
    // 使用std::visit来调用对应检测器的setDebug方法
    std::visit([debug](auto& detector) {
        if (detector) {
            detector->setDebug(debug);
        }
    }, detector_);
}

std::vector<Armor> Detector::detect(const cv::Mat& image, [[maybe_unused]] cv::Mat* annotated_image) {
    if (!initialized_) {
        throw std::runtime_error("装甲检测模块未初始化");
    }

    // 使用std::visit来调用对应检测器的detect方法，传递frame_count
    int current_frame = frame_count_;
    auto armors = std::visit([&image, current_frame](auto& detector) -> std::vector<Armor> {
        if (detector) {
            return detector->detect(image, current_frame);
        }
        return std::vector<Armor>();
    }, detector_);

    frame_count_++;

    return armors;
}


void  Detector::visualize_results(
  cv::Mat & canvas, const std::vector<Visualization> & armors,
  const cv::Point & center_point, int frame_index)
{
  for (const auto & armor : armors) {
    std::vector<cv::Point2f> points(armor.corners.begin(), armor.corners.end());

    utils::draw_quadrangle_with_corners(canvas, points, utils::colors::GREEN, 2, 3);

    cv::Point2f label_pos;
    label_pos.x = (points[0].x + points[1].x) / 2.0F;
    label_pos.y = std::min(points[0].y, points[1].y);

    std::ostringstream label;
    label << armor_auto_aim::armor_name_to_string(armor.name) << " ,"
          << (armor.type == armor_auto_aim::ArmorType::small ? "small" : "big");

    utils::draw_detection_label(canvas, label.str(), label_pos, utils::colors::GREEN, 0.7, 2);
  }

  utils::draw_crosshair(canvas, center_point, 15, utils::colors::YELLOW, 1);
  if (frame_index > 0) {
    utils::draw_frame_number(canvas, frame_index);
  }
}

}
