#include "node/armor_detector_node.hpp"
#include "logger.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace armor_auto_aim {

Detector::Config Detector::Config::from_yaml(const YAML::Node& node) {
    Config config;

    if (node["yolo"]) {
        const auto& yolo_node = node["yolo"];

        if (yolo_node["data_inspection"]) {
            const auto& inspection = yolo_node["data_inspection"];
            config.enable_visualization = inspection["publish_annotated_image"].as<bool>(config.enable_visualization);
        }

        if (yolo_node["visualization"] && yolo_node["visualization"]["center_point"]) {
            const auto& center = yolo_node["visualization"]["center_point"];
            config.center_point.x = center["x"].as<int>(config.center_point.x);
            config.center_point.y = center["y"].as<int>(config.center_point.y);
        }
    }

    return config;
}

Detector::Detector(const std::string& config_path)
    : config_file_path_(config_path) {
    try {
        YAML::Node yaml_config = YAML::LoadFile(config_file_path_);
        config_ = Config::from_yaml(yaml_config);

        detector_ = std::make_unique<YOLO11Detector>(config_file_path_, debug_);
        initialized_ = true;
    } catch (const std::exception& e) {
        initialized_ = false;
        throw std::runtime_error("初始化装甲检测模块失败: " + std::string(e.what()));
    }
}

Detector::~Detector() = default;

void Detector::set_debug(bool debug) {
    debug_ = debug;
    if (detector_) {
        detector_->setDebug(debug);
    }
}

std::vector<Armor> Detector::detect(const cv::Mat& image, cv::Mat* annotated_image) {
    if (!initialized_) {
        throw std::runtime_error("装甲检测模块未初始化");
    }

    auto armors = detector_->detect(image);
    frame_count_++;

    // if (annotated_image && config_.enable_visualization && !image.empty()) {
    //     if (annotated_image->empty()) {
    //         *annotated_image = image;
    //     }
    //     visualize_results(*annotated_image, armors);
    // }

    // utils::logger()->debug(
    //     "检测到的armor的点的数据是:[({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f})]",
    //     armors.points[0].x, armors.points[0].y,
    //     armors.points[1].x, armors.points[1].y,
    //     armors.points[2].x, armors.points[2].y,
    //     armors.points[3].x, armors.points[3].y
    // );

    // // 遍历每个检测到的装甲板
    // for (const auto& armor : armors) {
    //     utils::logger()->debug(
    //         "检测到的装甲板的置信度是：{:.2f}, class_id: {}",
    //         armor.detection_confidence,
    //         armor.class_id
    //     );
    // }

    return armors;
}

// void Detector::visualize_results(cv::Mat& canvas, const std::vector<Armor>& armors) const {
//     for (const auto& armor : armors) {
//         if (armor.points.size() == 4) {
//             utils::draw_quadrangle_with_corners(canvas, armor.points, utils::colors::GREEN, 2, 3);

//             cv::Point2f label_pos;
//             label_pos.x = (armor.points[0].x + armor.points[1].x) / 2.0f;
//             label_pos.y = std::min(armor.points[0].y, armor.points[1].y);

//             std::ostringstream label;
//             label << std::fixed << std::setprecision(2)
//                   << armor.detection_confidence << " "
//                   << COLORS[armor.color] << ", "
//                   << armor.getNameString() << " ,"
//                   << (armor.type == small ? "small" : "big");

//             utils::draw_detection_label(canvas, label.str(), label_pos, utils::colors::GREEN, 0.7, 2);
//         }
//     }

//     utils::draw_crosshair(canvas, config_.center_point, 15, utils::colors::YELLOW, 1);
//     utils::draw_frame_number(canvas, frame_count_);
// }

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

}  // namespace armor_auto_aim
