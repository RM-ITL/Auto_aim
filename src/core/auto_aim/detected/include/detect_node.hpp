#ifndef ARMOR_AUTO_AIM_ARMOR_DETECTOR_NODE_HPP_
#define ARMOR_AUTO_AIM_ARMOR_DETECTOR_NODE_HPP_

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <vector>
#include <variant>

#include "armor.hpp"
#include "yolo11.hpp"
#include "yolov5.hpp"
#include "draw_tools.hpp"

namespace armor_auto_aim {

// 使用variant来存储不同类型的检测器
using DetectorVariant = std::variant<
    std::unique_ptr<YOLO11Detector>,
    std::unique_ptr<YOLOV5Detector>
>;

class Detector {
public:
    explicit Detector(const std::string& config_path);
    ~Detector();

     Detector(const  Detector&) = delete;
     Detector& operator=(const  Detector&) = delete;

    bool is_initialized() const { return initialized_; }
    void set_debug(bool debug);

    std::vector<Armor> detect(const cv::Mat& image, cv::Mat* annotated_image = nullptr);


    void  visualize_results(cv::Mat & canvas, const std::vector<Visualization> & armors,
    const cv::Point & center_point, int frame_index);

private:
    DetectorVariant detector_;
    std::string detector_type_;  // "yolo11" 或 "yolov5"

    bool initialized_{false};
    bool debug_{false};
    int frame_count_{0};

};

}  // namespace armor_auto_aim

#endif  // ARMOR_AUTO_AIM_ARMOR_DETECTOR_NODE_HPP_
