#ifndef ARMOR_AUTO_AIM_ARMOR_DETECTOR_NODE_HPP_
#define ARMOR_AUTO_AIM_ARMOR_DETECTOR_NODE_HPP_

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <vector>

#include "common/armor.hpp"
#include "detector/yolo11.hpp"
#include "draw_tools.hpp"

namespace armor_auto_aim {

class Detector {
public:
    struct Config {
        bool enable_visualization{true};
        cv::Point center_point{640, 384};

        static Config from_yaml(const YAML::Node& node);
    };

    explicit Detector(const std::string& config_path);
    ~Detector();

     Detector(const  Detector&) = delete;
     Detector& operator=(const  Detector&) = delete;

    bool is_initialized() const { return initialized_; }
    void set_debug(bool debug);

    std::vector<Armor> detect(const cv::Mat& image, cv::Mat* annotated_image = nullptr);

    const Config& config() const { return config_; }

    void  visualize_results(cv::Mat & canvas, const std::vector<Visualization> & armors, 
    const cv::Point & center_point, int frame_index);

private:
    Config config_;
    std::string config_file_path_;
    std::unique_ptr<YOLO11Detector> detector_;

    bool initialized_{false};
    bool debug_{false};
    int frame_count_{0};

};

}  // namespace armor_auto_aim

#endif  // ARMOR_AUTO_AIM_ARMOR_DETECTOR_NODE_HPP_
