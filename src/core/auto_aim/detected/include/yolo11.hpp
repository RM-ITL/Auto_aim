#ifndef ARMOR_AUTO_AIM__YOLO11_DETECTOR_HPP
#define ARMOR_AUTO_AIM__YOLO11_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "armor.hpp"

namespace armor_auto_aim
{

class YOLO11Detector
{
public:
    YOLO11Detector(const std::string& config_path, bool debug = false);
    std::vector<Armor> detect(const cv::Mat& image, int frame_count = -1);
    void setDebug(bool debug) { debug_ = debug; }

private:
    // OpenVINO核心
    ov::Core core_;
    ov::CompiledModel compiled_model_;

    // 配置参数
    std::string device_;
    std::string model_path_;
    std::string enemy_color_ = "red";  // 新增敌方颜色配置
    bool debug_;

    // 检测参数
    const int class_num_ = 38;
    float nms_threshold_ = 0.3f;
    float score_threshold_ = 0.7f;
    double min_confidence_ = 0.8;

    // 核心处理函数
    std::vector<Armor> parse(double scale, cv::Mat& output, const cv::Mat& bgr_img, int frame_count);
    void sort_keypoints(std::vector<cv::Point2f>& keypoints);
    bool check_name(const Armor& armor) const;
    bool check_type(const Armor& armor) const;
    cv::Point2f get_center_norm(const cv::Mat& bgr_img, const cv::Point2f& center) const;
    void draw_detections(const cv::Mat& img, const std::vector<Armor>& armors, int frame_count) const;
};

} // namespace armor_auto_aim

#endif