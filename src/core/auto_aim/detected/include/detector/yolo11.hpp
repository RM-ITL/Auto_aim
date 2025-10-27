#ifndef ARMOR_AUTO_AIM__YOLO11_DETECTOR_HPP
#define ARMOR_AUTO_AIM__YOLO11_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "common/armor.hpp"

namespace armor_auto_aim
{

class YOLO11Detector
{
public:
    YOLO11Detector(const std::string& config_path, bool debug = false);
    std::vector<Armor> detect(const cv::Mat& image);
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
    
    // 检测参数 - 现在从配置文件读取
    const int class_num_ = 38;
    float nms_threshold_ = 0.3f;      // 不再是const，可从配置读取
    float score_threshold_ = 0.7f;    // 不再是const，可从配置读取
    double min_confidence_ = 0.8;
    
    // 核心处理函数
    std::vector<Armor> parse(double scale, cv::Mat& output, const cv::Mat& bgr_img);
    void sort_keypoints(std::vector<cv::Point2f>& keypoints);
    bool check_name(const Armor& armor) const;
    bool check_type(const Armor& armor) const;
    cv::Point2f get_center_norm(const cv::Mat& bgr_img, const cv::Point2f& center) const;
    void draw_detections(const cv::Mat& img, const std::vector<Armor>& armors) const;
};

} // namespace armor_auto_aim

#endif