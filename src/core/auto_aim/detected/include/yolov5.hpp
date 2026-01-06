#ifndef AUTO_AIM__YOLOV5_HPP
#define AUTO_AIM__YOLOV5_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "armor.hpp"
#include "detector.hpp"

namespace armor_auto_aim
{

class YOLOV5Detector
{
public:
  YOLOV5Detector(const std::string & config_path, bool debug = false);

  std::vector<Armor> detect(const cv::Mat & bgr_img, int frame_count = -1);
  void setDebug(bool debug) { debug_ = debug; }

private:
  std::string device_, model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_, use_traditional_;

  const int class_num_ = 13;
  float nms_threshold_ = 0.3f;
  float score_threshold_ = 0.7f;
  double min_confidence_, binary_threshold_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  Traditional_Detector detector_;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  std::vector<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::vector<Armor> & armors, int frame_count) const;
  double sigmoid(double x);
};

}  // namespace armor_auto_aim

#endif  //AUTO_AIM__YOLOV5_HPP