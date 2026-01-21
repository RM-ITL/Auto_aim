#ifndef BASE_HIT_DETECTOR_HPP_
#define BASE_HIT_DETECTOR_HPP_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "openvino_infer.hpp"

namespace base_hit
{

class Detector
{
public:
  using GreenLight = OpenvinoInfer::GreenLight;

  explicit Detector(const std::string & config_path);
  ~Detector() = default;

  // 检测接口
  std::vector<GreenLight> detect(const cv::Mat & img);

  // 可视化：在图像上绘制检测结果
  void visualize(cv::Mat & img, const std::vector<GreenLight> & detections);

private:
  std::unique_ptr<OpenvinoInfer> infer_;

  // 可视化颜色
  cv::Scalar box_color_{0, 255, 0};  // 绿色框
  int box_thickness_{2};
};

}  // namespace base_hit

#endif  // BASE_HIT_DETECTOR_HPP_
