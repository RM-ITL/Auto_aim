#include "detector.hpp"

#include "logger.hpp"

namespace base_hit
{

Detector::Detector(const std::string & config_path)
{
  infer_ = std::make_unique<OpenvinoInfer>(config_path);
  utils::logger()->info("[Detector] 检测器初始化完成");
}

std::vector<Detector::GreenLight> Detector::detect(const cv::Mat & img)
{
  if (img.empty()) {
    return {};
  }

  return infer_->infer(img);
}

void Detector::visualize(cv::Mat & img, const std::vector<GreenLight> & detections)
{
  for (const auto & det : detections) {
    // 绘制边界框
    cv::Rect rect(
      static_cast<int>(det.box.x),
      static_cast<int>(det.box.y),
      static_cast<int>(det.box.width),
      static_cast<int>(det.box.height));

    cv::rectangle(img, rect, box_color_, box_thickness_);

    // 绘制中心点
    cv::circle(
      img,
      cv::Point(static_cast<int>(det.center.x), static_cast<int>(det.center.y)),
      4, cv::Scalar(0, 0, 255), -1);

    // 绘制置信度文本
    std::string label = cv::format("%.2f", det.score);
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    cv::Point text_origin(rect.x, rect.y - 5);
    if (text_origin.y < text_size.height) {
      text_origin.y = rect.y + rect.height + text_size.height + 5;
    }

    cv::putText(img, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color_, 1);
  }

  // 显示检测数量
  std::string count_text = cv::format("Detections: %zu", detections.size());
  cv::putText(img, count_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
    cv::Scalar(255, 255, 255), 2);
}

}  // namespace base_hit
