#ifndef BASE_HIT_OPENVINO_INFER_HPP_
#define BASE_HIT_OPENVINO_INFER_HPP_

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace auto_base
{

class OpenvinoInfer
{
public:
  struct GreenLight
  {
    cv::Rect2d box; // 这里的box是center_x，center_y，w，h的构成
    cv::Point2d center;
    double score;
    int class_id;
  };

  explicit OpenvinoInfer(const std::string & config_path);
  ~OpenvinoInfer() = default;

  // 推理接口：输入图像，返回检测结果
  std::vector<GreenLight> infer(const cv::Mat & src);

private:
  // LetterBox 预处理
  struct LetterBoxInfo
  {
    cv::Mat resized_image;
    float scale;
    int pad_w;
    int pad_h;
  };

  LetterBoxInfo letterBox(const cv::Mat & src);

  // 将检测框坐标还原到原图尺寸
  void restoreCoords(std::vector<GreenLight> & detections, const LetterBoxInfo & info);

  ov::Core core_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;

  // 配置参数
  cv::Size input_size_;
  float score_threshold_;
  float nms_threshold_;
  std::string device_;
};

}  // namespace base_hit

#endif  // BASE_HIT_OPENVINO_INFER_HPP_
