#include "openvino_infer.hpp"

#include <yaml-cpp/yaml.h>
#include <fmt/format.h>

#include "logger.hpp"

namespace base_hit
{

OpenvinoInfer::OpenvinoInfer(const std::string & config_path)
{
  // 加载 YAML 配置
  auto yaml = YAML::LoadFile(config_path);
  const auto & cfg = yaml["Base_Hit"];

  std::string xml_path = cfg["Openvino_XML"].as<std::string>();
  device_ = cfg["device"].as<std::string>("CPU");
  score_threshold_ = cfg["score_threshold"].as<float>(0.5f);
  nms_threshold_ = cfg["nms_threshold"].as<float>(0.45f);

  utils::logger()->info("[OpenvinoInfer] 加载模型: {}, 设备: {}", xml_path, device_);

  // 读取模型
  auto model = core_.read_model(xml_path);

  // 从模型获取实际输入尺寸
  // 原始模型通常是 NCHW 格式 [N, C, H, W]
  auto input_shape = model->input().get_shape();
  if (input_shape.size() == 4) {
    // NCHW: [1, 3, 384, 640] -> 宽=shape[3], 高=shape[2]
    input_size_ = cv::Size(static_cast<int>(input_shape[3]), static_cast<int>(input_shape[2]));
  }
  utils::logger()->info("[OpenvinoInfer] 从模型读取输入尺寸: {}x{} (宽x高)", input_size_.width, input_size_.height);

  // 配置预处理
  ov::preprocess::PrePostProcessor ppp(model);

  ppp.input()
    .tensor()
    .set_element_type(ov::element::u8)
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  ppp.input()
    .preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale({255.f, 255.f, 255.f});

  ppp.input()
    .model()
    .set_layout("NCHW");

  // 配置输出
  for (size_t i = 0; i < model->outputs().size(); ++i) {
    ppp.output(i).tensor().set_element_type(ov::element::f32);
  }

  model = ppp.build();

  // 编译模型
  compiled_model_ = core_.compile_model(model, device_);
  infer_request_ = compiled_model_.create_infer_request();

  // 输出模型详细信息
  utils::logger()->info("[OpenvinoInfer] 模型加载完成");
  utils::logger()->info("[OpenvinoInfer] ===== 模型详细信息 =====");

  // 输入信息
  auto input = compiled_model_.input();
  auto compiled_input_shape = input.get_shape();
  utils::logger()->info("[OpenvinoInfer] 输入名称: {}", input.get_any_name());
  utils::logger()->info("[OpenvinoInfer] 输入形状: [{}]",
    fmt::format("{}, {}, {}, {}", compiled_input_shape[0], compiled_input_shape[1], compiled_input_shape[2], compiled_input_shape[3]));
  utils::logger()->info("[OpenvinoInfer] 输入类型: {}", input.get_element_type().get_type_name());

  // 输出信息
  utils::logger()->info("[OpenvinoInfer] 输出数量: {}", compiled_model_.outputs().size());
  for (size_t i = 0; i < compiled_model_.outputs().size(); ++i) {
    auto output = compiled_model_.output(i);
    auto output_shape = output.get_shape();
    std::string shape_str = "[";
    for (size_t j = 0; j < output_shape.size(); ++j) {
      shape_str += std::to_string(output_shape[j]);
      if (j < output_shape.size() - 1) shape_str += ", ";
    }
    shape_str += "]";
    utils::logger()->info("[OpenvinoInfer] 输出{} - 名称: {}, 形状: {}, 类型: {}",
      i, output.get_any_name(), shape_str, output.get_element_type().get_type_name());
  }
  utils::logger()->info("[OpenvinoInfer] ========================");
}

std::vector<OpenvinoInfer::Detection> OpenvinoInfer::infer(const cv::Mat & src)
{
  if (src.empty()) {
    return {};
  }

  // 预处理
  LetterBoxInfo lb_info = letterBox(src);

  // 调试：保存预处理后的图像
  static int debug_count = 0;
  if (debug_count < 3) {
    utils::logger()->info("[OpenvinoInfer] 原图尺寸: {}x{}, 预处理后尺寸: {}x{}",
      src.cols, src.rows, lb_info.resized_image.cols, lb_info.resized_image.rows);
    utils::logger()->info("[OpenvinoInfer] scale={:.4f}, pad_w={}, pad_h={}",
      lb_info.scale, lb_info.pad_w, lb_info.pad_h);
  }

  // 创建输入张量
  auto * input_data = reinterpret_cast<uint8_t *>(lb_info.resized_image.data);
  ov::Tensor input_tensor(
    compiled_model_.input().get_element_type(),
    compiled_model_.input().get_shape(),
    input_data);

  // 推理
  infer_request_.set_input_tensor(input_tensor);
  infer_request_.infer();

  // 获取输出
  const ov::Tensor & output_tensor = infer_request_.get_output_tensor(0);
  ov::Shape output_shape = output_tensor.get_shape();
  auto * output_data = output_tensor.data<float>();

  // 调试：输出原始数据信息
  if (debug_count < 3) {
    utils::logger()->info("[OpenvinoInfer] 输出形状: [{}, {}, {}]",
      output_shape.size() > 0 ? output_shape[0] : 0,
      output_shape.size() > 1 ? output_shape[1] : 0,
      output_shape.size() > 2 ? output_shape[2] : 0);

    // 打印前几个检测框的原始数据
    size_t num_detections = output_shape.size() > 1 ? output_shape[1] : 0;
    size_t detection_size = output_shape.size() > 2 ? output_shape[2] : 0;
    utils::logger()->info("[OpenvinoInfer] 检测数量: {}, 每个检测的特征数: {}", num_detections, detection_size);

    // 找出置信度最高的几个检测
    float max_conf = 0;
    int max_idx = -1;
    for (size_t i = 0; i < std::min(num_detections, size_t(100)); ++i) {
      float * det = &output_data[i * detection_size];
      if (detection_size > 4 && det[4] > max_conf) {
        max_conf = det[4];
        max_idx = static_cast<int>(i);
      }
    }
    if (max_idx >= 0 && detection_size > 4) {
      float * det = &output_data[max_idx * detection_size];
      // 计算真正的置信度
      float obj = det[4];
      float * cls = &det[5];
      float max_cls = 0;
      int max_cls_id = 0;
      for (int c = 0; c < 9; ++c) {
        if (cls[c] > max_cls) {
          max_cls = cls[c];
          max_cls_id = c;
        }
      }
      float real_conf = obj * max_cls;
      utils::logger()->info("[OpenvinoInfer] 最高objectness检测[{}]: cx={:.2f}, cy={:.2f}, w={:.2f}, h={:.2f}",
        max_idx, det[0], det[1], det[2], det[3]);
      utils::logger()->info("[OpenvinoInfer] objectness={:.4f}, 最高类别[{}]={:.4f}, 真实置信度={:.6f}",
        obj, max_cls_id, max_cls, real_conf);
    } else {
      utils::logger()->warn("[OpenvinoInfer] 没有找到有效检测！最大置信度: {:.4f}", max_conf);
    }
    debug_count++;
  }

  // 解析输出
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<int> class_ids;
  std::vector<Detection> raw_detections;

  // 输出格式: [1, num_detections, 27]
  // 27 = 4(box: cx,cy,w,h) + 1(objectness) + 9(classes) + 4(colors) + 9(其他)
  // 或者可能是: 4(box) + 9(classes) + 4(colors) + ... (没有单独objectness)

  size_t num_detections = output_shape.size() > 1 ? output_shape[1] : 0;
  size_t detection_size = output_shape.size() > 2 ? output_shape[2] : 0;

  for (size_t i = 0; i < num_detections; ++i) {
    float * detection = &output_data[i * detection_size];

    // 尝试两种解析方式：
    // 方式1: det[4] 是 objectness，det[5:14] 是类别分数
    // 方式2: det[4:13] 直接是类别分数（没有 objectness）

    float objectness = detection[4];
    float * class_scores_ptr = &detection[5];  // 假设从 det[5] 开始是类别分数
    int num_classes = 9;

    // 找最大类别分数
    int best_class_id = 0;
    float best_class_score = class_scores_ptr[0];
    for (int c = 1; c < num_classes; ++c) {
      if (class_scores_ptr[c] > best_class_score) {
        best_class_score = class_scores_ptr[c];
        best_class_id = c;
      }
    }

    // 计算最终置信度
    // 标准 YOLO 格式: final_conf = objectness * max_class_score
    float confidence = objectness * best_class_score;

    if (confidence < score_threshold_) {
      continue;
    }

    // 只保留 class_id == 8 的目标（根据原代码逻辑）
    if (best_class_id != 8) {
      continue;
    }

    float cx = detection[0];
    float cy = detection[1];
    float w = detection[2];
    float h = detection[3];

    int left = static_cast<int>(cx - w / 2);
    int top = static_cast<int>(cy - h / 2);

    boxes.emplace_back(left, top, static_cast<int>(w), static_cast<int>(h));
    confidences.emplace_back(confidence);
    class_ids.emplace_back(best_class_id);

    Detection det;
    det.box = cv::Rect2d(cx - w / 2, cy - h / 2, w, h);
    det.center = cv::Point2d(cx, cy);
    det.score = confidence;
    det.class_id = best_class_id;
    raw_detections.emplace_back(det);
  }

  // NMS
  std::vector<int> nms_indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, nms_indices);

  std::vector<Detection> results;
  results.reserve(nms_indices.size());
  for (int idx : nms_indices) {
    results.emplace_back(raw_detections[idx]);
  }

  // 还原坐标到原图尺寸
  restoreCoords(results, lb_info);

  return results;
}

OpenvinoInfer::LetterBoxInfo OpenvinoInfer::letterBox(const cv::Mat & src)
{
  int src_w = src.cols;
  int src_h = src.rows;

  float scale_w = static_cast<float>(input_size_.width) / src_w;
  float scale_h = static_cast<float>(input_size_.height) / src_h;
  float scale = std::min(scale_w, scale_h);

  int new_w = static_cast<int>(src_w * scale);
  int new_h = static_cast<int>(src_h * scale);

  cv::Mat resized;
  cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  cv::Mat canvas(input_size_, CV_8UC3, cv::Scalar(128, 128, 128));

  int pad_w = (input_size_.width - new_w) / 2;
  int pad_h = (input_size_.height - new_h) / 2;

  resized.copyTo(canvas(cv::Rect(pad_w, pad_h, new_w, new_h)));

  return {canvas, scale, pad_w, pad_h};
}

void OpenvinoInfer::restoreCoords(
  std::vector<Detection> & detections, const LetterBoxInfo & info)
{
  for (auto & det : detections) {
    det.box.x = (det.box.x - info.pad_w) / info.scale;
    det.box.y = (det.box.y - info.pad_h) / info.scale;
    det.box.width /= info.scale;
    det.box.height /= info.scale;

    det.center.x = (det.center.x - info.pad_w) / info.scale;
    det.center.y = (det.center.y - info.pad_h) / info.scale;
  }
}

}  // namespace base_hit
