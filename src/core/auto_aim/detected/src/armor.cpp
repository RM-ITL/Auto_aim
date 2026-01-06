#include "armor.hpp"
#include <cmath>

namespace armor_auto_aim
{

// clang-format off
// 装甲板属性表：定义了所有可能的装甲板类型
// 格式：{颜色, 装甲板名称, 装甲板大小}
const std::vector<std::tuple<Color, ArmorName, ArmorType>> armor_properties = {
  {blue, sentry, small},     {red, sentry, small},     {extinguish, sentry, small},
  {blue, one, small},        {red, one, small},        {extinguish, one, small},
  {blue, two, small},        {red, two, small},        {extinguish, two, small},
  {blue, three, small},      {red, three, small},      {extinguish, three, small},
  {blue, four, small},       {red, four, small},       {extinguish, four, small},
  {blue, five, small},       {red, five, small},       {extinguish, five, small},
  {blue, outpost, small},    {red, outpost, small},    {extinguish, outpost, small},
  {blue, base, big},         {red, base, big},         {extinguish, base, big},      {purple, base, big},       
  {blue, base, small},       {red, base, small},       {extinguish, base, small},    {purple, base, small},    
  {blue, three, big},        {red, three, big},        {extinguish, three, big}, 
  {blue, four, big},         {red, four, big},         {extinguish, four, big},  
  {blue, five, big},         {red, five, big},         {extinguish, five, big}};
// clang-format on


Lightbar::Lightbar(const cv::RotatedRect & rotated_rect, std::size_t id)
: id(id), rotated_rect(rotated_rect)
{
  std::vector<cv::Point2f> corners(4);
  rotated_rect.points(&corners[0]);
  std::sort(corners.begin(), corners.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.y < b.y;
  });

  center = rotated_rect.center;
  top = (corners[0] + corners[1]) / 2;
  bottom = (corners[2] + corners[3]) / 2;
  top2bottom = bottom - top;

  points.emplace_back(top);
  points.emplace_back(bottom);

  width = cv::norm(corners[0] - corners[1]);
  angle = std::atan2(top2bottom.y, top2bottom.x);
  angle_error = std::abs(angle - CV_PI / 2);
  length = cv::norm(top2bottom);
  ratio = length / width;
}

// 传统检测器构造函数实现（从两个Lightbar构造Armor）
Armor::Armor(const Lightbar & left, const Lightbar & right)
: left(left), right(right), duplicated(false)
{
  color = left.color;
  center = (left.center + right.center) / 2;

  points.emplace_back(left.top);
  points.emplace_back(right.top);
  points.emplace_back(right.bottom);
  points.emplace_back(left.bottom);

  auto left2right = right.center - left.center;
  auto width = cv::norm(left2right);
  auto max_lightbar_length = std::max(left.length, right.length);
  auto min_lightbar_length = std::min(left.length, right.length);
  ratio = width / max_lightbar_length;
  side_ratio = max_lightbar_length / min_lightbar_length;

  auto roll = std::atan2(left2right.y, left2right.x);
  auto left_rectangular_error = std::abs(left.angle - roll - CV_PI / 2);
  auto right_rectangular_error = std::abs(right.angle - roll - CV_PI / 2);
  rectangular_error = std::max(left_rectangular_error, right_rectangular_error);
}

// YOLO11构造函数实现
// 通过YOLO检测结果直接获取装甲板的所有属性，包括颜色、名称、类型和优先级
Armor::Armor(int yolo_class_id, float confidence, const cv::Rect& box,
             const std::vector<cv::Point2f>& keypoints)
    : class_id(yolo_class_id),
      color_id(-1),
      num_id(-1),
      confidence(confidence),  // 初始化引用
      color(red),
      name(not_armor),
      type(small),
      rank(getRankFromName(not_armor)),  // 初始化为默认rank
      box(box),
      points(keypoints),
      ratio(0.0f),
      rectangular_error(0.0f),
      is_valid(true),
      duplicated(false) {
    
    // 验证关键点数量（装甲板应该有4个角点）
    if (keypoints.size() != 4) {
        is_valid = false;
        return;
    }
    
    // 计算装甲板中心点（4个角点的平均值）
    center = (keypoints[0] + keypoints[1] + keypoints[2] + keypoints[3]) / 4;
    
    // 计算装甲板的边长
    auto left_width = cv::norm(keypoints[0] - keypoints[3]);    // 左边长度
    auto right_width = cv::norm(keypoints[1] - keypoints[2]);   // 右边长度
    auto max_width = std::max(left_width, right_width);         // 取较大的宽度
    
    auto top_length = cv::norm(keypoints[0] - keypoints[1]);    // 上边长度
    auto bottom_length = cv::norm(keypoints[3] - keypoints[2]); // 下边长度
    auto max_length = std::max(top_length, bottom_length);      // 取较大的长度
    
    // 计算长宽比（用于验证装甲板形状）
    if (max_width > 0) {
        ratio = max_length / max_width;
    }
    
    // 计算矩形度误差（用于评估装甲板的矩形程度）
    auto left_center = (keypoints[0] + keypoints[3]) / 2;   // 左边中点
    auto right_center = (keypoints[1] + keypoints[2]) / 2;  // 右边中点
    auto left2right = right_center - left_center;           // 左到右的向量
    auto roll = std::atan2(left2right.y, left2right.x);     // 装甲板的倾斜角度
    
    // 计算左右边与理想垂直方向的误差
    auto left_rectangular_error = std::abs(
        std::atan2((keypoints[3] - keypoints[0]).y, (keypoints[3] - keypoints[0]).x) - 
        roll - CV_PI / 2);
    
    auto right_rectangular_error = std::abs(
        std::atan2((keypoints[2] - keypoints[1]).y, (keypoints[2] - keypoints[1]).x) - 
        roll - CV_PI / 2);
    
    rectangular_error = std::max(left_rectangular_error, right_rectangular_error);
    
    // 解析YOLO类别ID到具体的装甲板属性
    if (yolo_class_id >= 0 &&
        static_cast<std::size_t>(yolo_class_id) < armor_properties.size()) {
        // 从属性表中获取对应的颜色、名称和类型
        auto [parsed_color, parsed_name, parsed_type] = armor_properties[yolo_class_id];
        
        this->color = parsed_color;
        this->name = parsed_name;
        this->type = parsed_type;
        
        // 关键修改：通过映射表自动设置rank值
        // rank值由装甲板名称决定，实现了动态优先级分配
        this->rank = getRankFromName(parsed_name);
        
        // 为了兼容性，生成color_id
        // 注意：这里的映射可能与传统检测器使用的不同
        if (parsed_color == blue) {
            this->color_id = 0;
        } else if (parsed_color == red) {
            this->color_id = 1;
        } else if (parsed_color == extinguish) {
            this->color_id = 2;
        } else if (parsed_color == purple) {
            this->color_id = 3;
        }
    } else {
        // 无效的class_id，保持默认的not_armor和对应的rank值
        this->is_valid = false;
        this->rank = getRankFromName(not_armor);
    }
}

// yolov5构造函数
// YOLOv5输出格式：color_id (0=blue, 1=red, 2=extinguish), num_id (0=sentry, 1-5=数字, 6=outpost, 7=base)
Armor::Armor(
  int color_id, int num_id, float confidence, const cv::Rect & box,
  std::vector<cv::Point2f> armor_keypoints)
: class_id(-1),            // YOLOv5不使用统一的class_id，设为-1
  color_id(color_id),      // 保存颜色ID
  num_id(num_id),          // 保存数字ID
  confidence(confidence),
  color(red),              // 默认值，后续会被覆盖
  name(not_armor),         // 默认值，后续会被覆盖
  type(small),             // 默认值，后续会被覆盖
  rank(getRankFromName(not_armor)),
  box(box),
  points(armor_keypoints),
  ratio(0.0),
  side_ratio(0.0),
  rectangular_error(0.0),
  is_valid(true),
  duplicated(false)
{
  // 验证关键点数量
  if (armor_keypoints.size() != 4) {
    is_valid = false;
    return;
  }

  center = (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
  auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
  auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
  auto max_width = std::max(left_width, right_width);
  auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
  auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
  auto max_length = std::max(top_length, bottom_length);
  auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2;
  auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2;
  auto left2right = right_center - left_center;
  auto roll = std::atan2(left2right.y, left2right.x);
  auto left_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[3] - armor_keypoints[0]).y, (armor_keypoints[3] - armor_keypoints[0]).x) -
    roll - CV_PI / 2);
  auto right_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[2] - armor_keypoints[1]).y, (armor_keypoints[2] - armor_keypoints[1]).x) -
    roll - CV_PI / 2);
  rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

  // 计算宽高比
  if (max_width > 0) {
    ratio = max_length / max_width;
  }

  // 解析颜色
  switch (color_id) {
    case 0: color = Color::blue; break;
    case 1: color = Color::red; break;
    case 2: color = Color::extinguish; break;
    case 3: color = Color::purple; break;
    default: color = Color::unknown; break;
  }

  // 解析装甲板名称
  // YOLOv5的num_id映射：0=sentry, 1=英雄(one), 2=工程(two), 3=步兵3(three),
  //                    4=步兵4(four), 5=步兵5(five), 6=outpost, 7=base, 8=base大
  switch (num_id) {
    case 0: name = ArmorName::sentry; break;
    case 1: name = ArmorName::one; break;
    case 2: name = ArmorName::two; break;
    case 3: name = ArmorName::three; break;
    case 4: name = ArmorName::four; break;
    case 5: name = ArmorName::five; break;
    case 6: name = ArmorName::outpost; break;
    case 7: name = ArmorName::base; break;
    case 8: name = ArmorName::base; break;  // 基地大装甲板
    default: name = ArmorName::not_armor; is_valid = false; break;
  }

  // 解析装甲板类型
  // 英雄(1)和基地大(8)是大装甲板，其他是小装甲板
  type = (num_id == 1 || num_id == 8) ? ArmorType::big : ArmorType::small;

  // 设置优先级
  rank = getRankFromName(name);
}


}  // namespace armor_auto_aim
