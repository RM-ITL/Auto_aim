#include "common/armor.hpp"
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

// YOLO11构造函数实现
// 通过YOLO检测结果直接获取装甲板的所有属性，包括颜色、名称、类型和优先级
Armor::Armor(int yolo_class_id, float confidence, const cv::Rect& box, 
             const std::vector<cv::Point2f>& keypoints)
    : class_id(yolo_class_id),
      color_class_id(-1),
      classify_class_id(-1),
      detection_confidence(confidence),
      classify_confidence(0.0f),
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
    if (yolo_class_id >= 0 && yolo_class_id < armor_properties.size()) {
        // 从属性表中获取对应的颜色、名称和类型
        auto [parsed_color, parsed_name, parsed_type] = armor_properties[yolo_class_id];
        
        this->color = parsed_color;
        this->name = parsed_name;
        this->type = parsed_type;
        
        // 关键修改：通过映射表自动设置rank值
        // rank值由装甲板名称决定，实现了动态优先级分配
        this->rank = getRankFromName(parsed_name);
        
        // 为了兼容性，生成color_class_id
        // 注意：这里的映射可能与传统检测器使用的不同
        if (parsed_color == blue) {
            this->color_class_id = 0;
        } else if (parsed_color == red) {
            this->color_class_id = 1;
        } else if (parsed_color == extinguish) {
            this->color_class_id = 2;
        } else if (parsed_color == purple) {
            this->color_class_id = 3;
        }
    } else {
        // 无效的class_id，保持默认的not_armor和对应的rank值
        this->is_valid = false;
        this->rank = getRankFromName(not_armor);
    }
}

}  // namespace armor_auto_aim