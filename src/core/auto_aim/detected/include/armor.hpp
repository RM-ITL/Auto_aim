#ifndef ARMOR_AUTO_AIM__ARMOR_HPP
#define ARMOR_AUTO_AIM__ARMOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <cstdint>

namespace armor_auto_aim
{

enum Color
{
  red,
  blue,
  extinguish,
  purple
};
const std::vector<std::string> COLORS = {"red", "blue", "extinguish", "purple"};

enum ArmorType
{
  big,
  small
};
const std::vector<std::string> ARMOR_TYPES = {"big", "small"};

enum ArmorName
{
  one = 1,
  two = 2,
  three = 3,
  four = 4,
  five = 5,
  sentry = 6,
  outpost = 7,
  base = 8,
  not_armor = -1
};
const std::vector<std::string> ARMOR_NAMES = {"one",    "two",     "three", "four",     "five",
                                              "sentry", "outpost", "base",  "not_armor"};

// 保留ArmorPriority枚举用于映射，但不再作为Armor结构体的成员
enum ArmorPriority
{
  first = 1,
  second = 2,
  third = 3,
  forth = 4,
  fifth = 5
};

using PriorityMap = std::unordered_map<armor_auto_aim::ArmorName, armor_auto_aim::ArmorPriority>;

// 优先级映射表：从装甲板名称到优先级的映射
// 数值越小，优先级越高
const PriorityMap rank_map = {
{armor_auto_aim::ArmorName::one, armor_auto_aim::ArmorPriority::second},      // 英雄1号 - 第二优先级
{armor_auto_aim::ArmorName::two, armor_auto_aim::ArmorPriority::forth},       // 工程2号 - 第四优先级
{armor_auto_aim::ArmorName::three, armor_auto_aim::ArmorPriority::first},     // 步兵3号 - 第一优先级（主要目标）
{armor_auto_aim::ArmorName::four, armor_auto_aim::ArmorPriority::first},      // 步兵4号 - 第一优先级（主要目标）
{armor_auto_aim::ArmorName::five, armor_auto_aim::ArmorPriority::third},      // 步兵5号 - 第三优先级
{armor_auto_aim::ArmorName::sentry, armor_auto_aim::ArmorPriority::third},    // 哨兵 - 第三优先级
{armor_auto_aim::ArmorName::outpost, armor_auto_aim::ArmorPriority::fifth},   // 前哨站 - 第五优先级
{armor_auto_aim::ArmorName::base, armor_auto_aim::ArmorPriority::fifth},      // 基地 - 第五优先级
{armor_auto_aim::ArmorName::not_armor, armor_auto_aim::ArmorPriority::fifth}};// 非装甲板 - 第五优先级


struct Visualization
{
  std::array<cv::Point2f, 4> corners{};
  ArmorName name{ArmorName::not_armor};
  ArmorType type{ArmorType::small};
};

struct Lightbar
{
  std::size_t id;
  Color color;
  cv::Point2f center, top, bottom, top2bottom;
  std::vector<cv::Point2f> points;
  double angle, angle_error, length, width, ratio;
  cv::RotatedRect rotated_rect;

  Lightbar(const cv::RotatedRect & rotated_rect, std::size_t id);
  Lightbar()
    : id(0),
      color(red),  // 或者其他合理的默认值
      center(0, 0),
      top(0, 0),
      bottom(0, 0),
      top2bottom(0, 0),
      angle(0.0),
      angle_error(0.0),
      length(0.0),
      width(0.0),
      ratio(0.0) {}

};

struct Armor
{
    int class_id;              // YOLO原始输出ID (0-37)
    int color_class_id;        // 颜色类别ID
    int classify_class_id;     // 分类器输出ID

    float confidence;

    Color color;
    ArmorName name;
    ArmorType type;
    int32_t rank;              // 使用int32_t存储优先级，通过rank_map映射获取
    Lightbar left, right;

    ArmorPriority priority;
    
    
    cv::Rect box;
    cv::Point2f center;
    cv::Point2f center_norm;
    std::vector<cv::Point2f> points;
    
    double ratio;              // 两灯条的中点连线与长灯条的长度之比
    double side_ratio;         // 长灯条与短灯条的长度之比
    double rectangular_error;  // 灯条和中点连线所成夹角与π/2的差值
    
    cv::Mat pattern;
    
    bool is_valid;
    bool duplicated;
    
    // 辅助函数：根据ArmorName获取对应的rank值
    static int32_t getRankFromName(ArmorName armor_name) {
        auto it = rank_map.find(armor_name);
        if (it != rank_map.end()) {
            return static_cast<int32_t>(it->second);
        }
        return static_cast<int32_t>(ArmorPriority::fifth);  // 默认返回最低优先级
    }
    
    // 默认构造函数
    Armor()
        : class_id(0),
          color_class_id(0),
          classify_class_id(0),
          confidence(0.0f),
          color(red),
          name(not_armor),
          type(small),
          rank(getRankFromName(not_armor)),  // 通过映射获取rank
          ratio(0.0f),
          rectangular_error(0.0f),
          is_valid(false),
          duplicated(false) {}
    
    // 用于传统的灯条检测方法
    Armor(const Lightbar& left_bar, const Lightbar& right_bar);


    // YOLO11构造函数 - 在cpp文件中实现
    Armor(int yolo_class_id, float confidence, const cv::Rect& box,
          const std::vector<cv::Point2f>& keypoints);
    
    std::string getNameString() const {
        if (name >= one && name <= base) {
            return ARMOR_NAMES[static_cast<int>(name) - 1];
        }
        return "not_armor";
    }
    
    
};

inline std::string armor_name_to_string(ArmorName name)
{
  switch (name) {
    case ArmorName::one:
      return "one";
    case ArmorName::two:
      return "two";
    case ArmorName::three:
      return "three";
    case ArmorName::four:
      return "four";
    case ArmorName::five:
      return "five";
    case ArmorName::sentry:
      return "sentry";
    case ArmorName::outpost:
      return "outpost";
    case ArmorName::base:
      return "base";
    case ArmorName::not_armor:
    default:
      return "not_armor";
  }
}

}  // namespace armor_auto_aim

#endif