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


struct Armor
{
    int class_id;              // YOLO原始输出ID (0-37)
    int color_class_id;        // 颜色类别ID
    int classify_class_id;     // 分类器输出ID
    
    float detection_confidence;
    float classify_confidence;
    
    Color color;
    ArmorName name;
    ArmorType type;
    int32_t rank;              // 使用int32_t存储优先级，通过rank_map映射获取

    ArmorPriority priority;
    
    
    cv::Rect box;
    cv::Point2f center;
    cv::Point2f center_norm;
    std::vector<cv::Point2f> points;
    
    float ratio;
    float rectangular_error;
    
    cv::Mat roi_image;
    
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
        : class_id(-1),
          color_class_id(-1),
          classify_class_id(-1),
          detection_confidence(0.0f),
          classify_confidence(0.0f),
          color(red),
          name(not_armor),
          type(small),
          rank(getRankFromName(not_armor)),  // 通过映射获取rank
          ratio(0.0f),
          rectangular_error(0.0f),
          is_valid(false),
          duplicated(false) {}
    
    // 传统检测器构造函数
    // 用于先检测颜色，后分类数字的传统方法
    Armor(const cv::Rect& b, float det_conf, int color_id)
        : class_id(-1),  // 传统方法没有YOLO的class_id
          color_class_id(color_id),
          classify_class_id(-1),
          detection_confidence(det_conf),
          classify_confidence(0.0f),
          color(red),
          name(not_armor),
          type(small),
          rank(getRankFromName(not_armor)),  // 初始化为默认值，后续可通过setName更新
          box(b),
          ratio(0.0f),
          rectangular_error(0.0f),
          is_valid(true),
          duplicated(false) {
        // 根据color_class_id设置颜色枚举
        if (color_class_id == 0) {
            color = blue;
        } else if (color_class_id == 1) {
            color = red;
        } else if (color_class_id == 2) {
            color = extinguish;
        } else if (color_class_id == 3) {
            color = purple;
        }
        
        // 计算中心点
        center.x = box.x + box.width / 2.0f;
        center.y = box.y + box.height / 2.0f;
    }
    
    // YOLO11构造函数 - 在cpp文件中实现
    Armor(int yolo_class_id, float confidence, const cv::Rect& box, 
          const std::vector<cv::Point2f>& keypoints);
    
    // 设置装甲板名称并自动更新rank
    void setName(ArmorName armor_name) {
        name = armor_name;
        rank = getRankFromName(armor_name);
    }
    
    // 辅助方法
    float getCombinedConfidence() const { 
        return detection_confidence * classify_confidence; 
    }
    
    std::string getNameString() const {
        if (name >= one && name <= base) {
            return ARMOR_NAMES[static_cast<int>(name) - 1];
        }
        return "not_armor";
    }
    
    int getClassNumber() const {
        if (name >= one && name <= five) {
            return static_cast<int>(name);
        }
        return -1;
    }
    
    // 获取优先级值（数值越小，优先级越高）
    int32_t getPriority() const {
        return rank;
    }
    
    // 比较两个装甲板的优先级
    bool hasHigherPriorityThan(const Armor& other) const {
        return rank < other.rank;  // rank值越小，优先级越高
    }
};

}  // namespace armor_auto_aim

#endif