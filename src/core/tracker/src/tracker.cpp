#include "tracker.hpp"
#include <yaml-cpp/yaml.h>
#include <tuple>
#include "math_tools.hpp"
#include <rclcpp/rclcpp.hpp>  // 添加ROS日志支持

namespace tracker
{
Tracker::Tracker(const std::string & config_path, solver::Solver & solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now())  // 修正：去掉末尾多余的逗号
{
  auto yaml = YAML::LoadFile(config_path);
  
  // 解析敌方颜色配置
  std::string enemy_color_str = yaml["Tracker"]["enemy_color"].as<std::string>();
  enemy_color_ = (enemy_color_str == "red") ? 
                 armor_auto_aim::Color::red : 
                 armor_auto_aim::Color::blue;
  
  // 加载跟踪参数
  min_detect_count_ = yaml["Tracker"]["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["Tracker"]["max_temp_lost_count"].as<int>();
  outpost_max_temp_lost_count_ = yaml["Tracker"]["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;
  
  RCLCPP_INFO(rclcpp::get_logger("Tracker"), 
              "跟踪器初始化完成 - 敌方颜色: %s, 最小检测次数: %d", 
              enemy_color_str.c_str(), min_detect_count_);
}

std::string Tracker::state() const { return state_; }

std::list<predict::Target> Tracker::track(
  std::list<Armors> & armors, 
  std::chrono::steady_clock::time_point t, 
  bool use_enemy_color)
{
  auto dt = utils::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 检测是否长时间无数据（可能相机离线）
  if (state_ != "lost" && dt > 0.1) {
    RCLCPP_WARN(rclcpp::get_logger("Tracker"), 
                "时间间隔过长 (%.3fs)，重置为lost状态", dt);
    state_ = "lost";
  }
  
  // // 根据配置过滤敌方装甲板
  // if (use_enemy_color) {
  //   // 将字符串颜色转换为枚举类型进行比较
  //   armors.remove_if([this](Armors & a) { 
  //     armor_auto_aim::Color msg_color;
  //     if (a.color == "red") {
  //       msg_color = armor_auto_aim::Color::red;
  //     } else if (a.color == "blue") {
  //       msg_color = armor_auto_aim::Color::blue;
  //     } else if (a.color == "extinguish") {
  //       msg_color = armor_auto_aim::Color::extinguish;
  //     } else {
  //       msg_color = armor_auto_aim::Color::purple;
  //     }
  //     return msg_color != enemy_color_; 
  //   });
  // }
  // 根据配置过滤敌方装甲板
  if (use_enemy_color) {
    // 直接比较两个枚举值，简单高效
    armors.remove_if([this](Armors & a) { 
      return a.color != enemy_color_;  // 保留颜色匹配的，移除不匹配的
    });
  }

  // 按距离图像中心排序，优先选择靠近中心的装甲板
  armors.sort([](Armors & a, Armors & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO: 从相机参数读取
    cv::Point2f a_center(a.center.x, a.center.y);
    cv::Point2f b_center(b.center.x, b.center.y);
    auto distance_a = cv::norm(a_center - img_center);
    auto distance_b = cv::norm(b_center - img_center);
    return distance_a < distance_b;
  });

  // 按优先级排序（数字越小优先级越高）
  armors.sort([](Armors & a, Armors & b) { 
    return a.rank < b.rank; 
  });

  bool found = false;
  
  // 根据当前状态执行不同的操作
  if (state_ == "lost") {
    found = set_target(armors, t);
  } else {
    found = update_target(armors, t);
  }

  // 更新状态机
  state_machine(found);

  if (state_ == "tracking" || state_ == "temp_lost") {
    if (target_.diverged()) {
      RCLCPP_WARN(rclcpp::get_logger("Tracker"), 
                  "目标状态发散（当前状态: %s），重置为lost", 
                  state_.c_str());
      state_ = "lost";
      detect_count_ = 0;      // 重置检测计数
      temp_lost_count_ = 0;    // 重置临时丢失计数
      return {};
    }
  }

  // 收敛效果检测：同样只在tracking状态下进行
  if (state_ == "tracking" && target_.convergened()) {
    auto & ekf = target_.ekf();
    int failures = std::accumulate(
      ekf.recent_nis_failures.begin(), 
      ekf.recent_nis_failures.end(), 0);
    
    if (failures >= (0.4 * ekf.window_size)) {
      RCLCPP_WARN(rclcpp::get_logger("Tracker"), 
                  "NIS失败率过高 (%d/%d)，重置为lost", 
                  failures, ekf.window_size);
      state_ = "lost";
      detect_count_ = 0;
      temp_lost_count_ = 0;
      return {};
    }
  }

  // 如果处于lost状态，返回空列表
  if (state_ == "lost") {
    return {};
  }

  // 返回当前跟踪的目标
  std::list<predict::Target> targets = {target_};
  return targets;
}

void Tracker::state_machine(bool found)
{
  pre_state_ = state_;  // 保存前一状态用于调试
  
  if (state_ == "lost") {
    if (found) {
      state_ = "detecting";
      detect_count_ = 1;
      RCLCPP_INFO(rclcpp::get_logger("Tracker"), "状态转换: lost -> detecting");
    }
  }
  else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) {
        state_ = "tracking";
        RCLCPP_INFO(rclcpp::get_logger("Tracker"), 
                    "状态转换: detecting -> tracking (检测次数: %d)", detect_count_);
      }
    } else {
      detect_count_ = 0;
      state_ = "lost";
      RCLCPP_INFO(rclcpp::get_logger("Tracker"), "状态转换: detecting -> lost");
    }
  }
  else if (state_ == "tracking") {
    if (!found) {
      temp_lost_count_ = 1;
      state_ = "temp_lost";
      RCLCPP_DEBUG(rclcpp::get_logger("Tracker"), "状态转换: tracking -> temp_lost");
    }
  }
  else if (state_ == "switching") {
    if (found) {
      state_ = "detecting";
      detect_count_ = 1;
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > 200) {
        state_ = "lost";
        RCLCPP_INFO(rclcpp::get_logger("Tracker"), "状态转换: switching -> lost");
      }
    }
  }
  else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
      temp_lost_count_ = 0;
      RCLCPP_DEBUG(rclcpp::get_logger("Tracker"), "状态转换: temp_lost -> tracking");
    } else {
      temp_lost_count_++;
      
      // 根据目标类型设置不同的丢失容忍度
      if (target_.name == armor_auto_aim::ArmorName::outpost) {
        max_temp_lost_count_ = outpost_max_temp_lost_count_;
      } else {
        max_temp_lost_count_ = normal_temp_lost_count_;
      }

      if (temp_lost_count_ > max_temp_lost_count_) {
        state_ = "lost";
        RCLCPP_INFO(rclcpp::get_logger("Tracker"), 
                    "状态转换: temp_lost -> lost (丢失计数: %d/%d)", 
                    temp_lost_count_, max_temp_lost_count_);
      }
    }
  }
}

bool Tracker::set_target(std::list<Armors> & armors, 
                         std::chrono::steady_clock::time_point t)
{
  if (armors.empty()) {
    return false;
  }

  // 获取优先级最高的装甲板
  auto & armor = armors.front();
  
  // 使用Solver进行PnP解算，得到装甲板的3D位姿
  double timestamp = std::chrono::duration<double>(t.time_since_epoch()).count();
  solver::Armor_pose armor_pose = solver_.processArmor(armor, timestamp);

  utils::logger()->debug(
    "【Target构造】接收装甲板数据:\n"
    "  装甲板ID: {}, 类型: {}\n"
    "  相机坐标: [{:.3f}, {:.3f}, {:.3f}]\n"
    "  云台坐标: [{:.3f}, {:.3f}, {:.3f}]\n"
    "  世界坐标: [{:.3f}, {:.3f}, {:.3f}]\n"
    "  世界球坐标: [yaw={:.3f}rad, pitch={:.3f}rad, distance={:.3f}m]\n"
    "  云台姿态: [yaw={:.3f}, pitch={:.3f}, roll={:.3f}]rad\n"
    "  世界姿态: [yaw={:.3f}, pitch={:.3f}, roll={:.3f}]rad",
    static_cast<int>(armor_pose.id), 
    static_cast<int>(armor_pose.type),
    armor_pose.camera_position[0], armor_pose.camera_position[1], armor_pose.camera_position[2],
    armor_pose.gimbal_position[0], armor_pose.gimbal_position[1], armor_pose.gimbal_position[2],
    armor_pose.world_position[0], armor_pose.world_position[1], armor_pose.world_position[2],
    armor_pose.world_spherical.yaw, armor_pose.world_spherical.pitch, armor_pose.world_spherical.distance,
    armor_pose.gimbal_orientation.yaw, armor_pose.gimbal_orientation.pitch, armor_pose.gimbal_orientation.roll,
    armor_pose.world_orientation.yaw, armor_pose.world_orientation.pitch, armor_pose.world_orientation.roll
  );

  
  // 检查解算是否成功
  if (!solver_.getLastPnPResult().success) {
    RCLCPP_WARN(rclcpp::get_logger("Tracker"), 
                "PnP解算失败，无法初始化目标");
    return false;
  }


  // 根据兵种类型选择不同的初始化参数
  // 平衡步兵特殊处理（大装甲板的3、4、5号）
  bool is_balance = (armor_pose.type == armor_auto_aim::ArmorType::big) &&
                   (armor_pose.id == armor_auto_aim::ArmorName::three || 
                    armor_pose.id == armor_auto_aim::ArmorName::four ||
                    armor_pose.id == armor_auto_aim::ArmorName::five);

  if (is_balance) {
    // 平衡步兵：2块装甲板，半径约0.2m
    Eigen::VectorXd P0_dig(11);
    P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
    target_ = predict::Target(armor_pose, t, 0.2, 2, P0_dig);
    RCLCPP_INFO(rclcpp::get_logger("Tracker"), 
                "初始化平衡步兵目标 (ID: %d)", static_cast<int>(armor_pose.id));
  }
  else if (armor_pose.id == armor_auto_aim::ArmorName::outpost) {
    // 前哨站：3块装甲板，半径约0.2765m，不旋转或慢速旋转
    Eigen::VectorXd P0_dig(11);
    P0_dig << 1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0;
    target_ = predict::Target(armor_pose, t, 0.2765, 3, P0_dig);
    RCLCPP_INFO(rclcpp::get_logger("Tracker"), "初始化前哨站目标");
  }
  else if (armor_pose.id == armor_auto_aim::ArmorName::base) {  
    // 基地：大型目标，3块装甲板，半径约0.3205m
    Eigen::VectorXd P0_dig(11);
    P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0;
    target_ = predict::Target(armor_pose, t, 0.3205, 3, P0_dig);
    RCLCPP_INFO(rclcpp::get_logger("Tracker"), "初始化基地目标");
  }
  else {
    // 标准步兵：4块装甲板，半径约0.2m
    Eigen::VectorXd P0_dig(11);
    P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
    target_ = predict::Target(armor_pose, t, 0.2, 4, P0_dig);
    RCLCPP_INFO(rclcpp::get_logger("Tracker"), 
                "初始化标准步兵目标 (ID: %d)", static_cast<int>(armor_pose.id));
  }

  return true;
}

bool Tracker::update_target(std::list<Armors> & armors, 
                           std::chrono::steady_clock::time_point t)
{
  // 先进行预测，推进卡尔曼滤波器状态
  target_.predict(t);

  // 查找匹配当前目标的装甲板
  int found_count = 0;
  Armors* best_armor = nullptr;
  double min_center_x = 1e10;  // 用于选择最左侧的装甲板（可选策略）
  
  for (auto & armor : armors) {
    // 检查装甲板编号和类型是否匹配
    if (armor.name != static_cast<int>(target_.name)) continue;
    
    // 类型匹配检查
    // armor_auto_aim::ArmorType msg_type = (armor.type == "big" || armor.type == "large") ? 
    //                                      armor_auto_aim::ArmorType::big : 
    //                                      armor_auto_aim::ArmorType::small;
    if (armor.type != target_.armor_type) continue;
    
    found_count++;
    
    // 选择最合适的装甲板（这里选择最靠近画面中心的）
    if (!best_armor || armor.center.x < min_center_x) {
      best_armor = &armor;
      min_center_x = armor.center.x;
    }
  }

  if (found_count == 0 || !best_armor) {
    RCLCPP_DEBUG(rclcpp::get_logger("Tracker"), 
                 "未找到匹配的装甲板 (目标ID: %d)", 
                 static_cast<int>(target_.name));
    return false;
  }

  // 使用Solver解算最佳装甲板的位姿
  double timestamp = std::chrono::duration<double>(t.time_since_epoch()).count();
  solver::Armor_pose armor_pose = solver_.processArmor(*best_armor, timestamp);
  
  // 检查解算是否成功
  if (!solver_.getLastPnPResult().success) {
    RCLCPP_WARN(rclcpp::get_logger("Tracker"), 
                "PnP解算失败，跳过本次更新");
    return false;
  }

  // 更新目标状态
  target_.update(armor_pose);
  
  RCLCPP_DEBUG(rclcpp::get_logger("Tracker"), 
               "目标更新成功 (ID: %d, 找到%d个匹配装甲板)", 
               static_cast<int>(target_.name), found_count);

  return true;
}

}  // namespace tracker