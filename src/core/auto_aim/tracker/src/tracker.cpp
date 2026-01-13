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
  detect_fail_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now())
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

  // 加载前哨站特化参数
  outpost_min_detect_count_ = yaml["Tracker"]["outpost_min_detect_count"].as<int>();
  outpost_detect_fail_tolerance_ = yaml["Tracker"]["outpost_detect_fail_tolerance"].as<int>();

  RCLCPP_INFO(rclcpp::get_logger("Tracker"),
              "跟踪器初始化完成 - 敌方颜色: %s",
              enemy_color_str.c_str());
}

std::string Tracker::state() const { return state_; }

std::list<TargetVariant> Tracker::track(
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

  // 根据配置过滤敌方装甲板
  if (use_enemy_color) {
    armors.remove_if([this](Armors & a) {
      return a.color != enemy_color_;
    });
  }

  // 按距离图像中心排序
  armors.sort([](Armors & a, Armors & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);
    cv::Point2f a_center(a.center.x, a.center.y);
    cv::Point2f b_center(b.center.x, b.center.y);
    auto distance_a = cv::norm(a_center - img_center);
    auto distance_b = cv::norm(b_center - img_center);
    return distance_a < distance_b;
  });

  // 按优先级排序
  armors.sort([](Armors & a, Armors & b) {
    return a.rank < b.rank;
  });

  bool found = false;

  if (state_ == "lost") {
    found = set_target(armors, t);
  } else {
    found = update_target(armors, t);
  }

  state_machine(found);

  // 发散检测
  if (state_ == "tracking" || state_ == "temp_lost") {
    bool diverged = std::visit([](auto & target) { return target.diverged(); }, target_);
    if (diverged) {
      RCLCPP_WARN(rclcpp::get_logger("Tracker"),
                  "目标状态发散（当前状态: %s），重置为lost",
                  state_.c_str());
      state_ = "lost";
      detect_count_ = 0;
      temp_lost_count_ = 0;
      return {};
    }
  }

  // 收敛效果检测
  if (state_ == "tracking") {
    bool converged = std::visit([](auto & target) { return target.convergened(); }, target_);
    if (converged) {
      const auto & ekf = std::visit([](auto & target) -> const motion_model::ExtendedKalmanFilter & {
        return target.ekf();
      }, target_);

      int failures = std::accumulate(
        ekf.recent_nis_failures.begin(),
        ekf.recent_nis_failures.end(), 0);

      // 【诊断日志】NIS检测情况
      if (is_tracking_outpost_) {
        utils::logger()->debug(
          "[NIS诊断] 前哨站NIS检测: failures={}/{}, 阈值={}",
          failures, ekf.window_size, static_cast<int>(0.4 * ekf.window_size));
      }

      if (failures >= (0.4 * ekf.window_size)) {
        utils::logger()->warn(
          "[Tracker] NIS失败率过高 ({}/{}), 重置为lost (前哨站:{})",
          failures, ekf.window_size, is_tracking_outpost_ ? "是" : "否");
        state_ = "lost";
        detect_count_ = 0;
        temp_lost_count_ = 0;
        return {};
      }
    }
  }

  if (state_ == "lost") {
    return {};
  }

  std::list<TargetVariant> targets = {target_};
  return targets;
}

void Tracker::state_machine(bool found)
{
  pre_state_ = state_;  // 保存前一状态用于调试
  
  if (state_ == "lost") {
    if (found) {
      state_ = "detecting";
      detect_count_ = 1;
      detect_fail_count_ = 0;  // 重置失败计数
      RCLCPP_INFO(rclcpp::get_logger("Tracker"), "状态转换: lost -> detecting");
    }
  }
  else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      detect_fail_count_ = 0;  // 成功时重置失败计数

      // 前哨站使用独立的最小检测次数
      int required_count = is_tracking_outpost_ ? outpost_min_detect_count_ : min_detect_count_;
      if (detect_count_ >= required_count) {
        state_ = "tracking";
        RCLCPP_INFO(rclcpp::get_logger("Tracker"),
                    "状态转换: detecting -> tracking (检测次数: %d, 前哨站: %s)",
                    detect_count_, is_tracking_outpost_ ? "是" : "否");
      }
    } else {
      // 前哨站特判：允许一定次数的检测失败
      if (is_tracking_outpost_) {
        detect_fail_count_++;
        if (detect_fail_count_ > outpost_detect_fail_tolerance_) {
          detect_count_ = 0;
          detect_fail_count_ = 0;
          state_ = "lost";
          RCLCPP_INFO(rclcpp::get_logger("Tracker"),
                      "状态转换: detecting -> lost (前哨站失败次数超限: %d)",
                      outpost_detect_fail_tolerance_);
        }
        // 否则保持detecting状态，继续等待
      } else {
        // 普通目标：一帧失败就重置
        detect_count_ = 0;
        state_ = "lost";
        RCLCPP_INFO(rclcpp::get_logger("Tracker"), "状态转换: detecting -> lost");
      }
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
      if (is_tracking_outpost_) {
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

  auto & armor = armors.front();

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

  if (!solver_.getLastPnPResult().success) {
    RCLCPP_WARN(rclcpp::get_logger("Tracker"),
                "PnP解算失败，无法初始化目标");
    return false;
  }

  // 平衡步兵特殊处理
  bool is_balance = (armor_pose.type == armor_auto_aim::ArmorType::big) &&
                   (armor_pose.id == armor_auto_aim::ArmorName::three ||
                    armor_pose.id == armor_auto_aim::ArmorName::four ||
                    armor_pose.id == armor_auto_aim::ArmorName::five);

  if (is_balance) {
    Eigen::VectorXd P0_dig(11);
    P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 2, 2;
    target_ = predict::Target(armor_pose, t, 0.2, 2, P0_dig);
    is_tracking_outpost_ = false;
    RCLCPP_INFO(rclcpp::get_logger("Tracker"),
                "初始化平衡步兵目标 (ID: %d)", static_cast<int>(armor_pose.id));
  }
  else if (armor_pose.id == armor_auto_aim::ArmorName::outpost) {
    // 前哨站：使用 OutpostTarget
    Eigen::VectorXd P0_dig(11);
    // cx vx cy vy cz vz θ ω r h1 h2
    P0_dig << 1, 8, 1, 8, 1, 4, 0.4, 50, 1e-4, 2, 2;
    target_ = predict::OutpostTarget(armor_pose, t, 0.2765, P0_dig);
    is_tracking_outpost_ = true;
    RCLCPP_INFO(rclcpp::get_logger("Tracker"), "初始化前哨站目标 (OutpostTarget)");
  }
  else if (armor_pose.id == armor_auto_aim::ArmorName::base) {
    Eigen::VectorXd P0_dig(11);
    P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0;
    target_ = predict::Target(armor_pose, t, 0.3205, 3, P0_dig);
    is_tracking_outpost_ = false;
    RCLCPP_INFO(rclcpp::get_logger("Tracker"), "初始化基地目标");
  }
  else {
    Eigen::VectorXd P0_dig(11);
    P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 2, 3;
    target_ = predict::Target(armor_pose, t, 0.24, 4, P0_dig);
    is_tracking_outpost_ = false;
    RCLCPP_INFO(rclcpp::get_logger("Tracker"),
                "初始化标准步兵目标 (ID: %d)", static_cast<int>(armor_pose.id));
  }

  return true;
}

bool Tracker::update_target(std::list<Armors> & armors,
                           std::chrono::steady_clock::time_point t)
{
  // 先进行预测
  std::visit([t](auto & target) { target.predict(t); }, target_);

  // 获取目标名称和类型
  auto target_name = std::visit([](auto & target) { return target.name; }, target_);
  auto target_type = std::visit([](auto & target) { return target.armor_type; }, target_);

  // 查找匹配当前目标的装甲板
  int found_count = 0;
  Armors* best_armor = nullptr;
  double min_center_x = 1e10;

  for (auto & armor : armors) {
    if (armor.name != static_cast<int>(target_name)) continue;
    if (armor.type != target_type) continue;

    found_count++;

    if (!best_armor || armor.center.x < min_center_x) {
      best_armor = &armor;
      min_center_x = armor.center.x;
    }
  }

  if (found_count == 0 || !best_armor) {
    return false;
  }

  double timestamp = std::chrono::duration<double>(t.time_since_epoch()).count();
  solver::Armor_pose armor_pose = solver_.processArmor(*best_armor, timestamp);

  if (!solver_.getLastPnPResult().success) {
    return false;
  }

  // 更新目标状态
  std::visit([&armor_pose](auto & target) { target.update(armor_pose); }, target_);

  return true;
}

}  // namespace tracker
