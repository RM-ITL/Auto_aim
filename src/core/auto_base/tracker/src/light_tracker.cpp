#include "light_tracker.hpp"

#include <algorithm>
#include <cmath>

#include "logger.hpp"
#include "math_tools.hpp"
#include "yaml.hpp"

namespace auto_base
{

LightTracker::LightTracker(const std::string & config_path)
: state_("lost"),
  detect_count_(0),
  temp_lost_count_(0),
  last_timestamp_(std::chrono::steady_clock::now())
{
  auto yaml = utils::load(config_path);

  // 读取 LightTracker 配置参数
  min_detect_count_ = yaml["LightTracker"]["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["LightTracker"]["max_temp_lost_count"].as<int>();

  utils::logger()->info(
    "[LightTracker] 初始化完成 - min_detect_count={}, max_temp_lost_count={}",
    min_detect_count_, max_temp_lost_count_);
}

std::string LightTracker::state() const { return state_; }

std::list<LightTarget*> LightTracker::track(
  const std::vector<OpenvinoInfer::GreenLight> & detections,
  std::chrono::steady_clock::time_point t)
{
  // 计算时间间隔
  auto dt = utils::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 检测时间间隔是否过长（相机可能离线）
  if (state_ != "lost" && dt > 0.1) {
    utils::logger()->warn(
      "[LightTracker] 时间间隔过长 ({:.3f}s)，重置为lost状态", dt);
    state_ = "lost";
    detect_count_ = 0;
    temp_lost_count_ = 0;
  }

  detections_cache_ = detections;

  // 根据状态分发处理
  bool found = false;

  if (state_ == "lost") {
    found = set_target(detections, t);
  } else {
    found = update_target(detections, t);
  }

  // 执行状态转移
  state_machine(found);

  // 发散检测
  if (state_ == "tracking" || state_ == "temp_lost") {
    if (target_ && target_->is_diverged()) {
      utils::logger()->warn(
        "[LightTracker] 目标状态发散（当前状态: {}），重置为lost", state_);
      state_ = "lost";
      detect_count_ = 0;
      temp_lost_count_ = 0;
      target_.reset();
      return {};
    }
  }

  // 收敛检测
  if (state_ == "tracking") {
    if (target_ && target_->is_converged()) {
      utils::logger()->debug(
        "[LightTracker] 目标已收敛 (更新次数: {})",
        target_->update_count());
    }
  }

  // 返回结果 - 返回指针而不是移动所有权
  if (state_ == "lost" || !target_) {
    return {};
  }

  std::list<LightTarget*> targets;
  targets.push_back(target_.get());  // 返回指针，不移动所有权
  return targets;
}

void LightTracker::state_machine(bool found)
{
  if (state_ == "lost") {
    if (found) {
      state_ = "detecting";
      detect_count_ = 1;
      temp_lost_count_ = 0;
      utils::logger()->info("[LightTracker] 状态转换: lost → detecting");
    }
  } else if (state_ == "detecting") {
    if (found) {
      detect_count_++;

      if (detect_count_ >= min_detect_count_) {
        state_ = "tracking";
        utils::logger()->info(
          "[LightTracker] 状态转换: detecting → tracking (检测次数: {})",
          detect_count_);
      }
    } else {
      // detecting 状态下一帧失败就重置
      detect_count_ = 0;
      state_ = "lost";
      utils::logger()->info("[LightTracker] 状态转换: detecting → lost");
    }
  } else if (state_ == "tracking") {
    if (!found) {
      temp_lost_count_ = 1;
      state_ = "temp_lost";
      utils::logger()->debug("[LightTracker] 状态转换: tracking → temp_lost");
    }
  } else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
      temp_lost_count_ = 0;
      utils::logger()->debug("[LightTracker] 状态转换: temp_lost → tracking");
    } else {
      temp_lost_count_++;

      if (temp_lost_count_ > max_temp_lost_count_) {
        state_ = "lost";
        detect_count_ = 0;
        utils::logger()->info(
          "[LightTracker] 状态转换: temp_lost → lost (丢失计数: {}/{})",
          temp_lost_count_, max_temp_lost_count_);
        target_.reset();
      }
    }
  }
}

bool LightTracker::set_target(
  const std::vector<OpenvinoInfer::GreenLight> & detections,
  std::chrono::steady_clock::time_point t)
{
  if (detections.empty()) {
    return false;
  }

  // 选择置信度最高的检测框作为初始目标
  const auto & best_detection = *std::max_element(
    detections.begin(), detections.end(),
    [](const OpenvinoInfer::GreenLight & a, const OpenvinoInfer::GreenLight & b) {
      return a.score < b.score;
    });

  // 定义初始协方差矩阵对角元素
  // [cx, cy, w, h, dx, dy, dw, dh]
  Eigen::VectorXd P0_dig(8);
  P0_dig << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5;            

  target_ = std::make_unique<LightTarget>(best_detection, t, P0_dig);

  utils::logger()->info(
    "[LightTracker] 创建新目标: center=({:.1f}, {:.1f}), score={:.3f}",
    best_detection.center.x, best_detection.center.y, best_detection.score);

  return true;
}

bool LightTracker::update_target(
  const std::vector<OpenvinoInfer::GreenLight> & detections,
  std::chrono::steady_clock::time_point t)
{
  if (!target_) {
    return false;
  }

  // 先进行预测
  target_->predict(t);

  // 如果检测框为空，返回 false（继续追踪但标记为未找到）
  if (detections.empty()) {
    return false;
  }

  // 找到最接近历史目标的检测框
  auto best_detection = find_best_detection(
    const_cast<std::vector<OpenvinoInfer::GreenLight> &>(detections));

  if (best_detection != nullptr) {
    target_->update(*best_detection);
    return true;
  }

  return false;
}

OpenvinoInfer::GreenLight * LightTracker::find_best_detection(
  std::vector<OpenvinoInfer::GreenLight> & detections)
{
  if (detections.empty()) {
    return nullptr;
  }

  // 选择置信度最高的检测框作为最佳匹配
  auto best = std::max_element(
    detections.begin(), detections.end(),
    [](const OpenvinoInfer::GreenLight & a, const OpenvinoInfer::GreenLight & b) {
      return a.score < b.score;
    });

  if (best != detections.end()) {
    utils::logger()->debug(
      "[LightTracker] 匹配到检测框: center=({:.1f}, {:.1f}), score={:.3f}",
      best->center.x, best->center.y, best->score);
    return &(*best);
  }

  return nullptr;
}

}  // namespace auto_base
