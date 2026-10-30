// sim_node.cpp
//
// rm_aim_sim 闭环接入节点：用 SimClient 替换实车相机/串口，复用 Auto_aim 的
// Solver/Tracker/Planner/Shooter 内核，不改动既有包。
//
// 两种模式（--mode）：
//   ideal  仿真理想角点(state.camera.armors)→合成 Armor，绕过检测，验证解算/跟踪/控制闭环。
//          可配合无界面仿真 ./run.sh --headless -- --remote。
//   image  取渲染帧→Auto_aim 真实 Detector→全链路。需图形仿真 ./run.sh -- --remote。
//
// 坐标/符号约定已对齐（见 src/config/sim.yaml 的内外参注释）。
// 默认绝不开火；通过 --fire 显式开启火控，前提是已完成不开火验收。
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <list>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>

#include "app_config/app_config.hpp"
#include "armor.hpp"
#include "detect_node.hpp"
#include "gimbal.hpp"
#include "logger.hpp"
#include "math_tools.hpp"
#include "planner.hpp"
#include "shooter.hpp"
#include "solver_node.hpp"
#include "tracker.hpp"

#include "sim_client.hpp"

namespace
{
using Armor = armor_auto_aim::Armor;
using Clock = std::chrono::steady_clock;
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

// 仿真车型 → Auto_aim 装甲名 / 类型。sim 的 sentry 逻辑编号 7，映射到 ArmorName::sentry。
std::pair<armor_auto_aim::ArmorName, armor_auto_aim::ArmorType> robot_to_armor(const std::string & r)
{
  using N = armor_auto_aim::ArmorName;
  using T = armor_auto_aim::ArmorType;
  if (r == "hero") return {N::one, T::big};
  if (r == "infantry_3") return {N::three, T::small};
  if (r == "infantry_4") return {N::four, T::small};
  if (r == "infantry_5") return {N::five, T::small};
  if (r == "sentry") return {N::sentry, T::small};
  return {N::three, T::small};
}

// 从 state.camera.armors 的理想角点(TL/TR/BR/BL 像素)合成 Armor，供 solver/tracker 使用。
std::list<Armor> ideal_armors(const sim::Json & state)
{
  std::list<Armor> armors;
  if (!state.contains("camera") || !state["camera"].contains("armors")) return armors;
  const auto [name, type] = robot_to_armor(state.value("appearance", sim::Json::object())
                                             .value("robot", std::string("infantry_3")));
  const bool red = state.value("appearance", sim::Json::object()).value("team", std::string()) == "red";
  const auto color = red ? armor_auto_aim::Color::red : armor_auto_aim::Color::blue;
  for (const auto & a : state["camera"]["armors"]) {
    if (!a.contains("corners")) continue;
    std::vector<cv::Point2f> pts;
    for (const auto & c : a["corners"]) {
      if (c.size() != 2) continue;
      pts.emplace_back(c[0].get<float>(), c[1].get<float>());
    }
    if (pts.size() != 4) continue;
    Armor armor;
    armor.color = color;
    armor.name = name;
    armor.type = type;
    armor.rank = Armor::getRankFromName(name);
    armor.points = pts;
    armor.center = (pts[0] + pts[1] + pts[2] + pts[3]) / 4.0F;
    armor.confidence = 1.0F;
    armor.is_valid = true;
    armors.push_back(std::move(armor));
  }
  return armors;
}
}  // namespace

int main(int argc, char ** argv)
{
  std::string channel = "default";
  std::string config_path = std::filesystem::current_path().string() + "/src/config/sim.yaml";
  std::string mode = "ideal";
  bool allow_fire = false;
  double duration = 0.0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      std::printf(
        "sim_node [--channel=default] [--config=PATH] [--mode=ideal|image] [--fire] [--duration=0]\n"
        "  先启动仿真: ideal 可 ./run.sh --headless -- --remote; image 需 ./run.sh -- --remote\n"
        "  角度单位 rad；默认绝不开火，--fire 才启用火控。\n");
      return 0;
    }
    if (arg == "--fire") { allow_fire = true; continue; }
    const auto eq = arg.find('=');
    if (eq == std::string::npos) { std::fprintf(stderr, "Use --opt=value: %s\n", arg.c_str()); return 1; }
    const auto key = arg.substr(0, eq), val = arg.substr(eq + 1);
    if (key == "--channel") channel = val;
    else if (key == "--config") config_path = val;
    else if (key == "--mode") mode = val;
    else if (key == "--duration") duration = std::stod(val);
    else { std::fprintf(stderr, "Unknown option: %s\n", arg.c_str()); return 1; }
  }
  if (mode != "ideal" && mode != "image") {
    std::fprintf(stderr, "--mode must be ideal or image\n");
    return 1;
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  try {
    const auto app_config = app_config::AppConfig::load(config_path);
    auto solver = std::make_unique<solver::Solver>(app_config.solver);
    auto tracker = std::make_unique<tracker::Tracker>(app_config.tracker, *solver);
    auto planner = std::make_unique<plan::Planner>(app_config.planner);
    auto shooter = std::make_unique<shooter::Shooter>(app_config.shooter);
    std::unique_ptr<armor_auto_aim::Detector> detector;
    if (mode == "image") detector = std::make_unique<armor_auto_aim::Detector>(app_config.detector);

    sim::SimClient client(channel);
    utils::logger()->info(
      "[Sim] 已连接通道 '{}' | mode={} | fire={}", channel, mode, allow_fire ? "on" : "off");

    const auto start = Clock::now();
    auto last_report = start;
    std::string last_state = "lost";
    // 每秒窗口统计：处理帧数 / 进入 tracker 前的装甲数、颜色分布、与仿真车型一致的编号/类型数，
    // 以及 tracker 内部 PnP 失败次数（据 solver 最近一次结果判定）。
    std::uint64_t frames_window = 0, armors_window = 0, red_window = 0, blue_window = 0;
    std::uint64_t named_window = 0, pnp_fail_window = 0;
    std::map<std::string, std::uint64_t> mismatch_window;  // "name/type" → 次数

    // 单次闭环：解算姿态 → 跟踪 → 规划 → 火控判定 → 下发绝对目标角。
    auto cycle = [&](std::list<Armor> & armors, const Eigen::Quaterniond & q,
                     Clock::time_point now, bool use_color) {
      const double t_sec = utils::delta_time(now, start);
      solver->updateIMU(q, t_sec);
      // tracker 会就地按 enemy_color 过滤，先统计输入，便于区分「检不出」「颜色不符」「编号/类型误判」。
      ++frames_window;
      const auto [want_name, want_type] = robot_to_armor(client.robot());
      for (const auto & a : armors) {
        ++armors_window;
        if (a.color == armor_auto_aim::Color::red) ++red_window;
        else if (a.color == armor_auto_aim::Color::blue) ++blue_window;
        if (a.name == want_name && a.type == want_type) {
          ++named_window;
        } else {
          ++mismatch_window[a.getNameString() + "/" + armor_auto_aim::ARMOR_TYPES[a.type]];
        }
      }
      const bool had_armor = !armors.empty();
      auto targets = tracker->track(armors, now, use_color);
      if (had_armor && !solver->getLastPnPResult().success) ++pnp_fail_window;
      std::optional<tracker::TargetVariant> target;
      if (!targets.empty()) target = targets.front();

      io::GimbalState gs{};
      gs.yaw = static_cast<float>(client.gimbal_yaw());
      gs.pitch = static_cast<float>(client.gimbal_pitch());
      gs.yaw_vel = static_cast<float>(client.gimbal_yaw_vel());
      gs.pitch_vel = static_cast<float>(client.gimbal_pitch_vel());
      gs.bullet_speed = static_cast<float>(client.bullet_speed());

      auto plan = planner->plan(target, gs.bullet_speed);
      bool fire = false;
      if (allow_fire && target.has_value())
        fire = plan.fire && shooter->checkfire(plan.yaw, plan.pitch, gs, target.value());

      const bool safe = plan.control && std::isfinite(plan.yaw) && std::isfinite(plan.pitch);
      if (safe)
        client.send(
          true, fire, plan.yaw, plan.pitch, plan.yaw_vel, plan.pitch_vel, plan.yaw_acc,
          plan.pitch_acc);
      else
        client.send(false, false, gs.yaw, gs.pitch);

      const auto st = tracker->state();
      if (st != last_state) {
        utils::logger()->info("[Sim] tracker: {} -> {}", last_state, st);
        last_state = st;
      }
      if (now - last_report > std::chrono::seconds(1)) {
        last_report = now;
        utils::logger()->info(
          "[Sim] frames={} armors={}(red={},blue={},named={}) pnp_fail={} state={} target={} "
          "cmd(yaw={:.3f},pitch={:.3f}) actual(yaw={:.3f},pitch={:.3f}) bullet={:.1f} fire={}",
          frames_window, armors_window, red_window, blue_window, named_window, pnp_fail_window, st,
          target.has_value(), safe ? plan.yaw : 0.F, safe ? plan.pitch : 0.F, gs.yaw, gs.pitch,
          gs.bullet_speed, fire);
        if (!mismatch_window.empty()) {
          std::string other;
          for (const auto & [label, count] : mismatch_window)
            other += (other.empty() ? "" : ", ") + label + "x" + std::to_string(count);
          utils::logger()->info("[Sim]   编号/类型与仿真车型不符: {}", other);
        }
        frames_window = armors_window = red_window = blue_window = named_window = 0;
        pnp_fail_window = 0;
        mismatch_window.clear();
      }
    };

    auto last_fresh = start;
    auto last_stale_warn = start;
    while (!g_stop.load() &&
           (duration == 0.0 || utils::delta_time(Clock::now(), start) < duration)) {
      const bool got = client.poll_state();
      const auto loop_now = Clock::now();
      if (got) {
        last_fresh = loop_now;
      } else if (
        loop_now - last_fresh > std::chrono::seconds(1) &&
        loop_now - last_stale_warn > std::chrono::seconds(1)) {
        // 状态断流：仿真未启动/已退出/通道不一致。SimClient::send 内部已因 !fresh() 停控。
        last_stale_warn = loop_now;
        utils::logger()->warn(
          "[Sim] {:.1f}s 未收到新状态：检查仿真是否在运行、--channel 是否一致",
          utils::delta_time(loop_now, last_fresh));
      }
      if (mode == "ideal") {
        if (got) {
          auto armors = ideal_armors(client.state());
          cycle(armors, client.quaternion(), Clock::now(), false);
        }
      } else {
        client.request_frame();
        sim::Sample s;
        if (client.try_frame(s)) {
          // 观测时刻回推到采样时刻：最新状态与该帧的 sim_time 差即图像在仿真时间域的年龄
          // （渲染+读回+发布）。仿真近实时，故可在单调钟上等量回推；检测耗时也不计入观测戳。
          const double frame_age = std::clamp(
            client.state().value("sim_time", 0.0) - s.metadata.value("sim_time", 0.0), 0.0, 0.2);
          const auto observed = s.received - std::chrono::duration_cast<Clock::duration>(
                                               std::chrono::duration<double>(frame_age));
          // Detector 期望真实 BGR（YOLOv5 预处理按 BGR、传统取色 [0]=B/[2]=R）。
          // 实车节点的 cvtColor(BGR2RGB) 是在抵消 HikCamera 的 Bayer 命名偏差（BayerXX2BGR
          // 实得 RGB）；仿真帧已是真实 BGR，直接送入，不能再交换，否则红蓝颠倒。
          const auto q = sim::quaternion_from(s.metadata.at("gimbal"));
          auto dets = detector->detect(s.bgr);
          std::list<Armor> armors(dets.begin(), dets.end());
          cycle(armors, q, observed, true);
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    client.send(false, false, 0.0, 0.0);
    utils::logger()->info("[Sim] 正常退出");
    return 0;
  } catch (const std::exception & e) {
    utils::logger()->error("[Sim] 异常终止: {}", e.what());
    return 1;
  }
}
