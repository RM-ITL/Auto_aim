// sim_client.hpp
//
// rm_aim_sim 本机共享内存客户端（协议 rm_aim_sim/2）。
//
// 职责：替换实车的 camera::Camera(取图) 与 io::Gimbal(串口姿态/下发)，
// 通过 /dev/shm/rm_aim_sim_<channel>/ 的 JSON 邮箱 + 原始 RGB 帧与仿真交换数据。
// 不链接 Godot，仅用文件 IO + nlohmann-json + flock + Linux 文件 API。
//
// 单线程参考适配：本实例非线程安全，勿跨线程共享 state_/seq/request。
// 仿真约定「一个通道一个客户端」，本类独占 client.lock。
#ifndef SIM_BRIDGE__SIM_CLIENT_HPP
#define SIM_BRIDGE__SIM_CLIENT_HPP

#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

namespace sim
{
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

// 图像模式下的一帧样本：BGR 图 + 该帧同步元数据（含采样时刻 gimbal/sim_time/标定）。
struct Sample
{
  cv::Mat bgr;
  Json metadata;
  Clock::time_point received;
};

// 从 state/frame 的 gimbal 节点取 wxyz 四元数（R_gimbal2world）。失败返回单位四元数。
Eigen::Quaterniond quaternion_from(const Json & gimbal);

class SimClient
{
public:
  explicit SimClient(const std::string & channel);
  ~SimClient();

  SimClient(const SimClient &) = delete;
  SimClient & operator=(const SimClient &) = delete;

  // ---- 状态 ----
  // 非阻塞读取 state.json；收到「新发布」(publisher/publication 变化) 返回 true。
  bool poll_state();
  bool has_state() const { return !state_.empty(); }
  // 最近一次新状态的单调接收年龄 <= 200ms。
  bool fresh() const;
  const Json & state() const { return state_; }

  bool paused() const;
  bool enabled() const;  // ipc.enabled

  // ---- 云台 / 弹速反馈（取自最近状态；缺字段回退 0）----
  double gimbal_yaw() const { return num(state_["gimbal"], "yaw"); }
  double gimbal_pitch() const { return num(state_["gimbal"], "pitch"); }
  double gimbal_yaw_vel() const { return num(state_["gimbal"], "yaw_vel"); }
  double gimbal_pitch_vel() const { return num(state_["gimbal"], "pitch_vel"); }
  double bullet_speed() const { return num(state_["weapon"], "speed"); }
  Eigen::Quaterniond quaternion() const;  // state.gimbal.quaternion_wxyz

  std::string robot() const;  // appearance.robot
  std::string team() const;   // appearance.team

  // ---- 图像（图形模式）----
  // 无未完成请求且状态新鲜时，原子发布唯一 request_id。
  void request_frame();
  // 非阻塞检查匹配帧，校验上下文/时间/格式/大小，读入并转成独立 BGR；超 5s 抛异常。
  bool try_frame(Sample & out);

  // ---- 控制 ----
  // 绝对目标角(rad)，可选前馈。control 仅在 fresh/未暂停/已启用时真正置 true。
  void send(
    bool control, bool fire, double yaw, double pitch, double yaw_vel = 0.0,
    double pitch_vel = 0.0, double yaw_acc = 0.0, double pitch_acc = 0.0);

private:
  static double num(const Json & node, const char * key);
  Json context(const Json & packet) const;
  Json read_json(const char * filename) const;
  void write_json(const char * filename, const Json & value) const;

  int directory_ = -1;
  int lock_ = -1;
  Json state_ = Json::object();
  std::string request_;
  std::string token_;
  std::string sequence_session_;
  std::int64_t sequence_ = 0;
  std::uint64_t request_count_ = 0;
  Clock::time_point last_state_{};
  Clock::time_point requested_{};
};

}  // namespace sim

#endif  // SIM_BRIDGE__SIM_CLIENT_HPP
