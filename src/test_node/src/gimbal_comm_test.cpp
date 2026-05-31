#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "logger.hpp"
#include "serial/serial.h"

namespace
{

enum class Protocol
{
  V1,
  V2
};

struct __attribute__((packed)) GimbalFeedbackV1
{
  uint8_t head[2] = {'G', 'V'};
  uint8_t mode;
  float q[4];
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float bullet_speed;
  uint16_t bullet_count;
  uint16_t feedback_seq;
  uint8_t tail = 'G';
};

struct __attribute__((packed)) GimbalCommandWithSeq
{
  uint8_t head[2] = {'V', 'G'};
  uint8_t mode = 0;
  float yaw = 0.0f;
  float yaw_vel = 0.0f;
  float yaw_acc = 0.0f;
  float pitch = 0.0f;
  float pitch_vel = 0.0f;
  float pitch_acc = 0.0f;
  uint16_t command_seq = 0;
  uint8_t tail = 'V';
};

struct __attribute__((packed)) GimbalFeedbackV2
{
  uint8_t head[2] = {'G', 'V'};
  uint8_t mode;
  float q[4];
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float bullet_speed;
  uint16_t bullet_count;
  uint16_t feedback_seq;
  uint16_t echo_command_seq;
  uint32_t command_rx_count;
  uint32_t command_lost_count;
  uint16_t command_max_gap;
  uint8_t tail = 'G';
};

static_assert(sizeof(GimbalFeedbackV1) == 44);
static_assert(offsetof(GimbalFeedbackV1, feedback_seq) == 41);
static_assert(offsetof(GimbalFeedbackV1, tail) == 43);
static_assert(sizeof(GimbalCommandWithSeq) == 30);
static_assert(offsetof(GimbalCommandWithSeq, command_seq) == 27);
static_assert(offsetof(GimbalCommandWithSeq, tail) == 29);
static_assert(sizeof(GimbalFeedbackV2) == 56);
static_assert(offsetof(GimbalFeedbackV2, feedback_seq) == 41);
static_assert(offsetof(GimbalFeedbackV2, echo_command_seq) == 43);
static_assert(offsetof(GimbalFeedbackV2, command_rx_count) == 45);
static_assert(offsetof(GimbalFeedbackV2, command_lost_count) == 49);
static_assert(offsetof(GimbalFeedbackV2, command_max_gap) == 53);
static_assert(offsetof(GimbalFeedbackV2, tail) == 55);

struct Options
{
  std::string port = "/dev/gimbal";
  uint32_t baud = 115200;
  double duration_sec = 30.0;
  double log_period_sec = 1.0;
  Protocol protocol = Protocol::V1;
  double tx_rate_hz = 100.0;
};

struct FeedbackStats
{
  bool seq_inited = false;
  uint16_t last_seq = 0;

  uint64_t rx_unique_total = 0;
  uint64_t lost_total = 0;
  uint64_t duplicate_total = 0;
  uint64_t reset_total = 0;
  uint64_t parse_error_total = 0;

  uint64_t rx_unique_window = 0;
  uint64_t lost_window = 0;
  uint64_t parse_error_window = 0;
  uint16_t max_gap_window = 0;
  double max_dt_ms_window = 0.0;

  std::chrono::steady_clock::time_point last_rx_time;
  std::chrono::steady_clock::time_point window_start_time;
};

struct CommandStats
{
  uint16_t next_seq = 0;
  uint64_t tx_total = 0;
  uint64_t tx_window = 0;
  uint64_t write_fail_total = 0;
  std::array<std::chrono::steady_clock::time_point, 65536> send_time{};
  std::array<bool, 65536> send_time_valid{};

  bool lower_counter_inited = false;
  bool lower_counter_reset_detected = false;
  uint32_t lower_rx_baseline = 0;
  uint32_t lower_lost_baseline = 0;
  uint32_t last_lower_rx_count = 0;
  uint32_t last_lower_lost_count = 0;
  uint32_t lower_rx_window = 0;
  uint32_t lower_lost_window = 0;
  uint32_t lower_rx_total = 0;
  uint32_t lower_lost_total = 0;
  uint16_t lower_max_gap_total = 0;

  uint64_t rtt_count_total = 0;
  double rtt_sum_total_ms = 0.0;
  double rtt_min_total_ms = std::numeric_limits<double>::infinity();
  double rtt_max_total_ms = 0.0;
  std::vector<double> rtt_window_ms;
};

bool parse_arg_value(int argc, char ** argv, int & i, std::string & out)
{
  if (i + 1 >= argc) {
    return false;
  }
  out = argv[++i];
  return true;
}

bool parse_protocol(const std::string & value, Protocol & protocol)
{
  if (value == "v1") {
    protocol = Protocol::V1;
    return true;
  }
  if (value == "v2") {
    protocol = Protocol::V2;
    return true;
  }
  return false;
}

const char * protocol_name(Protocol protocol)
{
  return protocol == Protocol::V1 ? "v1" : "v2";
}

bool parse_options(int argc, char ** argv, Options & options)
{
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      return false;
    }

    std::string value;
    if (arg == "--port") {
      if (!parse_arg_value(argc, argv, i, value)) return false;
      options.port = value;
    } else if (arg == "--baud") {
      if (!parse_arg_value(argc, argv, i, value)) return false;
      options.baud = static_cast<uint32_t>(std::stoul(value));
    } else if (arg == "--duration") {
      if (!parse_arg_value(argc, argv, i, value)) return false;
      options.duration_sec = std::stod(value);
    } else if (arg == "--log-period") {
      if (!parse_arg_value(argc, argv, i, value)) return false;
      options.log_period_sec = std::stod(value);
    } else if (arg == "--protocol") {
      if (!parse_arg_value(argc, argv, i, value)) return false;
      if (!parse_protocol(value, options.protocol)) {
        utils::logger()->error("Invalid --protocol '{}', expected v1 or v2", value);
        return false;
      }
    } else if (arg == "--tx-rate") {
      if (!parse_arg_value(argc, argv, i, value)) return false;
      options.tx_rate_hz = std::stod(value);
    } else {
      utils::logger()->error("Unknown argument: {}", arg);
      return false;
    }
  }

  if (options.duration_sec <= 0.0 || options.log_period_sec <= 0.0) {
    return false;
  }
  if (options.protocol == Protocol::V2 && options.tx_rate_hz <= 0.0) {
    utils::logger()->error("--tx-rate must be > 0 in protocol v2");
    return false;
  }
  return true;
}

void print_usage()
{
  utils::logger()->info(
    "Usage: ros2 run test_pipeline gimbal_comm_test "
    "[--port /dev/gimbal] [--baud 115200] [--duration 30] [--log-period 1] "
    "[--protocol v1|v2] [--tx-rate 100]");
}

bool read_exact(serial::Serial & port, uint8_t * buffer, size_t size)
{
  size_t offset = 0;
  while (offset < size) {
    size_t n = 0;
    try {
      n = port.read(buffer + offset, size - offset);
    } catch (const std::exception & e) {
      utils::logger()->error("[GimbalCommTest] Serial read failed: {}", e.what());
      return false;
    }
    if (n == 0) {
      return false;
    }
    offset += n;
  }
  return true;
}

template<typename FrameT>
bool read_frame(serial::Serial & port, FrameT & frame, FeedbackStats & stats)
{
  bool have_prev = false;
  bool skipped_bytes = false;
  uint8_t prev = 0;
  uint8_t byte = 0;

  while (true) {
    if (!read_exact(port, &byte, 1)) {
      return false;
    }

    if (have_prev && prev == 'G' && byte == 'V') {
      frame.head[0] = 'G';
      frame.head[1] = 'V';
      if (skipped_bytes) {
        stats.parse_error_total++;
        stats.parse_error_window++;
      }
      break;
    }

    if (!have_prev) {
      skipped_bytes = byte != 'G';
    } else if (prev != 'G' || byte != 'G') {
      skipped_bytes = true;
    }

    prev = byte;
    have_prev = true;
  }

  auto * body = reinterpret_cast<uint8_t *>(&frame) + sizeof(frame.head);
  const size_t body_size = sizeof(frame) - sizeof(frame.head);
  if (!read_exact(port, body, body_size)) {
    return false;
  }

  if (frame.tail != 'G') {
    stats.parse_error_total++;
    stats.parse_error_window++;
    return false;
  }

  return true;
}

void update_feedback_stats(uint16_t seq, FeedbackStats & stats)
{
  const auto now = std::chrono::steady_clock::now();

  if (!stats.seq_inited) {
    stats.seq_inited = true;
    stats.last_seq = seq;
    stats.rx_unique_total++;
    stats.rx_unique_window++;
    stats.last_rx_time = now;
    stats.window_start_time = now;
    return;
  }

  const double dt_ms =
    std::chrono::duration<double, std::milli>(now - stats.last_rx_time).count();
  if (dt_ms > stats.max_dt_ms_window) {
    stats.max_dt_ms_window = dt_ms;
  }
  stats.last_rx_time = now;

  const uint16_t diff = static_cast<uint16_t>(seq - stats.last_seq);

  if (diff == 0) {
    stats.duplicate_total++;
    return;
  }

  if (diff > 30000) {
    stats.reset_total++;
    stats.last_seq = seq;
    stats.rx_unique_total++;
    stats.rx_unique_window++;
    return;
  }

  if (diff > 1) {
    const uint16_t lost = static_cast<uint16_t>(diff - 1);
    stats.lost_total += lost;
    stats.lost_window += lost;
    if (diff > stats.max_gap_window) {
      stats.max_gap_window = diff;
    }
  } else if (stats.max_gap_window == 0) {
    stats.max_gap_window = 1;
  }

  stats.last_seq = seq;
  stats.rx_unique_total++;
  stats.rx_unique_window++;
}

bool write_command(serial::Serial & port, CommandStats & stats)
{
  GimbalCommandWithSeq command{};
  command.command_seq = stats.next_seq;

  try {
    const size_t written = port.write(reinterpret_cast<uint8_t *>(&command), sizeof(command));
    if (written != sizeof(command)) {
      utils::logger()->warn(
        "[GimbalCommTest] Short command write: {} / {} bytes", written, sizeof(command));
      stats.write_fail_total++;
      return false;
    }
  } catch (const std::exception & e) {
    utils::logger()->error("[GimbalCommTest] Serial write failed: {}", e.what());
    stats.write_fail_total++;
    return false;
  }

  stats.send_time[command.command_seq] = std::chrono::steady_clock::now();
  stats.send_time_valid[command.command_seq] = true;
  stats.next_seq = static_cast<uint16_t>(stats.next_seq + 1);
  stats.tx_total++;
  stats.tx_window++;
  return true;
}

void update_command_feedback(const GimbalFeedbackV2 & frame, CommandStats & stats)
{
  const auto now = std::chrono::steady_clock::now();
  const uint16_t echo_seq = frame.echo_command_seq;

  if (stats.send_time_valid[echo_seq]) {
    const double rtt_ms =
      std::chrono::duration<double, std::milli>(now - stats.send_time[echo_seq]).count();
    if (rtt_ms >= 0.0 && rtt_ms < 10000.0) {
      stats.rtt_count_total++;
      stats.rtt_sum_total_ms += rtt_ms;
      stats.rtt_min_total_ms = std::min(stats.rtt_min_total_ms, rtt_ms);
      stats.rtt_max_total_ms = std::max(stats.rtt_max_total_ms, rtt_ms);
      stats.rtt_window_ms.push_back(rtt_ms);
    }
    stats.send_time_valid[echo_seq] = false;
  }

  if (!stats.lower_counter_inited) {
    stats.lower_counter_inited = true;
    stats.lower_rx_baseline = frame.command_rx_count;
    stats.lower_lost_baseline = frame.command_lost_count;
    stats.last_lower_rx_count = frame.command_rx_count;
    stats.last_lower_lost_count = frame.command_lost_count;
    stats.lower_rx_total = 0;
    stats.lower_lost_total = 0;
    stats.lower_max_gap_total = frame.command_max_gap;
    return;
  }

  if (
    frame.command_rx_count < stats.last_lower_rx_count ||
    frame.command_lost_count < stats.last_lower_lost_count ||
    frame.command_rx_count < stats.lower_rx_baseline ||
    frame.command_lost_count < stats.lower_lost_baseline) {
    stats.lower_counter_reset_detected = true;
    stats.last_lower_rx_count = frame.command_rx_count;
    stats.last_lower_lost_count = frame.command_lost_count;
    return;
  }

  stats.lower_rx_window += frame.command_rx_count - stats.last_lower_rx_count;
  stats.lower_lost_window += frame.command_lost_count - stats.last_lower_lost_count;

  stats.last_lower_rx_count = frame.command_rx_count;
  stats.last_lower_lost_count = frame.command_lost_count;
  stats.lower_rx_total = frame.command_rx_count - stats.lower_rx_baseline;
  stats.lower_lost_total = frame.command_lost_count - stats.lower_lost_baseline;
  stats.lower_max_gap_total = frame.command_max_gap;
}

double percentile(std::vector<double> values, double ratio)
{
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto index = static_cast<size_t>(ratio * static_cast<double>(values.size() - 1));
  return values[index];
}

void log_v1_window(FeedbackStats & feedback_stats, double elapsed_sec)
{
  const uint64_t expected_window = feedback_stats.rx_unique_window + feedback_stats.lost_window;
  const uint64_t expected_total = feedback_stats.rx_unique_total + feedback_stats.lost_total;

  const double feedback_rx_hz = elapsed_sec > 0.0 ?
    static_cast<double>(feedback_stats.rx_unique_window) / elapsed_sec : 0.0;
  const double loss_window = expected_window > 0 ?
    static_cast<double>(feedback_stats.lost_window) * 100.0 /
    static_cast<double>(expected_window) : 0.0;
  const double loss_total = expected_total > 0 ?
    static_cast<double>(feedback_stats.lost_total) * 100.0 /
    static_cast<double>(expected_total) : 0.0;

  utils::logger()->info(
    "[GimbalCommTest] protocol=v1 feedback_rx_hz={:.1f}, feedback_lost_window={}, "
    "feedback_expected_window={}, feedback_loss_window={:.3f}%, "
    "feedback_loss_total={:.3f}%, max_gap={}, max_dt={:.2f}ms, "
    "parse_error_window={}, parse_error_total={}, duplicate_total={}, reset_total={}",
    feedback_rx_hz,
    feedback_stats.lost_window,
    expected_window,
    loss_window,
    loss_total,
    feedback_stats.max_gap_window,
    feedback_stats.max_dt_ms_window,
    feedback_stats.parse_error_window,
    feedback_stats.parse_error_total,
    feedback_stats.duplicate_total,
    feedback_stats.reset_total);
}

void log_v2_window(FeedbackStats & feedback_stats, CommandStats & command_stats, double elapsed_sec)
{
  const uint64_t feedback_expected_window =
    feedback_stats.rx_unique_window + feedback_stats.lost_window;
  const uint64_t feedback_expected_total = feedback_stats.rx_unique_total + feedback_stats.lost_total;
  const uint64_t cmd_expected_window =
    static_cast<uint64_t>(command_stats.lower_rx_window) + command_stats.lower_lost_window;
  const uint64_t cmd_expected_total =
    static_cast<uint64_t>(command_stats.lower_rx_total) + command_stats.lower_lost_total;

  const double tx_hz = elapsed_sec > 0.0 ?
    static_cast<double>(command_stats.tx_window) / elapsed_sec : 0.0;
  const double cmd_rx_hz = elapsed_sec > 0.0 ?
    static_cast<double>(command_stats.lower_rx_window) / elapsed_sec : 0.0;
  const double feedback_rx_hz = elapsed_sec > 0.0 ?
    static_cast<double>(feedback_stats.rx_unique_window) / elapsed_sec : 0.0;
  const double cmd_loss_window = cmd_expected_window > 0 ?
    static_cast<double>(command_stats.lower_lost_window) * 100.0 /
    static_cast<double>(cmd_expected_window) : 0.0;
  const double cmd_loss_total = cmd_expected_total > 0 ?
    static_cast<double>(command_stats.lower_lost_total) * 100.0 /
    static_cast<double>(cmd_expected_total) : 0.0;
  const double feedback_loss_total = feedback_expected_total > 0 ?
    static_cast<double>(feedback_stats.lost_total) * 100.0 /
    static_cast<double>(feedback_expected_total) : 0.0;
  const double feedback_loss_window = feedback_expected_window > 0 ?
    static_cast<double>(feedback_stats.lost_window) * 100.0 /
    static_cast<double>(feedback_expected_window) : 0.0;
  const double rtt_avg_window = command_stats.rtt_window_ms.empty() ? 0.0 :
    std::accumulate(
      command_stats.rtt_window_ms.begin(), command_stats.rtt_window_ms.end(), 0.0) /
    static_cast<double>(command_stats.rtt_window_ms.size());
  const double rtt_min_window = command_stats.rtt_window_ms.empty() ? 0.0 :
    *std::min_element(command_stats.rtt_window_ms.begin(), command_stats.rtt_window_ms.end());
  const double rtt_max_window = command_stats.rtt_window_ms.empty() ? 0.0 :
    *std::max_element(command_stats.rtt_window_ms.begin(), command_stats.rtt_window_ms.end());
  const double rtt_p95_window = percentile(command_stats.rtt_window_ms, 0.95);
  const double rtt_avg_total = command_stats.rtt_count_total > 0 ?
    command_stats.rtt_sum_total_ms / static_cast<double>(command_stats.rtt_count_total) : 0.0;

  utils::logger()->info(
    "[GimbalCommTest] protocol=v2 tx_hz={:.1f}, cmd_rx_hz={:.1f}, "
    "cmd_loss_window={:.3f}%, cmd_loss_total={:.3f}%, cmd_max_gap={}, "
    "feedback_rx_hz={:.1f}, feedback_loss_window={:.3f}%, "
    "feedback_loss_total={:.3f}%, rtt_avg={:.2f}ms, rtt_min={:.2f}ms, "
    "rtt_max={:.2f}ms, rtt_p95={:.2f}ms, rtt_avg_total={:.2f}ms, "
    "write_fail={}, lower_counter_reset={}, parse_error_window={}, parse_error_total={}",
    tx_hz,
    cmd_rx_hz,
    cmd_loss_window,
    cmd_loss_total,
    command_stats.lower_max_gap_total,
    feedback_rx_hz,
    feedback_loss_window,
    feedback_loss_total,
    rtt_avg_window,
    rtt_min_window,
    rtt_max_window,
    rtt_p95_window,
    rtt_avg_total,
    command_stats.write_fail_total,
    command_stats.lower_counter_reset_detected,
    feedback_stats.parse_error_window,
    feedback_stats.parse_error_total);
}

void reset_window(FeedbackStats & feedback_stats, CommandStats * command_stats)
{
  feedback_stats.window_start_time = std::chrono::steady_clock::now();
  feedback_stats.rx_unique_window = 0;
  feedback_stats.lost_window = 0;
  feedback_stats.parse_error_window = 0;
  feedback_stats.max_gap_window = 0;
  feedback_stats.max_dt_ms_window = 0.0;

  if (command_stats) {
    command_stats->tx_window = 0;
    command_stats->lower_rx_window = 0;
    command_stats->lower_lost_window = 0;
    command_stats->rtt_window_ms.clear();
  }
}

void maybe_log_window(
  FeedbackStats & feedback_stats, CommandStats * command_stats, Protocol protocol,
  double log_period_sec)
{
  if (!feedback_stats.seq_inited) {
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  const double elapsed_sec =
    std::chrono::duration<double>(now - feedback_stats.window_start_time).count();

  if (elapsed_sec < log_period_sec) {
    return;
  }

  if (protocol == Protocol::V1) {
    log_v1_window(feedback_stats, elapsed_sec);
  } else if (command_stats) {
    log_v2_window(feedback_stats, *command_stats, elapsed_sec);
  }

  reset_window(feedback_stats, command_stats);
}

bool open_serial(serial::Serial & port, const Options & options)
{
  try {
    port.setPort(options.port);
    auto timeout = serial::Timeout::simpleTimeout(10);
    port.setTimeout(timeout);
    port.open();
    port.setBaudrate(options.baud);
  } catch (const std::exception & e) {
    utils::logger()->error("[GimbalCommTest] Failed to open serial: {}", e.what());
    return false;
  }

  if (!port.isOpen()) {
    utils::logger()->error("[GimbalCommTest] Serial is not open: {}", options.port);
    return false;
  }

  return true;
}

template<typename FrameT>
bool run_rx_loop(serial::Serial & port, const Options & options)
{
  FeedbackStats feedback_stats;
  FrameT frame{};
  const auto start_time = std::chrono::steady_clock::now();

  while (true) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
    if (elapsed_sec >= options.duration_sec) {
      break;
    }

    if (read_frame(port, frame, feedback_stats)) {
      update_feedback_stats(frame.feedback_seq, feedback_stats);
    }

    maybe_log_window(feedback_stats, nullptr, Protocol::V1, options.log_period_sec);
  }

  if (!feedback_stats.seq_inited || feedback_stats.rx_unique_total == 0) {
    utils::logger()->error(
      "[GimbalCommTest] No valid v1 feedback frame decoded. Test invalid: check port, baud, "
      "exclusive serial ownership, and lower feedback_seq protocol.");
    return false;
  }

  const uint64_t expected_total = feedback_stats.rx_unique_total + feedback_stats.lost_total;
  const double loss_total = expected_total > 0 ?
    static_cast<double>(feedback_stats.lost_total) * 100.0 / static_cast<double>(expected_total) : 0.0;

  utils::logger()->info(
    "[GimbalCommTest] FINAL protocol=v1 rx_unique={}, lost={}, expected={}, "
    "feedback_loss={:.3f}%, parse_error={}, duplicate={}, reset={}",
    feedback_stats.rx_unique_total,
    feedback_stats.lost_total,
    expected_total,
    loss_total,
    feedback_stats.parse_error_total,
    feedback_stats.duplicate_total,
    feedback_stats.reset_total);
  return true;
}

bool run_v2_loop(serial::Serial & port, const Options & options)
{
  FeedbackStats feedback_stats;
  CommandStats command_stats;
  GimbalFeedbackV2 frame{};
  auto next_send_time = std::chrono::steady_clock::now();
  const auto start_time = next_send_time;
  command_stats.next_seq = static_cast<uint16_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
      start_time.time_since_epoch()).count() & 0xFFFF);
  if (command_stats.next_seq == 0) {
    command_stats.next_seq = 1;
  }
  const auto tx_period = std::chrono::duration<double>(1.0 / options.tx_rate_hz);

  while (true) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
    if (elapsed_sec >= options.duration_sec) {
      break;
    }

    if (now >= next_send_time) {
      write_command(port, command_stats);
      next_send_time = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(tx_period);
    }

    if (read_frame(port, frame, feedback_stats)) {
      update_feedback_stats(frame.feedback_seq, feedback_stats);
      update_command_feedback(frame, command_stats);
    }

    maybe_log_window(feedback_stats, &command_stats, Protocol::V2, options.log_period_sec);
  }

  if (!feedback_stats.seq_inited || feedback_stats.rx_unique_total == 0) {
    utils::logger()->error(
      "[GimbalCommTest] No valid v2 feedback frame decoded. Test invalid: check port, baud, "
      "exclusive serial ownership, and lower v2 protocol size/layout.");
    return false;
  }

  const uint64_t feedback_expected_total = feedback_stats.rx_unique_total + feedback_stats.lost_total;
  const uint64_t cmd_expected_total =
    static_cast<uint64_t>(command_stats.lower_rx_total) + command_stats.lower_lost_total;
  const double feedback_loss_total = feedback_expected_total > 0 ?
    static_cast<double>(feedback_stats.lost_total) * 100.0 /
    static_cast<double>(feedback_expected_total) : 0.0;
  const double cmd_loss_total = cmd_expected_total > 0 ?
    static_cast<double>(command_stats.lower_lost_total) * 100.0 /
    static_cast<double>(cmd_expected_total) : 0.0;
  const double rtt_avg_total = command_stats.rtt_count_total > 0 ?
    command_stats.rtt_sum_total_ms / static_cast<double>(command_stats.rtt_count_total) : 0.0;
  const double rtt_min_total = command_stats.rtt_count_total > 0 ? command_stats.rtt_min_total_ms : 0.0;

  utils::logger()->info(
    "[GimbalCommTest] FINAL protocol=v2 tx_total={}, lower_cmd_rx={}, lower_cmd_lost={}, "
    "cmd_loss={:.3f}%, feedback_rx_unique={}, feedback_lost={}, feedback_loss={:.3f}%, "
    "rtt_count={}, rtt_avg={:.2f}ms, rtt_min={:.2f}ms, rtt_max={:.2f}ms, "
    "write_fail={}, lower_counter_reset={}, parse_error={}",
    command_stats.tx_total,
    command_stats.lower_rx_total,
    command_stats.lower_lost_total,
    cmd_loss_total,
    feedback_stats.rx_unique_total,
    feedback_stats.lost_total,
    feedback_loss_total,
    command_stats.rtt_count_total,
    rtt_avg_total,
    rtt_min_total,
    command_stats.rtt_max_total_ms,
    command_stats.write_fail_total,
    command_stats.lower_counter_reset_detected,
    feedback_stats.parse_error_total);

  if (command_stats.write_fail_total > 0) {
    utils::logger()->error("[GimbalCommTest] Command write failures occurred; v2 test invalid.");
    return false;
  }
  if (command_stats.lower_counter_reset_detected) {
    utils::logger()->error("[GimbalCommTest] Lower command counters reset/wrapped; v2 test invalid.");
    return false;
  }
  if (command_stats.tx_total == 0 || command_stats.lower_rx_total == 0) {
    utils::logger()->error(
      "[GimbalCommTest] No command reception confirmed by lower counters; v2 test invalid.");
    return false;
  }
  if (command_stats.rtt_count_total == 0) {
    utils::logger()->error(
      "[GimbalCommTest] No echo_command_seq matched this test session; RTT invalid.");
    return false;
  }

  return true;
}

}  // namespace

int main(int argc, char ** argv)
{
  Options options;
  try {
    if (!parse_options(argc, argv, options)) {
      print_usage();
      return 1;
    }
  } catch (const std::exception & e) {
    utils::logger()->error("[GimbalCommTest] Invalid argument: {}", e.what());
    print_usage();
    return 1;
  }

  utils::logger()->info(
    "[GimbalCommTest] port={}, baud={}, duration={:.1f}s, log_period={:.1f}s, "
    "protocol={}, tx_rate={:.1f}Hz, v1_frame={} bytes, v2_cmd={} bytes, "
    "v2_feedback={} bytes",
    options.port,
    options.baud,
    options.duration_sec,
    options.log_period_sec,
    protocol_name(options.protocol),
    options.tx_rate_hz,
    sizeof(GimbalFeedbackV1),
    sizeof(GimbalCommandWithSeq),
    sizeof(GimbalFeedbackV2));
  utils::logger()->warn(
    "[GimbalCommTest] Stop deep_node/cv_node/aimer_node/standard3_node/hero_node before "
    "running this test; Linux TTY bytes are not broadcast to multiple readers.");
  if (options.protocol == Protocol::V1) {
    utils::logger()->warn(
      "[GimbalCommTest] protocol=v1 requires lower feedback layout: bullet_count, "
      "feedback_seq(uint16_t), tail. It only measures feedback loss/frequency.");
  } else {
    utils::logger()->warn(
      "[GimbalCommTest] protocol=v2 sends mode=0 test commands with command_seq and "
      "requires lower 56-byte feedback with echo_command_seq and command counters.");
  }

  serial::Serial port;
  if (!open_serial(port, options)) {
    return 1;
  }

  const bool ok = options.protocol == Protocol::V1 ?
    run_rx_loop<GimbalFeedbackV1>(port, options) : run_v2_loop(port, options);

  port.close();
  return ok ? 0 : 2;
}
