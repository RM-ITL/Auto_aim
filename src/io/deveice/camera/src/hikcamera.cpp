#include "hikcamera.hpp"

#include <libusb-1.0/libusb.h>
#include <yaml-cpp/yaml.h>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include "logger.hpp"

namespace camera
{

using namespace std::chrono_literals;

namespace
{
void release_camera_handle(void *& handle)
{
  if (!handle) {
    return;
  }

  MV_CC_StopGrabbing(handle);
  MV_CC_CloseDevice(handle);
  MV_CC_DestroyHandle(handle);
  handle = nullptr;
}
}  // namespace

HikCamera::HikCamera(const std::string & config_path)
: config_path_(config_path), queue_(3)
{
  if (!load_config(config_path_)) {
    utils::logger()->error("[HikCamera] 配置加载失败: {}", config_path_);
    throw std::runtime_error("Failed to load camera configuration");
  }

  if (libusb_init(nullptr) != LIBUSB_SUCCESS) {
    utils::logger()->error("[HikCamera] libusb初始化失败");
    throw std::runtime_error("Failed to initialize libusb");
  }

  utils::PerformanceMonitor::Config perf_config;
  perf_config.enable_logging = true;
  perf_config.print_interval_sec = 5.0;
  perf_config.logger_name = "hikcamera";
  perf_monitor_.set_config(perf_config);
  perf_monitor_.register_metric("capture");
  perf_monitor_.reset_all();

  shutdown_ = false;
  daemon_quit_ = false;
  capture_quit_ = false;

  if (start_capture()) {
    utils::logger()->info("[HikCamera] 相机采集启动成功");
  } else {
    utils::logger()->warn("[HikCamera] 初始采集失败，守护线程将尝试恢复");
  }

  daemon_thread_ = std::thread(&HikCamera::daemon_loop, this);

  utils::logger()->info("[HikCamera] 初始化完成，配置路径: {}", config_path_);
}

HikCamera::~HikCamera()
{
  shutdown_ = true;
  daemon_quit_ = true;
  capture_quit_ = true;

  queue_.clear();
  queue_.push(CameraData{});

  if (daemon_thread_.joinable()) {
    daemon_thread_.join();
  }

  stop_capture();
  libusb_exit(nullptr);
  utils::logger()->info("[HikCamera] 模块已停止");
}

void HikCamera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);

  img = data.img;
  timestamp = data.timestamp;
}

void HikCamera::daemon_loop()
{
  while (!daemon_quit_) {
    if (!capturing_ && !shutdown_) {
      utils::logger()->warn("[HikCamera] 采集异常，尝试恢复...");
      stop_capture();
      reset_usb();
      std::this_thread::sleep_for(500ms);
      if (!start_capture()) {
        std::this_thread::sleep_for(1s);
        continue;
      }
    }

    std::this_thread::sleep_for(1s);
  }

  stop_capture();
}

void HikCamera::capture_loop()
{
  capturing_ = true;
  MV_FRAME_OUT_INFO_EX frame_info;
  std::memset(&frame_info, 0, sizeof(frame_info));

  while (!capture_quit_) {
    auto timer = perf_monitor_.create_timer("capture");

    int ret = MV_CC_GetOneFrameTimeout(
      camera_handle_, raw_buffer_.get(), payload_size_, &frame_info, 1000);

    if (ret == MV_OK) {
      auto timestamp = std::chrono::steady_clock::now();
      cv::Mat raw_image(frame_info.nHeight, frame_info.nWidth, CV_8UC1, raw_buffer_.get());
      cv::Mat rgb_image = convert_bayer(raw_image, frame_info.enPixelType);
      cv::Mat resized;
      cv::resize(rgb_image, resized, target_size_, 0, 0, cv::INTER_LINEAR);

      queue_.push({resized.clone(), timestamp});
      timer.set_success(true);
    } else {
      timer.set_success(false);
      capturing_ = false;
      break;
    }

    std::this_thread::sleep_for(1ms);
  }

  capturing_ = false;
}

bool HikCamera::start_capture()
{
  capturing_ = false;
  capture_quit_ = false;

  MV_CC_DEVICE_INFO_LIST device_list;
  std::memset(&device_list, 0, sizeof(device_list));
  if (MV_CC_EnumDevices(MV_USB_DEVICE, &device_list) != MV_OK || device_list.nDeviceNum == 0) {
    utils::logger()->error("[HikCamera] 未找到可用的海康相机");
    return false;
  }

  if (MV_CC_CreateHandle(&camera_handle_, device_list.pDeviceInfo[0]) != MV_OK) {
    utils::logger()->error("[HikCamera] 创建设备句柄失败");
    camera_handle_ = nullptr;
    return false;
  }

  if (MV_CC_OpenDevice(camera_handle_) != MV_OK) {
    utils::logger()->error("[HikCamera] 打开相机失败");
    release_camera_handle(camera_handle_);
    return false;
  }

  set_camera_parameters();

  MVCC_INTVALUE payload;
  std::memset(&payload, 0, sizeof(payload));
  if (MV_CC_GetIntValue(camera_handle_, "PayloadSize", &payload) != MV_OK) {
    utils::logger()->error("[HikCamera] 获取PayloadSize失败");
    release_camera_handle(camera_handle_);
    return false;
  }
  payload_size_ = payload.nCurValue;
  raw_buffer_.reset(new unsigned char[payload_size_]);

  if (MV_CC_StartGrabbing(camera_handle_) != MV_OK) {
    utils::logger()->error("[HikCamera] 启动采集失败");
    release_camera_handle(camera_handle_);
    raw_buffer_.reset();
    return false;
  }

  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
  capture_thread_ = std::thread(&HikCamera::capture_loop, this);
  utils::logger()->info("[HikCamera] 相机开始采集");
  return true;
}

void HikCamera::stop_capture()
{
  capture_quit_ = true;
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }

  release_camera_handle(camera_handle_);
  raw_buffer_.reset();
  capturing_ = false;
}

void HikCamera::set_camera_parameters()
{
  MV_CC_SetEnumValue(camera_handle_, "TriggerMode", MV_TRIGGER_MODE_OFF);
  MV_CC_SetEnumValue(camera_handle_, "BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
  MV_CC_SetEnumValue(camera_handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
  MV_CC_SetFloatValue(camera_handle_, "ExposureTime", exposure_us_);
  MV_CC_SetEnumValue(camera_handle_, "GainAuto", MV_GAIN_MODE_OFF);
  MV_CC_SetFloatValue(camera_handle_, "Gain", gain_);
  MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", fps_);
}

cv::Mat HikCamera::convert_bayer(const cv::Mat & raw, unsigned int type)
{
  cv::Mat bgr;
  static const std::unordered_map<unsigned int, int> bayer_map = {
    {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2BGR},
    {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2BGR},
    {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2BGR},
    {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2BGR}};

  auto it = bayer_map.find(type);
  if (it != bayer_map.end()) {
    cv::cvtColor(raw, bgr, it->second);
  } else if (type == PixelType_Gvsp_Mono8) {
    cv::cvtColor(raw, bgr, cv::COLOR_GRAY2BGR);
  } else {
    bgr = raw.clone();
  }
  return bgr;
}

void HikCamera::reset_usb() const
{
  libusb_device_handle * handle = libusb_open_device_with_vid_pid(nullptr, vid_, pid_);
  if (handle) {
    libusb_reset_device(handle);
    libusb_close(handle);
  }
}

bool HikCamera::load_config(const std::string & config_path)
{
  try {
    std::ifstream file(config_path);
    if (!file.good()) {
      utils::logger()->error("[HikCamera] 配置文件不存在: {}", config_path);
      return false;
    }
    file.close();

    YAML::Node config = YAML::LoadFile(config_path);
    const YAML::Node & params = config["camera"]["parameters"];

    exposure_us_ = params["exposure_ms"].as<double>() * 1000.0;
    gain_ = params["gain"].as<double>();
    fps_ = params["fps"].as<double>();
    target_size_ =
      cv::Size(params["target_width"].as<int>(), params["target_height"].as<int>());
    image_topic_ = params["image_topic"].as<std::string>();

    utils::logger()->info(
      "[HikCamera] 配置加载成功 - 曝光:{:.1f}ms 增益:{:.1f} FPS:{:.1f}",
      exposure_us_ / 1000.0, gain_, fps_);
    return true;

  } catch (const std::exception & e) {
    utils::logger()->error("[HikCamera] 配置加载异常: {}", e.what());
    return false;
  }
}

}  // namespace camera
