#ifndef AUTO_BASE__LIGHT_TRACKER_HPP_
#define AUTO_BASE__LIGHT_TRACKER_HPP_

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "light_model.hpp"
#include "openvino_infer.hpp"

namespace auto_base
{

class LightTracker
{
public:
  explicit LightTracker(const std::string & config_path);

  std::list<LightTarget*> track(
    const std::vector<OpenvinoInfer::GreenLight> & detections,
    std::chrono::steady_clock::time_point t);

  std::string state() const;

private:
  std::string state_;  // "lost", "detecting", "tracking", "temp_lost"
  int detect_count_ = 0;
  int temp_lost_count_ = 0;

  std::unique_ptr<LightTarget> target_;
  std::chrono::steady_clock::time_point last_timestamp_;

  std::vector<OpenvinoInfer::GreenLight> detections_cache_;

  int min_detect_count_;        
  int max_temp_lost_count_;     

  void state_machine(bool found);

  bool set_target(
    const std::vector<OpenvinoInfer::GreenLight> & detections,
    std::chrono::steady_clock::time_point t);

  bool update_target(
    const std::vector<OpenvinoInfer::GreenLight> & detections,
    std::chrono::steady_clock::time_point t);

  OpenvinoInfer::GreenLight * find_best_detection(
    std::vector<OpenvinoInfer::GreenLight> & detections);
};

}  // namespace auto_base

#endif  // AUTO_BASE__LIGHT_TRACKER_HPP_
