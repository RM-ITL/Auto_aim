/**
 * @file test_model.cpp
 * @brief 解耦式运动模型实现
 */

#include "common_model_v1.hpp"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace motion_model {

// 构造函数
TestModel::TestModel(const std::string& model_name,
                     size_t init_threshold,
                     double tracker_timeout,
                     bool enable_coupling)
    : model_name_(model_name),
      init_threshold_(init_threshold),
      tracker_timeout_(tracker_timeout),
      enable_coupling_(enable_coupling),
      is_initialized_(false),
      update_count_(0),
      last_update_time_(0.0),
      enable_adaptive_noise_(false),
      confidence_scale_(1.0),
      log_level_(1) {
    
    // 设置默认的滤波器参数
    tracker_params_ = filter_lib::DecoupledTracker::Parameters();
    
    // 如果不启用耦合，设置最大耦合强度为0
    if (!enable_coupling) {
        tracker_params_.max_rho = 0.0;
    }
    
    if (log_level_ > 0) {
        std::cout << "[TestModel] Created model: " << model_name
                  << ", init_threshold: " << init_threshold
                  << ", coupling: " << (enable_coupling ? "enabled" : "disabled")
                  << std::endl;
    }
}

// 更新观测数据
bool TestModel::updateObservation(const DecoupledObservation& observation) {
    // 检查观测有效性
    if (!isObservationValid(observation)) {
        if (log_level_ > 0) {
            std::cout << "[TestModel] Invalid observation rejected at t=" 
                      << observation.timestamp << std::endl;
        }
        return false;
    }
    
    // 如果尚未初始化，收集数据
    if (!is_initialized_) {
        collectInitData(observation);
        
        // 尝试初始化
        if (init_observations_.size() >= init_threshold_) {
            if (tryInitialize()) {
                if (log_level_ > 0) {
                    std::cout << "[TestModel] Successfully initialized with "
                              << init_observations_.size() << " observations" << std::endl;
                }
            } else {
                if (log_level_ > 0) {
                    std::cout << "[TestModel] Initialization failed, clearing data" << std::endl;
                }
                init_observations_.clear();
                return false;
            }
        } else {
            if (log_level_ > 1) {
                std::cout << "[TestModel] Collecting init data: "
                          << init_observations_.size() << "/" << init_threshold_ << std::endl;
            }
            return true;
        }
    }
    
    // 检查超时
    if (update_count_ > 0 && (observation.timestamp - last_update_time_) > tracker_timeout_) {
        if (log_level_ > 0) {
            std::cout << "[TestModel] Timeout detected (gap=" 
                      << observation.timestamp - last_update_time_ 
                      << "s), resetting..." << std::endl;
        }
        reset();
        collectInitData(observation);
        return true;
    }
    
    // 更新滤波器
    auto measurement = observationToMeasurement(observation);
    auto time_point = timestampToTimePoint(observation.timestamp);
    
    // 自适应调整测量噪声
    if (enable_adaptive_noise_) {
        auto current_params = tracker_params_;
        
        // 根据置信度调整噪声
        double pos_noise_factor = 2.0 - observation.position_confidence;
        double ori_noise_factor = 2.0 - observation.orientation_confidence;
        
        current_params.trans_r *= pos_noise_factor * confidence_scale_;
        current_params.pose_r *= ori_noise_factor * confidence_scale_;
        
        tracker_->setParameters(current_params);
    }
    
    // 执行更新
    tracker_->update(measurement, time_point);
    
    // 更新状态
    last_update_time_ = observation.timestamp;
    update_count_++;
    
    // 记录运动快照
    if (motion_history_.size() >= MAX_HISTORY_SIZE) {
        motion_history_.pop_front();
    }
    
    auto state = tracker_->getFusedState();
    auto coupling_info = tracker_->getCouplingInfo();
    
    MotionSnapshot snapshot;
    snapshot.timestamp = observation.timestamp;
    snapshot.position = Eigen::Vector3d(state(0), state(1), state(2));
    snapshot.velocity = Eigen::Vector3d(state(4), state(5), state(6));
    snapshot.orientation = state(3);
    snapshot.angular_velocity = state(7);
    snapshot.coupling_strength = coupling_info.rho;
    snapshot.motion_mode = coupling_info.mode_description;
    
    motion_history_.push_back(snapshot);
    
    if (log_level_ > 1) {
        std::cout << "[TestModel] Update #" << update_count_ 
                  << " at t=" << std::fixed << std::setprecision(3) << observation.timestamp
                  << ", mode: " << coupling_info.mode_description
                  << ", ρ=" << std::setprecision(2) << coupling_info.rho << std::endl;
    }
    
    return true;
}

// 获取预测结果
DecoupledPrediction TestModel::getPrediction(double prediction_time) const {
    DecoupledPrediction prediction;
    
    if (!is_initialized_ || !tracker_) {
        if (log_level_ > 0) {
            std::cout << "[TestModel] WARNING: getPrediction called on uninitialized model" << std::endl;
        }
        prediction.is_valid = false;
        return prediction;
    }
    
    // 获取预测值
    auto pred_measurement = tracker_->predict(prediction_time);
    
    // 填充位置和朝向
    prediction.position = Eigen::Vector3d(pred_measurement(0), pred_measurement(1), pred_measurement(2));
    prediction.orientation = pred_measurement(3);
    
    // 获取完整状态用于速度和加速度
    auto current_state = tracker_->getFusedState();
    
    // 使用简单的运动学公式预测未来速度和加速度
    // （这是一个近似，因为predict函数只返回位置）
    prediction.velocity = Eigen::Vector3d(current_state(4), current_state(5), current_state(6));
    prediction.acceleration = Eigen::Vector3d(current_state(8), current_state(9), 0.0);
    
    // 角速度和角加速度
    prediction.angular_velocity = current_state(7);
    prediction.angular_acceleration = current_state(10);
    
    // 获取耦合信息
    auto coupling_info = tracker_->getCouplingInfo();
    prediction.coupling_strength = coupling_info.rho;
    prediction.motion_consistency = coupling_info.consistency;
    prediction.velocity_angle = coupling_info.velocity_angle;
    prediction.heading_velocity_diff = coupling_info.angle_diff;
    prediction.motion_mode = coupling_info.mode_description;
    prediction.speed = coupling_info.speed;
    
    // 获取不确定性估计
    auto [trans_state, trans_cov] = tracker_->getTransStateWithCovariance();
    auto [pose_state, pose_cov] = tracker_->getPoseStateWithCovariance();
    
    // 提取标准差
    prediction.position_std = Eigen::Vector3d(
        std::sqrt(trans_cov(0, 0)),
        std::sqrt(trans_cov(1, 1)),
        std::sqrt(trans_cov(2, 2))
    );
    
    prediction.velocity_std = Eigen::Vector3d(
        std::sqrt(trans_cov(3, 3)),
        std::sqrt(trans_cov(4, 4)),
        std::sqrt(trans_cov(5, 5))
    );
    
    prediction.orientation_std = std::sqrt(pose_cov(0, 0));
    prediction.angular_velocity_std = std::sqrt(pose_cov(1, 1));
    
    // 计算预测置信度
    prediction.prediction_confidence = calculatePredictionConfidence(prediction_time);
    
    // 检查预测有效性
    prediction.is_valid = (prediction.position.norm() > 1e-6) && 
                         (prediction.position.norm() < 1000.0) &&
                         std::isfinite(prediction.position.norm()) &&
                         (prediction.prediction_confidence > 0.1);
    
    if (!prediction.is_valid && log_level_ > 0) {
        std::cout << "[TestModel] WARNING: Invalid prediction detected" << std::endl;
    }
    
    return prediction;
}

// 批量更新
int TestModel::batchUpdate(const std::vector<DecoupledObservation>& observations,
                           bool force_init) {
    if (force_init) {
        reset();
    }
    
    int success_count = 0;
    for (const auto& obs : observations) {
        if (updateObservation(obs)) {
            success_count++;
        }
    }
    
    if (log_level_ > 0) {
        std::cout << "[TestModel] Batch update: " << success_count 
                  << "/" << observations.size() << " observations processed" << std::endl;
    }
    
    return success_count;
}

// 获取状态信息
void TestModel::getStatusInfo(std::vector<std::string>& info) const {
    info.clear();
    std::stringstream ss;
    
    // 基本信息
    ss << "Model Name: " << model_name_;
    info.push_back(ss.str());
    ss.str("");
    
    ss << "Initialized: " << (is_initialized_ ? "Yes" : "No");
    info.push_back(ss.str());
    ss.str("");
    
    ss << "Update Count: " << update_count_;
    info.push_back(ss.str());
    ss.str("");
    
    ss << "Last Update Time: " << std::fixed << std::setprecision(3) << last_update_time_ << " s";
    info.push_back(ss.str());
    ss.str("");
    
    ss << "Coupling: " << (enable_coupling_ ? "Enabled" : "Disabled");
    info.push_back(ss.str());
    ss.str("");
    
    // 如果已初始化，获取详细状态
    if (is_initialized_ && tracker_) {
        info.push_back("--- Current State ---");
        
        auto state = tracker_->getFusedState();
        ss << "Position: [" << std::fixed << std::setprecision(3) 
           << state(0) << ", " << state(1) << ", " << state(2) << "] m";
        info.push_back(ss.str());
        ss.str("");
        
        ss << "Velocity: [" << std::fixed << std::setprecision(3) 
           << state(4) << ", " << state(5) << ", " << state(6) << "] m/s";
        info.push_back(ss.str());
        ss.str("");
        
        double speed = std::sqrt(state(4)*state(4) + state(5)*state(5) + state(6)*state(6));
        ss << "Speed: " << std::fixed << std::setprecision(3) << speed << " m/s";
        info.push_back(ss.str());
        ss.str("");
        
        ss << "Orientation: " << std::fixed << std::setprecision(3) 
           << state(3) * 180.0 / M_PI << " deg";
        info.push_back(ss.str());
        ss.str("");
        
        ss << "Angular Velocity: " << std::fixed << std::setprecision(3) 
           << state(7) * 180.0 / M_PI << " deg/s";
        info.push_back(ss.str());
        ss.str("");
        
        // 耦合信息
        auto coupling_info = tracker_->getCouplingInfo();
        info.push_back("--- Coupling Info ---");
        
        ss << "Coupling Strength (ρ): " << std::fixed << std::setprecision(3) 
           << coupling_info.rho;
        info.push_back(ss.str());
        ss.str("");
        
        ss << "Motion Mode: " << coupling_info.mode_description;
        info.push_back(ss.str());
        ss.str("");
        
        ss << "Velocity Angle: " << std::fixed << std::setprecision(1) 
           << coupling_info.velocity_angle * 180.0 / M_PI << " deg";
        info.push_back(ss.str());
        ss.str("");
        
        ss << "Heading-Velocity Diff: " << std::fixed << std::setprecision(1) 
           << coupling_info.angle_diff * 180.0 / M_PI << " deg";
        info.push_back(ss.str());
        ss.str("");
        
        ss << "Motion Consistency: " << std::fixed << std::setprecision(3) 
           << coupling_info.consistency;
        info.push_back(ss.str());
    }
}

// 设置滤波器参数
void TestModel::setFilterParameters(const filter_lib::DecoupledTracker::Parameters& params) {
    tracker_params_ = params;
    
    // 如果不启用耦合，强制设置最大耦合强度为0
    if (!enable_coupling_) {
        tracker_params_.max_rho = 0.0;
    }
    
    if (tracker_) {
        tracker_->setParameters(tracker_params_);
    }
    
    if (log_level_ > 0) {
        std::cout << "[TestModel] Filter parameters updated" << std::endl;
    }
}

// 设置耦合参数
void TestModel::setCouplingParameters(double coupling_sigma, double min_speed, 
                                      double rho_threshold) {
    tracker_params_.coupling_sigma = coupling_sigma;
    tracker_params_.min_speed = min_speed;
    tracker_params_.rho_threshold = rho_threshold;
    
    if (tracker_) {
        tracker_->setParameters(tracker_params_);
    }
    
    if (log_level_ > 0) {
        std::cout << "[TestModel] Coupling parameters set: sigma=" << coupling_sigma
                  << ", min_speed=" << min_speed
                  << ", threshold=" << rho_threshold << std::endl;
    }
}

// 设置噪声自适应参数
void TestModel::setAdaptiveNoiseParameters(bool enable_adaptive, double confidence_scale) {
    enable_adaptive_noise_ = enable_adaptive;
    confidence_scale_ = confidence_scale;
    
    if (log_level_ > 0) {
        std::cout << "[TestModel] Adaptive noise " 
                  << (enable_adaptive ? "enabled" : "disabled")
                  << ", scale=" << confidence_scale << std::endl;
    }
}

// 重置模型
void TestModel::reset() {
    if (log_level_ > 0) {
        std::cout << "[TestModel] Resetting model" << std::endl;
    }
    
    tracker_.reset();
    is_initialized_ = false;
    update_count_ = 0;
    last_update_time_ = 0.0;
    init_observations_.clear();
    motion_history_.clear();
}

// 检查超时
bool TestModel::isTimeout(double current_time) const {
    if (!is_initialized_ || update_count_ == 0) {
        return false;
    }
    
    double time_since_update = current_time - last_update_time_;
    return time_since_update > tracker_timeout_;
}

// 获取运动分析报告
std::string TestModel::getMotionAnalysisReport() const {
    std::stringstream report;
    
    report << "=== Motion Analysis Report ===" << std::endl;
    report << "Model: " << model_name_ << std::endl;
    report << "Total Updates: " << update_count_ << std::endl;
    
    if (motion_history_.size() < 2) {
        report << "Insufficient data for analysis" << std::endl;
        return report.str();
    }
    
    // 分析运动模式分布
    std::map<std::string, int> mode_counts;
    double avg_speed = 0.0;
    double avg_coupling = 0.0;
    double max_speed = 0.0;
    double max_angular_vel = 0.0;
    
    for (const auto& snapshot : motion_history_) {
        mode_counts[snapshot.motion_mode]++;
        double speed = snapshot.velocity.norm();
        avg_speed += speed;
        avg_coupling += snapshot.coupling_strength;
        max_speed = std::max(max_speed, speed);
        max_angular_vel = std::max(max_angular_vel, std::abs(snapshot.angular_velocity));
    }
    
    avg_speed /= motion_history_.size();
    avg_coupling /= motion_history_.size();
    
    report << "\n--- Motion Statistics ---" << std::endl;
    report << "Average Speed: " << std::fixed << std::setprecision(3) 
           << avg_speed << " m/s" << std::endl;
    report << "Max Speed: " << max_speed << " m/s" << std::endl;
    report << "Max Angular Velocity: " << max_angular_vel * 180.0 / M_PI 
           << " deg/s" << std::endl;
    report << "Average Coupling Strength: " << avg_coupling << std::endl;
    
    report << "\n--- Motion Mode Distribution ---" << std::endl;
    for (const auto& [mode, count] : mode_counts) {
        double percentage = 100.0 * count / motion_history_.size();
        report << mode << ": " << count << " (" 
               << std::fixed << std::setprecision(1) << percentage << "%)" << std::endl;
    }
    
    // 分析运动变化
    double total_distance = 0.0;
    double total_rotation = 0.0;
    
    for (size_t i = 1; i < motion_history_.size(); ++i) {
        const auto& prev = motion_history_[i-1];
        const auto& curr = motion_history_[i];
        
        total_distance += (curr.position - prev.position).norm();
        
        double angle_change = std::abs(curr.orientation - prev.orientation);
        if (angle_change > M_PI) angle_change = 2*M_PI - angle_change;
        total_rotation += angle_change;
    }
    
    double time_span = motion_history_.back().timestamp - motion_history_.front().timestamp;
    
    report << "\n--- Path Analysis ---" << std::endl;
    report << "Time Span: " << std::fixed << std::setprecision(2) << time_span << " s" << std::endl;
    report << "Total Distance: " << std::setprecision(3) << total_distance << " m" << std::endl;
    report << "Total Rotation: " << total_rotation * 180.0 / M_PI << " deg" << std::endl;
    report << "Average Linear Velocity: " << total_distance / time_span << " m/s" << std::endl;
    report << "Average Angular Velocity: " << (total_rotation / time_span) * 180.0 / M_PI 
           << " deg/s" << std::endl;
    
    return report.str();
}

// 私有辅助函数实现

filter_lib::DecoupledTracker::MeasVec 
TestModel::observationToMeasurement(const DecoupledObservation& obs) const {
    filter_lib::DecoupledTracker::MeasVec meas;
    meas << obs.position.x(), obs.position.y(), obs.position.z(), obs.orientation;
    return meas;
}

std::chrono::steady_clock::time_point 
TestModel::timestampToTimePoint(double timestamp) const {
    return std::chrono::steady_clock::time_point(
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(timestamp)));
}

bool TestModel::isObservationValid(const DecoupledObservation& obs) const {
    // 检查位置是否合理
    if (!obs.position.allFinite() || obs.position.norm() > 1000.0) {
        return false;
    }
    
    // 检查朝向是否在合理范围
    if (!std::isfinite(obs.orientation) || 
        std::abs(obs.orientation) > 10 * M_PI) {
        return false;
    }
    
    // 检查时间戳
    if (!std::isfinite(obs.timestamp) || obs.timestamp < 0) {
        return false;
    }
    
    // 检查置信度
    if (obs.position_confidence < 0 || obs.position_confidence > 1 ||
        obs.orientation_confidence < 0 || obs.orientation_confidence > 1) {
        return false;
    }
    
    return true;
}

double TestModel::calculatePredictionConfidence(double prediction_time) const {
    // 基础置信度
    double confidence = 1.0;
    
    // 根据更新次数调整
    if (update_count_ < 10) {
        confidence *= 0.5;
    } else if (update_count_ < 50) {
        confidence *= 0.8;
    }
    
    // 根据预测时间调整
    if (prediction_time > 0.5) {
        confidence *= std::exp(-prediction_time);
    }
    
    // 根据最近的耦合强度调整
    if (!motion_history_.empty()) {
        double recent_coupling = motion_history_.back().coupling_strength;
        confidence *= (0.5 + 0.5 * recent_coupling);
    }
    
    return std::max(0.0, std::min(1.0, confidence));
}

void TestModel::collectInitData(const DecoupledObservation& obs) {
    Eigen::Matrix<double, 5, 1> init_obs;
    init_obs << obs.position.x(), obs.position.y(), obs.position.z(),
                obs.orientation, obs.timestamp;
    
    init_observations_.push_back(init_obs);
    
    // 限制缓存大小
    while (init_observations_.size() > init_threshold_ * 2) {
        init_observations_.pop_front();
    }
}

bool TestModel::tryInitialize() {
    if (init_observations_.size() < init_threshold_) {
        return false;
    }
    
    // 创建滤波器
    tracker_ = std::make_unique<filter_lib::DecoupledTracker>(tracker_params_);
    
    // 初始化
    if (!tracker_->init(init_observations_)) {
        tracker_.reset();
        return false;
    }
    
    is_initialized_ = true;
    update_count_ = init_observations_.size();
    last_update_time_ = init_observations_.back()(4);
    
    // 清空初始化缓存
    init_observations_.clear();
    
    return true;
}

} // namespace motion_model