/**
 * @file track_queue_v3.hpp
 * @brief 4D跟踪滤波器 - 位置和朝向角联合估计
 */

#ifndef FILTER_LIB__TRACK_QUEUE_V3_HPP_
#define FILTER_LIB__TRACK_QUEUE_V3_HPP_

#include "base/ekf.hpp"
#include "base/math.hpp"
#include <Eigen/Dense>
#include <deque>
#include <chrono>

namespace filter_lib {

class TrackQueueV3 {
public:
    // 维度定义
    static constexpr int DIM_STATE = 11;
    static constexpr int DIM_MEAS = 4;
    
    // 类型别名
    using State = Eigen::Matrix<double, DIM_STATE, 1>;
    using Meas = Eigen::Matrix<double, DIM_MEAS, 1>;
    using StateCov = Eigen::Matrix<double, DIM_STATE, DIM_STATE>;
    using MeasCov = Eigen::Matrix<double, DIM_MEAS, DIM_MEAS>;
    using TimePoint = std::chrono::steady_clock::time_point;
    
    /**
     * @brief 观测数据结构
     */
    struct Observation {
        double x, y, z;
        double theta;
        double timestamp;
        
        Meas vec() const { 
            return Meas(x, y, z, theta); 
        }
        
        Observation(const Meas& m, double t) 
            : x(m(0)), y(m(1)), z(m(2)), theta(m(3)), timestamp(t) {}
            
        Observation(double x_, double y_, double z_, double theta_, double t)
            : x(x_), y(y_), z(z_), theta(theta_), timestamp(t) {}
    };
    
    /**
     * @brief 滤波器参数结构
     */
    struct Parameters {
        double q_pos;
        double q_vel;
        double q_acc;
        double q_angle;
        double q_omega;
        double q_alpha;
        double r_pos;
        double r_angle;
        double init_pos_cov;
        double init_vel_cov;
        double init_acc_cov;
        double init_angle_cov;
        double init_omega_cov;
        double init_alpha_cov;
        
        Parameters() 
            : q_pos(0.1),
              q_vel(0.5),
              q_acc(1.0),
              q_angle(0.05),
              q_omega(0.1),
              q_alpha(0.2),
              r_pos(0.05),
              r_angle(0.1),
              init_pos_cov(0.01),
              init_vel_cov(0.1),
              init_acc_cov(1.0),
              init_angle_cov(0.01),
              init_omega_cov(0.1),
              init_alpha_cov(1.0) {}
    };
    
    /**
     * @brief 跟踪状态信息
     */
    struct TrackingInfo {
        double speed = 0.0;
        double heading_rate = 0.0;
        Eigen::Vector3d acceleration;
        double angular_acceleration = 0.0;
        int observation_count = 0;
        double estimation_quality = 0.0;
    };

public:
    explicit TrackQueueV3(const Parameters& params);
    TrackQueueV3();
    
    bool init(const std::deque<Observation>& observations);
    void update(const Meas& meas, double dt);
    void update(const Meas& meas, TimePoint timestamp);
    State predict(double dt) const;
    
    State state() const { return ekf_.get_x(); }
    StateCov covariance() const { return ekf_.get_P(); }
    
    Eigen::Vector3d position() const { 
        return ekf_.get_x().template head<3>(); 
    }
    
    Eigen::Vector3d velocity() const {
        const auto& x = ekf_.get_x();
        return Eigen::Vector3d(x(4), x(5), x(6));
    }
    
    Eigen::Vector3d acceleration() const {
        const auto& x = ekf_.get_x();
        return Eigen::Vector3d(x(8), x(9), 0.0);
    }
    
    double orientation() const { return ekf_.get_x()(3); }
    double angular_velocity() const { return ekf_.get_x()(7); }
    double angular_acceleration() const { return ekf_.get_x()(10); }
    
    TrackingInfo getTrackingInfo() const;
    bool isInitialized() const { return initialized_; }
    int getUpdateCount() const { return update_count_; }
    
    std::pair<State, StateCov> getStateWithCovariance() const;
    
    void reset();
    void setParameters(const Parameters& params) { params_ = params; }

private:
    struct TransitionModel {
        double dt;
        
        TransitionModel(double dt_) : dt(dt_) {}
        
        template<typename T>
        void operator()(const T x_prev[DIM_STATE], T x_curr[DIM_STATE]) const;
    };
    
    struct MeasurementModel {
        template<typename T>
        void operator()(const T x[DIM_STATE], T y[DIM_MEAS]) const {
            y[0] = x[0];
            y[1] = x[1];
            y[2] = x[2];
            y[3] = x[3];
        }
    };

private:
    bool estimateInitialState(const std::deque<Observation>& observations,
                              State& init_state);
    StateCov buildProcessNoise(double dt) const;
    MeasCov buildMeasurementNoise() const;
    void updateTrackingInfo();
    static double normalizeAngle(double angle);

private:
    AdaptiveEkf<DIM_STATE, DIM_MEAS> ekf_;
    bool initialized_ = false;
    int update_count_ = 0;
    TimePoint last_update_time_;
    Parameters params_;
    TrackingInfo tracking_info_;
};

} // namespace filter_lib

#endif // FILTER_LIB__TRACK_QUEUE_V3_HPP_