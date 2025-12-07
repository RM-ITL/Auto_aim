#ifndef SOLVER_NODE_HPP
#define SOLVER_NODE_HPP

#include "module/pnp_solver.hpp"
#include "module/coord_converter.hpp"
#include "module/optimize_yaw.hpp"
#include "module/solver.hpp"
#include "math_tools.hpp"
#include "armor.hpp"
#include <memory>
#include "logger.hpp"

namespace solver {

using ArmorType = armor_auto_aim::ArmorType;
using ArmorName = armor_auto_aim::ArmorName;
using Armors = armor_auto_aim::Armor;

class Solver {
public:
    explicit Solver(const std::string& yaml_config_path);
    
    void updateIMU(const Eigen::Quaterniond& q_absolute, double timestamp);
    void updateIMU(double yaw, double pitch, double timestamp);
    
    Armor_pose processArmor(Armors armor, double timestamp);
    
    Gimbal getCurrentGimbal() const;
    
    // bool isInitialized() const { return coord_converter_->isInitialized(); }
    YawPitch getCurrentAngles() const { return coord_converter_->getCurrentAngles(); }
    
    PnPResult getLastPnPResult() const { return last_pnp_result_; }
    Armor_pose getLastpose() const { return last_armor_pose_; }
    CoordConverter* getCoordConverter() { return coord_converter_.get(); }
    YawOptimizer* getYawOptimizer() { return yaw_optimizer_.get(); }
    
private:
    ArmorName parseArmorNumber(int number) const;
    ArmorType parseArmorType(const std::string& type_str) const;
    std::vector<cv::Point2f> extractCorners(Armors& armor) const;
    
private:
    std::unique_ptr<PnPSolver> pnp_solver_;
    std::unique_ptr<CoordConverter> coord_converter_;
    std::unique_ptr<YawOptimizer> yaw_optimizer_;
    
    Armor_pose last_armor_pose_;
    PnPResult last_pnp_result_;
};

}

#endif