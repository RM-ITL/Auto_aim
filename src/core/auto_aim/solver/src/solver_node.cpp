#include "solver_node.hpp"
#include <rclcpp/rclcpp.hpp>

namespace solver {

Solver::Solver(const std::string& yaml_config_path) {
    // 所有模块从同一个YAML文件初始化
    pnp_solver_ = std::make_unique<PnPSolver>(yaml_config_path);
    coord_converter_ = std::make_unique<CoordConverter>(yaml_config_path);
    yaw_optimizer_ = std::make_unique<YawOptimizer>(yaml_config_path, coord_converter_.get());
    
    RCLCPP_INFO(rclcpp::get_logger("Solver"), 
                "Solver系统初始化完成，配置文件: %s", 
                yaml_config_path.c_str());
}

void Solver::updateIMU(const Eigen::Quaterniond& q_absolute, double timestamp) {
    coord_converter_->updateIMU(q_absolute, timestamp);
}

void Solver::updateIMU(double yaw, double pitch, double timestamp) {
    coord_converter_->updateIMU(yaw, pitch, timestamp);
}

Armor_pose Solver::processArmor(Armors armor, 
                               double timestamp) {
    Armor_pose armor_pose;
    armor_pose.timestamp = timestamp;
    
    armor_pose.id = armor.name;
    armor_pose.type = armor.type;
    
    std::vector<cv::Point2f> corners = extractCorners(armor);
    
    
    last_pnp_result_ = pnp_solver_->solvePnP(corners, armor_pose.type, timestamp);

    
    if (!last_pnp_result_.success) {
        RCLCPP_WARN(rclcpp::get_logger("Solver"), 
                   "PnP求解失败，装甲板编号: %d", armor.name);
        return armor_pose;
    }
    
    // 判断是否需要优化
    Eigen::Matrix3d R_armor_to_camera = last_pnp_result_.R_armor_to_camera;   // pnp出来的旋转矩阵
    
    bool need_optimization = true;
    // 大装甲板的3、4、5号不需要优化（通常是步兵）
    if (armor_pose.type == armor_auto_aim::ArmorType::big &&
        (armor_pose.id == armor_auto_aim::ArmorName::three ||
         armor_pose.id == armor_auto_aim::ArmorName::four ||
         armor_pose.id == armor_auto_aim::ArmorName::five)) {
        need_optimization = false;
    }

    const auto rotations = coord_converter_->getCameraToWorldRotation();
    [[maybe_unused]] const auto& R_gimbal_to_world = rotations.first;
    [[maybe_unused]] const auto& R_camera_to_gimbal = rotations.second;
    

    // Eigen::Matrix3d optimized_R_armor_to_camera = Eigen::Matrix3d::Identity();   // 假设此时是正对的相机的旋转矩阵,R_armor_to_world

    // 进yaw优化
    
    // if (need_optimization) {
    //     optimized_R_armor_to_camera = yaw_optimizer_->optimizeWithPrior(
    //         last_pnp_result_.camera_position,
    //         last_pnp_result_.R_armor_to_camera,
    //         armor_pose.type,
    //         armor_pose.id,
    //         corners,
    //         *coord_converter_
    //     );
    // }

    // Eigen::Vector3d normal_camera = optimized_R_armor_to_camera.col(2); // 提取法向量

    // Eigen::Vector3d normal_world = R_camera_to_world * normal_camera;   // 变换到世界坐标系下


    // double yaw_world = std::atan2(normal_world[1], normal_world[0]);

    // utils::logger()->debug("原始的装甲板到相机的旋转矩阵是");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //         R_armor_to_camera(i,0), R_armor_to_camera(i,1), R_armor_to_camera(i,2));
    // }

    // utils::logger()->debug("云台到世界坐标系");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //         R_gimbal_to_world(i,0), R_gimbal_to_world(i,1), R_gimbal_to_world(i,2));
    // }

    // utils::logger()->debug("相机到云台坐标系");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //         R_camera_to_gimbal(i,0), R_camera_to_gimbal(i,1), R_camera_to_gimbal(i,2));
    // }


    // utils::logger()->debug(
    //     "提取相机坐标系下的法向量为：{:.2f}, {:.2f}, {:.2f}",
    //     normal_camera[0],normal_camera[1],normal_camera[2]
    // );

    // utils::logger()->debug(
    //     "提取世界坐标系下的法向量为：{:.2f}, {:.2f}, {:.2f}",
    //     normal_world[0],normal_world[1],normal_world[2]
    // );

    // utils::logger()->debug(
    //     "向量法提取出的yaw_world角是:{:.2f}",
    //     yaw_world
    // );

    // utils::logger()->debug(
    //     "向量法提取的yaw_camera角是:{:.2f}",
    //     yaw_camera
    // );


    
    // utils::logger()->debug("真实的装甲板到相机");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //         R_armor_to_camera(i,0), R_armor_to_camera(i,1), R_armor_to_camera(i,2));
    // }

    // utils::logger()->debug("装甲板到相机：");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //         optimized_R_armor_to_camera(i,0), optimized_R_armor_to_camera(i,1), optimized_R_armor_to_camera(i,2));
    // }


    
    // 保存相机坐标系结果
    armor_pose.camera_position = last_pnp_result_.camera_position / 1000.0;
    armor_pose.gimbal_position = coord_converter_->transform(
        last_pnp_result_.camera_position / 1000.0,
        CoordinateFrame::CAMERA,
        CoordinateFrame::GIMBAL
    );
    Eigen::Matrix3d R_armor_to_gimbal = coord_converter_->transformRotation(
        R_armor_to_camera,
        CoordinateFrame::CAMERA,
        CoordinateFrame::GIMBAL
    );



    armor_pose.gimbal_orientation = Orientation(R_armor_to_gimbal, solver::EulerOrder::ZYX, timestamp);


    // 转换到世界坐标系
    armor_pose.world_position = coord_converter_->transform(
        last_pnp_result_.camera_position / 1000.0,
        CoordinateFrame::CAMERA,
        CoordinateFrame::WORLD
    );
    
    Eigen::Matrix3d R_armor_to_world = coord_converter_->transformRotation(
        R_armor_to_camera,
        CoordinateFrame::CAMERA,
        CoordinateFrame::WORLD
    );

    Eigen::Matrix3d optimized_R_armor_to_world = R_armor_to_world;

    if (need_optimization) {
        optimized_R_armor_to_world = yaw_optimizer_->optimize_yaw(
            armor_pose.world_position,
            R_armor_to_world,
            armor_pose.type,
            armor_pose.id,
            corners,
            *coord_converter_
        );
    }

    std::vector<cv::Point2f> image_points = yaw_optimizer_->reproject_armor(
        armor_pose.world_position,
        R_armor_to_world,
        armor_pose.type
    );
    
    // utils::logger()->debug(
    //     "图像点的数据是：[({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f})]",
    //     image_points[0].x, image_points[0].y,
    //     image_points[1].x, image_points[1].y,
    //     image_points[2].x, image_points[2].y,
    //     image_points[3].x, image_points[3].y
    // );
    armor_pose.world_orientation = Orientation(
       optimized_R_armor_to_world, solver::EulerOrder::ZYX,timestamp
    );

    [[maybe_unused]] Eigen::Matrix3d test_R_armor_to_camera = coord_converter_->transformRotation(
        R_armor_to_world,
        CoordinateFrame::WORLD,
        CoordinateFrame::CAMERA
    );

    // utils::logger()->debug("计算的相机到云台的旋转矩阵是");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //         R_camera_to_gimbal(i,0), R_camera_to_gimbal(i,1), R_camera_to_gimbal(i,2));
    // }


    // utils::logger()->debug(
    //     "解算得到的世界坐标系下目标的位置为x:{:.2f}, y:{:.2f}, z:{:.2f}",
    //     armor_pose.world_position[0],
    //     armor_pose.world_position[1],
    //     armor_pose.world_position[2]
    // );
    

    // Eigen::Matrix3d optimized_R_armor_to_world = Eigen::Matrix3d::Zero();


    // utils::logger()->debug("装甲板到世界");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //         R_armor_to_world(i,0), R_armor_to_world(i,1), R_armor_to_world(i,2));
    // }
    

    // armor_pose.world_orientation.yaw = yaw_world;



    // utils::logger()->debug(
    //     "解算得到目标在云台坐标系下的为yaw:{:.2f}, pitch为{:.2f}, roll为{:.2f}",
    //     armor_pose.gimbal_orientation.yaw, 
    //     armor_pose.gimbal_orientation.pitch, 
    //     armor_pose.gimbal_orientation.roll
    // );


    // utils::logger()->debug(
    //     "解算得到目标在世界坐标系下的为yaw:{:.2f}, pitch为{:.2f}, roll为{:.2f}",
    //     armor_pose.world_orientation.yaw, 
    //     armor_pose.world_orientation.pitch, 
    //     armor_pose.world_orientation.roll
    // );

    // utils::logger()->debug("测试旋转矩阵：");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //         test_R_armor_to_camera(i,0), optimized_R_armor_to_world(i,1), optimized_R_armor_to_world(i,2));
    // }

    // armor_pose.world_orientation =  armor_pose.camera_orientation;
    
    armor_pose.world_spherical = SphericalCoord(armor_pose.world_position, timestamp);
    
    last_armor_pose_ = armor_pose;

    // 这里armor_pose的世界球坐标和姿态都需要是弧度值
    
    
    return armor_pose;
}

Gimbal Solver::getCurrentGimbal() const {
    YawPitch current = coord_converter_->getCurrentAngles();
    
    if (last_armor_pose_.world_position.norm() > 1e-6) {
        return Gimbal(current, last_armor_pose_.world_position, current.timestamp);
    }
    
    return Gimbal(current, current, current.timestamp);
}

ArmorName Solver::parseArmorNumber(int number) const {
    switch(number) {
        case 1: return armor_auto_aim::ArmorName::one;
        case 2: return armor_auto_aim::ArmorName::two;
        case 3: return armor_auto_aim::ArmorName::three;
        case 4: return armor_auto_aim::ArmorName::four;
        case 5: return armor_auto_aim::ArmorName::five;
        case 6: return armor_auto_aim::ArmorName::sentry;
        case 7: return armor_auto_aim::ArmorName::outpost;
        case 8: return armor_auto_aim::ArmorName::base;
        default: return armor_auto_aim::ArmorName::not_armor;
    }
}

ArmorType Solver::parseArmorType(const std::string& type_str) const {
    if (type_str == "big" || type_str == "large") {
        return armor_auto_aim::ArmorType::big;
    }
    return armor_auto_aim::ArmorType::small;
}

std::vector<cv::Point2f> Solver::extractCorners(Armors& armor) const {
    std::vector<cv::Point2f> corners;
    corners.reserve(4);
    
    for (const auto& point : armor.points) {
        corners.emplace_back(point.x, point.y);
    }
    
    return corners;
}

}
