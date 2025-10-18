#include "tracker_node.hpp"
#include <yaml-cpp/yaml.h>
#include <tf2/LinearMath/Quaternion.h>

namespace tracker {

TrackerNode::TrackerNode() 
    : Node("tracker_node") {
    
    this->declare_parameter("config_file", "/home/guo/ITL_sentry_auto_new/src/config/coord_converter.yaml");
    config_path_ = this->get_parameter("config_file").as_string();
    
    solver_ = std::make_unique<solver::Solver>(config_path_);
    tracker_ = std::make_unique<tracker::Tracker>(config_path_, *solver_);
    coord_converter_ = std::make_unique<solver::CoordConverter>(config_path_);
    
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    auto image_qos = rclcpp::QoS(rclcpp::KeepLast(1))
        .best_effort()
        .durability_volatile();
    
    // 订阅IMU数据
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data", qos,
        std::bind(&TrackerNode::imuCallback, this, std::placeholders::_1));
    
    // 订阅装甲板检测数据
    armor_sub_ = this->create_subscription<autoaim_msgs::msg::Armor>(
        "detected_data", qos,
        std::bind(&TrackerNode::armorCallback, this, std::placeholders::_1));
    
    // 订阅图像数据
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "yolo_detections_image", image_qos,
        std::bind(&TrackerNode::imageCallback, this, std::placeholders::_1));
    
    // 发布目标跟踪结果
    target_pub_ = this->create_publisher<autoaim_msgs::msg::Target>("target", 10);
    
    // 发布图像（供你自己决定何时使用）
    tracking_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "tracking_image", image_qos);
    
    RCLCPP_INFO(this->get_logger(), "TrackerNode初始化完成");
}

void TrackerNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    Eigen::Quaterniond q_imu(
        msg->orientation.w,
        msg->orientation.x,
        msg->orientation.y,
        msg->orientation.z
    );
    
    double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    solver_->updateIMU(q_imu, timestamp);
}

void TrackerNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
        // 将ROS图像消息转换为OpenCV格式
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        
        // 只保存图像数据和header，不做任何处理
        {
            std::lock_guard<std::mutex> lock(image_mutex_);
            latest_image_ = cv_ptr->image.clone();
            latest_image_header_ = msg->header;
        }
        
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "图像转换失败: %s", e.what());
    }
}

void TrackerNode::armorCallback(const autoaim_msgs::msg::Armor::SharedPtr msg) {
    if (!solver_->isInitialized()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "系统未初始化，等待IMU数据");
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(armor_mutex_);
        armor_buffer_.push_back(*msg);
        if (armor_buffer_.size() > 50) {
            armor_buffer_.pop_front();
        }
    }
    
    processArmors();
}

void TrackerNode::processArmors() {
    std::list<autoaim_msgs::msg::Armor> current_armors;
    {
        std::lock_guard<std::mutex> lock(armor_mutex_);
        current_armors = armor_buffer_;
        armor_buffer_.clear();
    }
    
    if (current_armors.empty()) {
        return;
    }
    
    auto now = std::chrono::steady_clock::now();
    auto targets = tracker_->track(current_armors, now, true);
    
    if (!targets.empty()) {
        cv::Mat img;
        std_msgs::msg::Header image_header;
        
        {
            std::lock_guard<std::mutex> lock(image_mutex_);
            if (latest_image_.empty()) {
                return;
            }
            img = latest_image_.clone();
            image_header = latest_image_header_;
        }
        
        for (const auto& target : targets) {
            auto x = target.ekf_x();
            auto armor_list = target.armor_xyza_list();
            auto armor_number = armor_list.size();
            
            for (const Eigen::Vector4d& xyza : armor_list) {
                Eigen::Vector3d world_point(xyza.x(), xyza.y(), xyza.z());
                Eigen::Vector3d camera_point = coord_converter_->transform(world_point, 
                                                            solver::CoordinateFrame::WORLD,
                                                            solver::CoordinateFrame::CAMERA);
                
                auto image_points = coord_converter_->reproject_armor(camera_point, xyza[3], target.armor_type, target.name);
                // utils::logger()->debug(
                //     "相机坐标系下的数据：[({:.2f}, {:.2f}, {:.2f})] 世界坐标系下的数据：[({:.3f}, {:.3f}, {:.3f})]",
                //     camera_point[0], camera_point[1], camera_point[2],
                //     world_point[0], world_point[1], world_point[2]
                // );

                // utils::logger()->debug(
                //     "图像点的数据是：[({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f})]",
                //     image_points[0].x, image_points[0].y,
                //     image_points[1].x, image_points[1].y,
                //     image_points[2].x, image_points[2].y,
                //     image_points[3].x, image_points[3].y
                // );
                utils::draw_quadrangle_with_corners(img, image_points, utils::colors::GREEN, 2, 3);
            }
        }
        
        auto image_msg = cv_bridge::CvImage(image_header, "bgr8", img).toImageMsg();
        tracking_image_pub_->publish(*image_msg);
    }
}

}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<tracker::TrackerNode>();
        RCLCPP_INFO(rclcpp::get_logger("main"), "启动TrackerNode...");
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "节点运行错误: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}