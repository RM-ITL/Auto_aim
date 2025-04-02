#include "/home/shaobing/rm_ws/src/anlesover/include/anlesover/AngleSolver_test.hpp"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include "math.h"
#include <iostream>

using namespace cv;	
using namespace std;
using std::placeholders::_1;

namespace rm
{
std::vector<cv::Point3f> AngleSolverParam::POINT_3D_OF_ARMOR_BIG = std::vector<cv::Point3f>
{

		cv::Point3f(-105, -30, 0),	//tl
		cv::Point3f(105, -30, 0),	//tr
		cv::Point3f(105, 30, 0),	//br
		cv::Point3f(-105, 30, 0)	//bl
};
std::vector<cv::Point3f> AngleSolverParam::POINT_3D_OF_RUNE = std::vector<cv::Point3f>
{
	cv::Point3f(-370, -220, 0),
	cv::Point3f(0, -220, 0),
	cv::Point3f(370, -220, 0),
	cv::Point3f(-370, 0, 0),
	cv::Point3f(0, 0, 0),
	cv::Point3f(370, 0, 0),
	cv::Point3f(-370, 220, 0),
	cv::Point3f(0, 220, 0),
	cv::Point3f(370, 220, 0)
};

std::vector<cv::Point3f> AngleSolverParam::POINT_3D_OF_ARMOR_SMALL = std::vector<cv::Point3f>
{
	cv::Point3f(-68.5, -66.5, 0),	//tl
	cv::Point3f(68.5, -66.5, 0),	//tr
	cv::Point3f(68.5, 66.5, 0),		//br
	cv::Point3f(-68.5, 66.5, 0)		//bl
};

AngleSolver::AngleSolver()
{
	for(int ll = 0; ll <= 3; ll++)
		target_nothing.push_back(cv::Point2f(0.0, 0.0));
};

void AngleSolver::init(const AngleSolverParam& AngleSolverParam)
{
	param = AngleSolverParam;
	_cam_instant_matrix = param.CAM_MATRIX.clone();
};

//algorithm=0 pnp 1 pinhole
void AngleSolver::set_Target(const std::vector<cv::Point2f> objectPoints, int objectType)//objectType=0 BIG 1 SMALL
{
	if(objectType == 0 || objectType == 1)
	{
		if(angle_solver_algorithm == 1 || angle_solver_algorithm == 2)
		{
			angle_solver_algorithm = 0; cout << "algorithm is reset to PNP Solution" << endl;
		}
		point_2d_of_armor = objectPoints;
		if(objectType == 0)
			enemy_type = 0;
		else
			enemy_type = 1;
		return;
	}
	// if(objectType == 3 || objectType == 4)
	// {
	// 	angle_solver_algorithm = 2;
	// 	point_2d_of_rune = objectPoints;
	// }

}

void AngleSolver::set_Target(const cv::Point2f Center_of_armor, int objectPoint)
{
	if(angle_solver_algorithm == 0 || angle_solver_algorithm == 2)
	{
		angle_solver_algorithm = 1; cout << "algorithm is reset to One Point Solution" << endl;
	}
	centerPoint = Center_of_armor;
	if(objectPoint == 3 || objectPoint == 4)
		is_shooting_rune = 1;
	else
	{
		is_shooting_rune = 0;
		rune_compensated_angle = 0;
	}
}

#ifdef DEBUG

void AngleSolver::showPoints2dOfArmor()
{
	cout << "the point 2D of armor is" << point_2d_of_armor << endl;
}


void AngleSolver::showTvec()
{
	cv::Mat tvect;
	transpose(t_Vec, tvect);
	cout << "the current t_Vec is:" << endl << tvect << endl;
}

void AngleSolver::showEDistance()
{
	cout << "  _euclideanDistance is  " << euclideanDistance / 1000 << "m" << endl;
}

void AngleSolver::showcenter_of_armor()
{
	cout << "the center of armor is" << centerPoint << endl;
}

void AngleSolver::showAngle()
{
	cout << "_xErr is  " << xErr << "  _yErr is  " << yErr << endl;
}

int AngleSolver::showAlgorithm()
{
	return angle_solver_algorithm;
}
#endif // DEBUG

AngleSolver::AngleFlag AngleSolver::solve()
{
	if(angle_solver_algorithm == 0)
	{
		if(enemy_type == 0)
			solvePnP(param.POINT_3D_OF_ARMOR_BIG, point_2d_of_armor, _cam_instant_matrix, param.DISTORTION_COEFF, r_Vec, t_Vec, false,SOLVEPNP_IPPE);
		if(enemy_type == 1)
			solvePnP(param.POINT_3D_OF_ARMOR_SMALL, point_2d_of_armor, _cam_instant_matrix, param.DISTORTION_COEFF, r_Vec, t_Vec, false,SOLVEPNP_IPPE);
		t_Vec.at<double>(1, 0) -= 120;
		euclideanDistance = sqrt(t_Vec.at<double>(0, 0)*t_Vec.at<double>(0, 0) + t_Vec.at<double>(1, 0)*t_Vec.at<double>(1, 0) + t_Vec.at<double>(2, 0)* t_Vec.at<double>(2, 0));
		xErr = atan(t_Vec.at<double>(0, 0) / t_Vec.at<double>(2, 0)) / CV_PI * 180;
		yErr = -atan(t_Vec.at<double>(1, 0) / sqrt(t_Vec.at<double>(2, 0)*t_Vec.at<double>(2,0) + t_Vec.at<double>(0,0)*t_Vec.at<double>(0,0)) )/ CV_PI * 180;
		yErr = yErr + euclideanDistance * 0.0006;
		if(euclideanDistance >= 8500)
		{
			return TOO_FAR;
		}
		return ANGLES_AND_DISTANCE;
	}
	if(angle_solver_algorithm == 1)
	{   
        double fx = _cam_instant_matrix.at<double>(0,0);
        double fy = _cam_instant_matrix.at<double>(1,1);
        double cx = _cam_instant_matrix.at<double>(0,2);
        double cy = _cam_instant_matrix.at<double>(1,2);
        cv::Point2f pnt;
        vector<cv::Point2f> in;
        vector<cv::Point2f> out;
        in.push_back(Point2f(centerPoint));
        
        undistortPoints(in,out,_cam_instant_matrix,param.DISTORTION_COEFF,noArray(),_cam_instant_matrix);
        pnt=out.front();

        double rx=(pnt.x-cx)/fx;
        double ry=(pnt.y-cy)/fy;
        double ry_new = ry - param.Y_DISTANCE_BETWEEN_GUN_AND_CAM / 1000;

        xErr = atan(rx) / CV_PI * 180;
	    yErr = -atan(ry_new) / CV_PI * 180;

		if(is_shooting_rune) yErr -= rune_compensated_angle;
		
		return ONLY_ANGLE;
	}
	if(angle_solver_algorithm == 2)
	{
		std::vector<Point2f> runeCenters;
		std::vector<Point3f> realCenters;
		for(size_t i = 0; i < 9; i++)
		{
			if(point_2d_of_rune[i].x > 0 && point_2d_of_rune[i].y > 0)
			{
				runeCenters.push_back(point_2d_of_rune[i]);
				realCenters.push_back(param.POINT_3D_OF_RUNE[i]);
			}
		}

		solvePnP(realCenters, runeCenters, _cam_instant_matrix, param.DISTORTION_COEFF, r_Vec, t_Vec, false);
		t_Vec.at<double>(1, 0) -= param.Y_DISTANCE_BETWEEN_GUN_AND_CAM;
		xErr = atan(t_Vec.at<double>(0, 0) / t_Vec.at<double>(2, 0)) / 2 / CV_PI * 360;
		yErr = atan(t_Vec.at<double>(1, 0) / t_Vec.at<double>(2, 0)) / 2 / CV_PI * 360;
		euclideanDistance = sqrt(t_Vec.at<double>(0, 0)*t_Vec.at<double>(0, 0) + t_Vec.at<double>(1, 0)*t_Vec.at<double>(1, 0) + t_Vec.at<double>(2, 0)* t_Vec.at<double>(2, 0));
		if(euclideanDistance >= 8500)
		{
			return TOO_FAR;
		}
		
		return ANGLES_AND_DISTANCE;
	}
	return ANGLE_ERROR;
}

void AngleSolver::compensateOffset()
{
	const auto offset_z = 115.0;
	const auto& d = euclideanDistance;
	const auto theta_y = xErr / 180 * CV_PI;
	const auto theta_p = yErr / 180 * CV_PI;
	const auto theta_y_prime = atan((d*sin(theta_y)) / (d*cos(theta_y) + offset_z));
	const auto theta_p_prime = atan((d*sin(theta_p)) / (d*cos(theta_p) + offset_z));
	const auto d_prime = sqrt(pow(offset_z + d * cos(theta_y), 2) + pow(d*sin(theta_y), 2));
	xErr = theta_y_prime / CV_PI * 180;
	yErr = theta_p_prime / CV_PI * 180;
	euclideanDistance = d_prime;
}

void AngleSolver::compensateGravity()
{
	const auto& theta_p_prime = yErr / 180 * CV_PI;
	const auto& d_prime = euclideanDistance;
	const auto& v = bullet_speed;
	const auto theta_p_prime2 = atan((sin(theta_p_prime) - 0.5*9.8*d_prime / pow(v, 2)) / cos(theta_p_prime));
	yErr = theta_p_prime2 / CV_PI * 180;
}


// void AngleSolver::set_Resolution(const cv::Size2i& image_resolution)
// {
// 	image_size = image_resolution;
	
// 	_cam_instant_matrix.at<double>(0, 2) = param.CAM_MATRIX.at<double>(0, 2) - (1920 / 2 - image_size.width / 2);
// 	_cam_instant_matrix.at<double>(1, 2) = param.CAM_MATRIX.at<double>(1, 2) - (1080 / 2 - image_size.height / 2);
// 	_cam_instant_matrix.at<double>(0, 0) = param.CAM_MATRIX.at<double>(0, 0) / (1080 / image_size.height);
// 	_cam_instant_matrix.at<double>(1, 1) = param.CAM_MATRIX.at<double>(1, 1) / (1080 / image_size.height);

// }

void AngleSolver::set_UserType(int usertype)
{
	user_type = usertype;
}

void AngleSolver::set_EnemyType(int enemytype)
{
	enemy_type = enemytype;
}

//const cv::Vec2f AngleSolver::getCompensateAngle()
//{
//	return cv::Vec2f(_xErr, _yErr);
//}

void AngleSolver::set_BulletSpeed(int bulletSpeed)
{
	bullet_speed = bulletSpeed;
}

const cv::Vec2f AngleSolver::get_Angle()
{
	return cv::Vec2f(xErr, yErr);
}


double AngleSolver::get_Distance()
{
	return euclideanDistance;
}

//void AngleSolver::selectAlgorithm(const int t)
//{
//	if (t == 0 || t == 1)
//		angle_solver_algorithm = t;
//
//}

void AngleSolverParam::readFile(const int id)
{
	cv::FileStorage fsread("/home/shaobing/rm_ws/src/anlesover/Pose/angle_solver_params.xml", cv::FileStorage::READ);
	if(!fsread.isOpened())
	{
		std::cerr << "failed to open xml" << std::endl;
		return;
	}
	fsread["Y_DISTANCE_BETWEEN_GUN_AND_CAM"] >> Y_DISTANCE_BETWEEN_GUN_AND_CAM;


	switch(id)
	{
	case 0://bubing
	{
		fsread["CAMERA_MARTRIX_0"] >> CAM_MATRIX;
		fsread["DISTORTION_COEFF_0"] >> DISTORTION_COEFF;
		return;
	}
	case 1://shaobing
	{
		fsread["CAMERA_MARTRIX_1"] >> CAM_MATRIX;
		fsread["DISTORTION_COEFF_1"] >> DISTORTION_COEFF;
		return;
	}
	case 2://yingxiong
	{
		fsread["CAMERA_MARTRIX_2"] >> CAM_MATRIX;
		fsread["DISTORTION_COEFF_2"] >> DISTORTION_COEFF;
		return;
	}
	default:
		std::cout << "wrong cam number given." << std::endl;
		return;
	}

}
}

class PointSubscriber : public rclcpp::Node
{
  public:
    PointSubscriber() 
    : Node("point_subscriber")
    {
      // 3-1.创建订阅方；
      subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "yolo_detections", 10, 
             std::bind(&PointSubscriber::topic_callback, this, std::placeholders::_1));

     publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("pitch_yaw", 20);
    }


private:
    void topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        std::vector<geometry_msgs::msg::Point> points;
        std::vector<cv::Point2f> cv_points;

        // 从消息中提取点坐标
        for (size_t i = 0; i < msg->data.size(); i += 3) {
            geometry_msgs::msg::Point point;
            point.x = msg->data[i];
            point.y = msg->data[i + 1];
            point.z = msg->data[i + 2];
            points.push_back(point);
            cv_points.push_back(cv::Point2f(point.x, point.y)); 
        }

		// 判断 cv_points 中的点是否全为 (0, 0)
		bool all_zeros = true;
		for (const auto& pt : cv_points) {
    		if (pt.x != 0.0f || pt.y != 0.0f) {
        	all_zeros = false;
        	break;
    		}
		}

		if (all_zeros) {

		rm::AngleSolverParam angleParam;
        angleParam.readFile(1);

        rm::AngleSolver angleSolver;
        angleSolver.init(angleParam);
        angleSolver.set_EnemyType(1);
    
		
        // 设置目标点并求解
        angleSolver.set_Target(cv_points, 1);
        angleSolver.solve();
        cv::Vec2f angle = angleSolver.get_Angle();
    		angle[0] = 0.0f;
    		angle[1] =  0.0f;
			// angleSolver.get_Distance() = 0.0f;
		RCLCPP_INFO(get_logger(), "nothing_here");
				// 创建一个新的消息并发布
		std_msgs::msg::Float64MultiArray new_msg;

		// unsigned char angle_true;

		new_msg.data.push_back(angle[0]);  // 距离值
		new_msg.data.push_back(angle[1]);  // 距离值
		new_msg.data.push_back(0.0);  // 帧头		
		// 发布消息
		publisher_->publish(new_msg);
   
			
		} else {
			       // 初始化角度求解器
        rm::AngleSolverParam angleParam;
        angleParam.readFile(1);
        rm::AngleSolver angleSolver;
        angleSolver.init(angleParam);
        angleSolver.set_EnemyType(1);
    
		
        // 设置目标点并求解
        angleSolver.set_Target(cv_points, 1);
        angleSolver.solve();
        cv::Vec2f angle = angleSolver.get_Angle();
		
		// 创建一个新的消息并发布
		std_msgs::msg::Float64MultiArray new_msg;

		// unsigned char angle_true;

		new_msg.data.push_back(angle[0]);  // 距离值
		new_msg.data.push_back(angle[1]);  // 距离值
		new_msg.data.push_back(angleSolver.get_Distance());  // 帧头		
		// 发布消息
		publisher_->publish(new_msg);

        RCLCPP_INFO(get_logger(), "Angle X: %f, Angle Y: %f", angle[0], angle[1]);
        RCLCPP_INFO(get_logger(), "Distance: %f", angleSolver.get_Distance());
        
    		
		}

 
    }

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
	rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
  // 2.初始化 ROS2 客户端；
  rclcpp::init(argc, argv);
  // 4.调用spin函数，并传入节点对象指针。
  rclcpp::spin(std::make_shared<PointSubscriber>());
  // 5.释放资源；
  rclcpp::shutdown();
  return 0;
}
