#pragma once
#include "opencv2/core/core.hpp"
#include<iostream>
#include <opencv2/opencv.hpp>
#define DEBUG

namespace rm 
{

struct AngleSolverParam
{
    cv::Mat CAM_MATRIX;
    cv::Mat DISTORTION_COEFF;
	
	static std::vector<cv::Point3f> POINT_3D_OF_ARMOR_BIG;
	static std::vector<cv::Point3f> POINT_3D_OF_ARMOR_SMALL;
	static std::vector<cv::Point3f> POINT_3D_OF_RUNE;
    double Y_DISTANCE_BETWEEN_GUN_AND_CAM = 0;
    cv::Size CAM_SIZE = cv::Size(1280, 1024);

    void readFile(const int id);
};
class AngleSolver
{
public:
    AngleSolver();
    AngleSolver(const AngleSolverParam& AngleSolverParam);

    
    void init(const AngleSolverParam& AngleSolverParam);

	enum AngleFlag
	{
		ANGLE_ERROR = 0,                
		ONLY_ANGLES = 1,		
		TOO_FAR = 2,			
		ANGLES_AND_DISTANCE = 3		
	}; 


	void setTarget(const std::vector<cv::Point2f> objectPoints, int objectType);  //set corner points for PNP
	void setTarget(const cv::Point2f centerPoint, int objectType);//set center points
	
	AngleFlag solve();

	
	void compensateOffset();
	void compensateGravity();

	
	void set_Resolution(const cv::Size2i& image_resolution);

	
	void set_UserType(int usertype);

	void set_EnemyType(int enemytype);

	void set_BulletSpeed(int bulletSpeed);

	const cv::Vec2f get_Angle();

	
    double get_Distance();
#ifdef DEBUG
	/*
	* @brief show 2d points of armor 
	*/
	void showPoints2dOfArmor();

	/*
	* @brief show tvec 
	*/
	void showTvec();

	/*
	* @brief show distance
	*/
	void showEDistance();

	/*
	* @brief show center of armor
	*/
	void showcenter_of_armor();

	/*
	* @brief show angles
	*/
	void showAngle();

	/*
	* @brief show the information of selected algorithm
	*/
	int showAlgorithm();
#endif // DEBUG


private:
	AngleSolverParam _params;
	cv::Mat _rVec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat _tVec = cv::Mat::zeros(3, 1, CV_64FC1);
	std::vector<cv::Point2f> point_2d_of_armor;
	std::vector<cv::Point2f> point_2d_of_rune;
	enum solver_way
	{
		PNP4 = 0,
		PinHole = 1
	};
	int angle_solver_algorithm = 0;
	cv::Point2f centerPoint;
	std::vector<cv::Point2f> target_nothing;
	double xErr, yErr, euclideanDistance;
	cv::Size2i image_size = cv::Size2i(1280, 1024);
	int user_type = 1;
	int enemy_type = 1;
	double bullet_speed = 22000;
	double rune_compensated_angle = 0;
	int is_shooting_rune = 0;
	cv::Mat _cam_instant_matrix;
};
}    
