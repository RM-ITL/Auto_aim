#pragma once
#include<iostream>
#include"opencv2/core/core.hpp"
#include<opencv2/opencv.hpp>

namespace rm
{
struct AngleSolverParam
{
    /* data */ 
    cv::Mat CAM_MATRIX;
    cv::Mat DISTORTION_COEFF;
	
	static std::vector<cv::Point3f> POINT_3D_OF_ARMOR_BIG;
	static std::vector<cv::Point3f> POINT_3D_OF_ARMOR_SMALL;
	static std::vector<cv::Point3f> POINT_3D_OF_RUNE;
    
    double Y_DISTANCE_BETWEEN_GUN_AND_CAM = 48;
    cv::Size CAM_SIZE = cv::Size(1280,1024);

    void readFile(const int id);
};
class AngleSolver
{
public:
AngleSolver();
void init(const AngleSolverParam& AngleSolverParam);

enum AngleFlag
{
    ANGLE_ERROR = 0,                
    ANGLES_AND_DISTANCE = 1,		
    ONLY_ANGLE = 2,			
    TOO_FAR = 3		
}; 

void set_Target(const std::vector<cv::Point2f> objectPoints, int objectType);  //set corner points for PNP
void set_Target(const cv::Point2f centerPoint, int objectType);

AngleFlag solve();

void compensateOffset();
void compensateGravity();

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
AngleSolverParam param;
cv::Mat r_Vec = cv::Mat::zeros(3, 1, CV_64FC1);
cv::Mat t_Vec = cv::Mat::zeros(3, 1, CV_64FC1);
std::vector<cv::Point2f> point_2d_of_armor;
std::vector<cv::Point2f> point_2d_of_rune;
enum solver_way
{
    PNP4 = 0,
    PinHole = 1
};
int angle_solver_algorithm = 0;//0 PNP 1PINHOLE

cv::Point2f centerPoint;
std::vector<cv::Point2f> target_nothing;

double xErr, yErr, euclideanDistance;

cv::Size2i image_size = cv::Size2i(1280,1024);

int user_type = 1;//0
int enemy_type = 1;//0 BIG 1 SMALL
double bullet_speed = 22000;
double rune_compensated_angle = 0;

cv::Mat _cam_instant_matrix;

int is_shooting_rune = 0;
};
}