#ifndef ARMOR_DETECOR
#define ARMOR_DETECOR

#include<opencv4/opencv2/opencv.hpp>
#include<array>
#include<opencv4/opencv2/ml.hpp>


namespace rm
{
enum ColorChannels
{
	BLUE = 0,
	RED = 1
};
enum ARMORTYPE
{	
	UNKNOWN_ARMOR = 0,
	SMALL_ARMOR = 1,
	BIG_ARMOR = 2,
	MIN_RUNE = 3
};
struct ArmorParam
{
    int brightness_threshold;
    int color_threshold;
    float light_color_detect_extend_ratio;

    float light_min_area;
    float light_max_area;
    float light_min_size;
    float light_max_angle;
    float light_contour_min_solidity;
    float light_max_ratio;

	float light_max_angle_diff_;
	float light_max_height_diff_ratio_;
	float light_max_y_diff_ratio_;
	float light_min_x_diff_ratio_;

    float armor_big_armor_ratio;
    float armor_small_armor_ratio;
	float armor_min_aspect_ratio_;
	float armor_max_aspect_ratio_;

    int enemy_color;

    ArmorParam()
    {
        brightness_threshold = 100;
        color_threshold = 40;
        light_color_detect_extend_ratio = 1.1;

        light_min_area = 10;
        light_max_angle = 45.0;
        light_min_size = 5.0;
        light_contour_min_solidity = 0.5;
        light_max_ratio = 1.0;

		light_max_angle_diff_=7.0;
		light_max_height_diff_ratio_=0.2;
		light_max_y_diff_ratio_=2.0;
		light_min_x_diff_ratio_=0.5;

        armor_big_armor_ratio = 3.2;
        armor_small_armor_ratio = 2;
		armor_min_aspect_ratio_=1.0;
		armor_max_aspect_ratio_=5.0;

        enemy_color = RED;
    }
};


class LightDescription
{
public:
    LightDescription(){}
    LightDescription(const cv::RotatedRect& light)
    {
        width = light.size.width;
        length = light.size.height;
        center = light.center;
        angle = light.angle;
        area = light.size.area();
		light.points(points);

	}
    //const LightDescription& operator =(const LightDescription& ld)//copy
	//{
	//	this->width = ld.width;
	//	this->length = ld.length;
	//	this->center = ld.center;
	//	this->angle = ld.angle;
	//	this->area = ld.area;
	//	return *this;
	//}

	
	cv::RotatedRect rec() const
	{
		return cv::RotatedRect(center, cv::Size2f(width, length), angle);
	}
	
public:
	float width;
	float length;
	cv::Point2f center;
	float angle;
	float area;
	cv::Point2f points[4];
};


class ArmorDescription
{
public:
	
	ArmorDescription();

	
	ArmorDescription( const LightDescription& l_Light, const LightDescription& r_Light, const int armorType, const cv::Mat& srcImg,ArmorParam param);
	~ArmorDescription(){}
	
	void clear()
	{
	//	rotationScore = 0;
	//	sizeScore = 0;
		distScore = 0;
	//	finalScore = 0;
		for(int i = 0; i < 4; i++)
		{
			vertex[i] = cv::Point2f(0, 0);
		}
		type = UNKNOWN_ARMOR;
	}

	
	//void getFrontImg(const cv::Mat& grayImg);
	//bool isArmorPattern() const;

	std::array<cv::RotatedRect, 2> lightPairs; //0 left, 1 right
	//float sizeScore;		//S1 = e^(size)
	float distScore;		//S2 = e^(-offset)
	//float rotationScore;		//S3 = -(ratio^2 + yDiff^2) 
	//float finalScore;		
	
	std::vector<cv::Point2f> vertex; //four vertex of armor area, lihgt bar area exclued!!	
    //cv::Mat frontImg; //front img after prespective transformation from vertex,1 channel gray img
	cv::Point2f center;
	//	0 -> small
	//	1 -> big
	//	-1 -> unkown
	int type;
};


class ArmorDetector
{
public:
	
	enum ArmorFlag
	{
		ARMOR_NO = 0,		// not found
		ARMOR_LOST = 1,		// lose tracking
		ARMOR_GLOBAL = 2,	// armor found globally
		ARMOR_LOCAL = 3		// armor found locally(in tracking mode)
	};

    ArmorDetector();
	ArmorDetector(const ArmorParam& armorParam);
    ~ArmorDetector(){}

	void init(const ArmorParam& armorParam);

	void setEnemyColor(ColorChannels enemy_color)
	{
		enemy_color = enemy_color;
		self_color = enemy_color == BLUE ? RED : BLUE;
	}

	//void loadImg(const cv::Mat&  srcImg);

	int detect();

	const std::vector<cv::Point2f> getArmorVertex() const;

    int getArmorType() const;


private:
	ArmorParam param;
	int enemy_color;
	int self_color;

	//cv::Rect roi; //relative coordinates

	cv::Mat srcImg; //source img
	//cv::Mat roiImg; //roi from the result of last frame
	cv::Mat grayImg; //gray img of roi

	//int trackCnt = 0;
	
	std::vector<ArmorDescription> armors;

	ArmorDescription targetArmor; //relative coordinates

	int flag;
	//bool isTracking;
	
};

}

#endif