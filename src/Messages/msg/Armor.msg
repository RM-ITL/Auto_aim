std_msgs/Header header
geometry_msgs/Point32[4] corners  # 0:左上，1：右上, 2：右下，3:左下
geometry_msgs/Point32[4] light_corners
string color
string type
int32 number               # 分类数字 
geometry_msgs/Point center        # 添加中心点，使用x,y坐标，z=0
int32 rank
float32 confidence         # yolo置信度
float32 conf_color         # 置信度得分
float32 conf_number        # 数字分类置信度
