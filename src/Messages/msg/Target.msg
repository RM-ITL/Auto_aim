# Target.msg - 单个目标信息
# 该消息描述了一个检测到的装甲板目标的完整信息

# 消息头，包含时间戳和坐标系信息
std_msgs/Header header

# 目标标识
string id                    # 装甲板编号（如"1", "2", "3"等）
string armor_type           # 装甲板类型（"small" 或 "big"）

geometry_msgs/Point32[4] corners 

# 3D位置信息（世界坐标系，单位：米）
geometry_msgs/Point position
geometry_msgs/Point position_camera

# 角度信息（相对于初始IMU状态）
float64 yaw                 # 水平角度（弧度）
float64 pitch               # 俯仰角度（弧度）
float64 roll

float64 yaw_self            # 自身yaw

# 距离信息
float64 distance            # 目标距离（米）