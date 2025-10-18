# 单个装甲板的预测结果
int32 armor_id                    # 装甲板ID
string armor_type                 # 装甲板类型名称

geometry_msgs/Point position      # 预测位置（米）
geometry_msgs/Vector3 velocity    # 预测速度（米/秒）

float64 yaw                      # 预测的yaw角（弧度）
float64 pitch                    # 预测的pitch角（弧度）
float64 distance                 # 预测的距离（米）

# 朝向角预测相关（新增）
bool has_orientation_prediction   # 是否包含有效的朝向角预测
float64 orientation              # 预测的朝向角（弧度），仅在has_orientation_prediction为true时有效
float64 angular_velocity         # 预测的角速度（弧度/秒），仅在has_orientation_prediction为true时有效


# 预测质量指标（新增）
float64 position_uncertainty     # 位置预测的不确定性（米）
float64 orientation_uncertainty  # 朝向角预测的不确定性（弧度）
