# 时间
std_msgs/Header header

# 新增字段
int32 target_id          # 目标ID（1英雄、3/4步兵、-1哨兵等）
int32 armor_count        # 检测到的装甲板数量
float32 roi_coefficient    # roi转换系数

# 装甲板信息
Armor[] armor 