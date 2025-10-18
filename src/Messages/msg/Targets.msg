# Targets.msg - 多个目标信息的集合
# 该消息包含了一帧中检测到的所有装甲板目标

# 消息头，包含时间戳和坐标系信息
std_msgs/Header header

# 目标数组
Target[] targets