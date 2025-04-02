from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

# 获取参数文件路径
node_params = os.path.join(
    get_package_share_directory('anlesover'),
    'config',
    'angle_solver_params.xml'
)

# 总launch启动
def generate_launch_description():
    return LaunchDescription([
         
        # 启动检测追踪节点
        Node(
            package='pkg02_helloworld_py',
            executable='camera_node',		# 调试使用camera,上场使用camera_node
            namespace='Vision',
            # parameters=[node_params],  # 如果需要参数文件，取消注释
            output='both',
        ),
        
        # 启动检测追踪节点
        Node(
            package='pkg02_helloworld_py',
            executable='sub_openvino_blue',		# sub_openvino_UKF追踪节点
            namespace='Vision',
            # parameters=[node_params],  # 如果需要参数文件，取消注释
            output='both',
        ),

        # 启动角度结算节点
        Node(
            package='anlesover',
            executable='anglesolve',
            namespace='Vision',
            parameters=[node_params],
            output='both',
        ),


    ])
