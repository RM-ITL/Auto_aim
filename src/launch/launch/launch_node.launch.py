from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

# 总launch启动
def generate_launch_description():
    # 获取角度解算参数文件路径
    angle_solver_params = os.path.join(
        get_package_share_directory('anlesover'),
        'config',
        'angle_solver_params.xml'
    )
    
    # 获取视觉系统YAML配置文件路径
    yaml_config_path = os.path.join(
        get_package_share_directory('detected'),
        'config',
        'robomaster_vision_config.yaml'
    )
    
    # 声明启动参数 - 是否使用YAML配置
    declare_use_yaml_config = DeclareLaunchArgument(
        'use_yaml_config',
        default_value='true',
        description='是否使用YAML配置文件'
    )
    
    # 获取启动参数
    use_yaml_config = LaunchConfiguration('use_yaml_config')
    
    return LaunchDescription([
        # 启动参数声明
        declare_use_yaml_config,
         
        # 启动相机节点
        Node(
            package='detected',
            executable='camera_catch_node',  # 调试使用camera,上场使用camera_node
            namespace='Vision',
            parameters=[{'config_file': yaml_config_path}, {'use_yaml_config': use_yaml_config}],
            output='both',
        ),
        
        # 启动检测追踪节点
        Node(
            package='detected',
            executable='armor_detector_node',  # sub_openvino_UKF追踪节点
            namespace='Vision',
            parameters=[{'config_file': yaml_config_path}, {'use_yaml_config': use_yaml_config}],
            output='both',
        ),

        # 启动角度结算节点
        Node(
            package='anlesover',
            executable='anglesolve',
            namespace='Vision',
            parameters=[angle_solver_params],  # 保持原有XML参数
            output='both',
        ),
    ])