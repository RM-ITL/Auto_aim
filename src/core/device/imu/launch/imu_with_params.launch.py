from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 参数文件路径
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('dm_imu'),
            'config',
            'imu_params.yaml'
        ]),
        description='参数配置文件路径'
    )
    
    # IMU节点
    imu_node = Node(
        package='dm_imu',
        executable='dm_imu_node',
        name='dm_imu_node',
        parameters=[LaunchConfiguration('params_file')],
        output='screen',
        emulate_tty=True
    )
    
    return LaunchDescription([
        params_file_arg,
        imu_node
    ])
