from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # 声明启动参数
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='/dev/ttyACM0',
        description='IMU串口设备路径'
    )
    
    baud_arg = DeclareLaunchArgument(
        'baud',
        default_value='921600',
        description='IMU串口波特率'
    )
    
    # RViz配置文件路径
    rviz_config_path = PathJoinSubstitution([
        FindPackageShare('dm_imu'),
        'rviz',
        'imu.rviz'
    ])
    
    # IMU节点
    imu_node = Node(
        package='dm_imu',
        executable='dm_imu_node',
        name='dm_imu_node',
        parameters=[{
            'port': LaunchConfiguration('port'),
            'baud': LaunchConfiguration('baud')
        }],
        output='screen',
        emulate_tty=True
    )
    
    # RViz节点
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )
    
    return LaunchDescription([
        port_arg,
        baud_arg,
        imu_node,
        rviz_node
    ])
