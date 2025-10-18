from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # 声明启动参数
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='/dev/ttyACM0',  # 匹配原始默认值
        description='IMU串口设备路径'
    )
    
    baud_arg = DeclareLaunchArgument(
        'baud',
        default_value='921600',
        description='IMU串口波特率'
    )
    
    # IMU节点 (使用与ROS1一致的节点名)
    imu_node = Node(
        package='dm_imu',
        executable='dm_imu_node',  # 匹配原始可执行文件名
        name='dm_imu_node',
        parameters=[{
            'port': LaunchConfiguration('port'),
            'baud': LaunchConfiguration('baud')
        }],
        output='screen',
        emulate_tty=True
    )
    
    return LaunchDescription([
        port_arg,
        baud_arg,
        imu_node
    ])
