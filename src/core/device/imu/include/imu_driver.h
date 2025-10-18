#ifndef IMU_DRIVER_H
#define IMU_DRIVER_H

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <libserial/SerialPort.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <thread>
#include <atomic>

namespace dmbot_serial
{

// IMU数据结构
struct ImuData
{
  float accx, accy, accz;    // 加速度计数据
  float gyrox, gyroy, gyroz; // 陀螺仪数据
  float roll, pitch, yaw;    // 欧拉角数据
};

// 原始接收数据结构（如果需要CRC校验可以保留）
struct ReceiveData
{
  uint8_t FrameHeader1;
  uint8_t flag1;
  uint8_t slave_id1;
  uint8_t reg_acc;
  uint32_t accx_u32;
  uint32_t accy_u32;
  uint32_t accz_u32;
  uint16_t crc1;
  
  uint8_t FrameHeader2;
  uint8_t flag2;
  uint8_t slave_id2;
  uint8_t reg_gyro;
  uint32_t gyrox_u32;
  uint32_t gyroy_u32;
  uint32_t gyroz_u32;
  uint16_t crc2;
  
  uint8_t FrameHeader3;
  uint8_t flag3;
  uint8_t slave_id3;
  uint8_t reg_euler;
  uint32_t roll_u32;
  uint32_t pitch_u32;
  uint32_t yaw_u32;
  uint16_t crc3;
};

class DmImu : public rclcpp::Node
{
public:
  DmImu();
  ~DmImu();

private:
  // 初始化函数
  void init_imu_serial();
  
  // IMU配置函数
  void enter_setting_mode();
  void turn_on_accel();
  void turn_on_gyro();
  void turn_on_euler();
  void turn_off_quat();
  void set_output_1000HZ();
  void save_imu_para();
  void exit_setting_mode();
  void restart_imu();
  
  // 数据处理函数
  void get_imu_data_thread();
  void publish_imu_data();  // 新添加的发布函数
  
  // CRC校验函数（如果需要）
  uint16_t Get_CRC16(uint8_t* data, int len)
  {
    uint16_t crc = 0xFFFF;
    for(int i = 0; i < len; i++)
    {
      crc ^= data[i];
      for(int j = 0; j < 8; j++)
      {
        if(crc & 0x01)
          crc = (crc >> 1) ^ 0xA001;
        else
          crc = crc >> 1;
      }
    }
    return crc;
  }

private:
  // 串口相关
  LibSerial::SerialPort serial_imu_;
  std::string imu_serial_port_;
  int imu_seial_baud_;
  
  // 线程相关
  std::thread rec_thread_;
  std::atomic<bool> stop_thread_{false};
  
  // ROS发布者
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr imu_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr euler2_pub_;
  
  // 消息和数据
  sensor_msgs::msg::Imu imu_msgs_;
  ImuData data_;
  ReceiveData receive_data_;
};

}

#endif // IMU_DRIVER_H