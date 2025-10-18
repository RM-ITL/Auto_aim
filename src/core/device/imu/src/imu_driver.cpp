#include "imu_driver.h"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace dmbot_serial
{
DmImu::DmImu() : Node("dm_imu_node")
{
  // 声明并获取参数
  this->declare_parameter("port", std::string("/dev/ttyACM0"));
  this->declare_parameter("baud", 921600);
  
  this->get_parameter("port", imu_serial_port_);
  this->get_parameter("baud", imu_seial_baud_);
                 
  imu_msgs_.header.frame_id = "imu_link";

  // 初始化数据为0
  data_.accx = data_.accy = data_.accz = 0.0;
  data_.gyrox = data_.gyroy = data_.gyroz = 0.0;
  data_.roll = data_.pitch = data_.yaw = 0.0;

  init_imu_serial();//初始化串口

  // 创建发布者
  imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("imu/data", 2);
  imu_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 100);
  euler2_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("dm/imu", 9);

  enter_setting_mode();
  rclcpp::sleep_for(std::chrono::milliseconds(10));

  turn_on_accel();
  rclcpp::sleep_for(std::chrono::milliseconds(10));

  turn_on_gyro();
  rclcpp::sleep_for(std::chrono::milliseconds(10));

  turn_on_euler();
  rclcpp::sleep_for(std::chrono::milliseconds(10));

  turn_off_quat();
  rclcpp::sleep_for(std::chrono::milliseconds(10));

  set_output_1000HZ();
  rclcpp::sleep_for(std::chrono::milliseconds(10));

  save_imu_para();
  rclcpp::sleep_for(std::chrono::milliseconds(10));
  
  exit_setting_mode();
  rclcpp::sleep_for(std::chrono::milliseconds(100));

  rec_thread_ = std::thread(&DmImu::get_imu_data_thread, this);

  RCLCPP_INFO(this->get_logger(), "imu init complete");
}

DmImu::~DmImu()
{ 
  RCLCPP_INFO(this->get_logger(), "enter ~DmImu()");
  
  stop_thread_ = true;
  
  if(rec_thread_.joinable())
  {
    rec_thread_.join(); 
  }

  if (serial_imu_.IsOpen())
  {
    serial_imu_.Close(); 
  } 
}

void DmImu::init_imu_serial()
{         
    try
    {
      // 打开串口
      serial_imu_.Open(imu_serial_port_);
      
      // 设置波特率
      LibSerial::BaudRate baud_rate;
      switch(imu_seial_baud_) {
        case 9600:
          baud_rate = LibSerial::BaudRate::BAUD_9600;
          break;
        case 19200:
          baud_rate = LibSerial::BaudRate::BAUD_19200;
          break;
        case 38400:
          baud_rate = LibSerial::BaudRate::BAUD_38400;
          break;
        case 57600:
          baud_rate = LibSerial::BaudRate::BAUD_57600;
          break;
        case 115200:
          baud_rate = LibSerial::BaudRate::BAUD_115200;
          break;
        case 230400:
          baud_rate = LibSerial::BaudRate::BAUD_230400;
          break;
        case 460800:
          baud_rate = LibSerial::BaudRate::BAUD_460800;
          break;
        case 921600:
          baud_rate = LibSerial::BaudRate::BAUD_921600;
          break;
        default:
          baud_rate = LibSerial::BaudRate::BAUD_921600;
          RCLCPP_WARN(this->get_logger(), "Unsupported baud rate %d, using 921600", imu_seial_baud_);
          break;
      }
      serial_imu_.SetBaudRate(baud_rate);
      
      // 设置字符大小
      serial_imu_.SetCharacterSize(LibSerial::CharacterSize::CHAR_SIZE_8);
      
      // 设置流控制
      serial_imu_.SetFlowControl(LibSerial::FlowControl::FLOW_CONTROL_NONE);
      
      // 设置奇偶校验
      serial_imu_.SetParity(LibSerial::Parity::PARITY_NONE);
      
      // 设置停止位
      serial_imu_.SetStopBits(LibSerial::StopBits::STOP_BITS_1);
      
      // 设置读取超时时间 - 使用 SetVTime (单位：0.1秒)
      serial_imu_.SetVTime(2); // 0.2秒超时
      serial_imu_.SetVMin(0);  // 立即返回
      
      RCLCPP_INFO_STREAM(this->get_logger(), "Serial port " << imu_serial_port_ << " opened successfully");
    } 
    catch (const LibSerial::OpenFailed& e)
    {
        RCLCPP_ERROR_STREAM(this->get_logger(), "Failed to open serial port: " << e.what());
        exit(0);
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR_STREAM(this->get_logger(), "Serial port exception: " << e.what());
        exit(0);
    }
    
    if (serial_imu_.IsOpen())
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "In initialization,Imu Serial Port initialized");
    }
    else
    {
        RCLCPP_ERROR_STREAM(this->get_logger(), "In initialization,Unable to open imu serial port ");
        exit(0);
    }
}

void DmImu::enter_setting_mode()
{
  uint8_t txbuf[4]={0xAA,0x06,0x01,0x0D};
  for(int i=0;i<5;i++)
  {
    try {
      LibSerial::DataBuffer buffer(txbuf, txbuf + sizeof(txbuf));
      serial_imu_.Write(buffer);
    } catch (const std::exception& e) {
      RCLCPP_ERROR_STREAM(this->get_logger(), "Write error in enter_setting_mode: " << e.what());
    }
    rclcpp::sleep_for(std::chrono::milliseconds(10));
  }
}

void DmImu::turn_on_accel()
{
  uint8_t txbuf[4]={0xAA,0x01,0x14,0x0D};
  for(int i=0;i<5;i++)
  {
    try {
      LibSerial::DataBuffer buffer(txbuf, txbuf + sizeof(txbuf));
      serial_imu_.Write(buffer);
    } catch (const std::exception& e) {
      RCLCPP_ERROR_STREAM(this->get_logger(), "Write error in turn_on_accel: " << e.what());
    }
    rclcpp::sleep_for(std::chrono::milliseconds(10));
  }
}

void DmImu::turn_on_gyro()
{
  uint8_t txbuf[4]={0xAA,0x01,0x15,0x0D};
  for(int i=0;i<5;i++)
  {
    try {
      LibSerial::DataBuffer buffer(txbuf, txbuf + sizeof(txbuf));
      serial_imu_.Write(buffer);
    } catch (const std::exception& e) {
      RCLCPP_ERROR_STREAM(this->get_logger(), "Write error in turn_on_gyro: " << e.what());
    }
    rclcpp::sleep_for(std::chrono::milliseconds(10));
  }
}

void DmImu::turn_on_euler()
{
  uint8_t txbuf[4]={0xAA,0x01,0x16,0x0D};
  for(int i=0;i<5;i++)
  {
    try {
      LibSerial::DataBuffer buffer(txbuf, txbuf + sizeof(txbuf));
      serial_imu_.Write(buffer);
    } catch (const std::exception& e) {
      RCLCPP_ERROR_STREAM(this->get_logger(), "Write error in turn_on_euler: " << e.what());
    }
    rclcpp::sleep_for(std::chrono::milliseconds(10));
  }
}

void DmImu::turn_off_quat()
{
  uint8_t txbuf[4]={0xAA,0x01,0x07,0x0D};
  for(int i=0;i<5;i++)
  {
    try {
      LibSerial::DataBuffer buffer(txbuf, txbuf + sizeof(txbuf));
      serial_imu_.Write(buffer);
    } catch (const std::exception& e) {
      RCLCPP_ERROR_STREAM(this->get_logger(), "Write error in turn_off_quat: " << e.what());
    }
    rclcpp::sleep_for(std::chrono::milliseconds(10));
  }
}

void DmImu::set_output_1000HZ()
{
  uint8_t txbuf[5]={0xAA,0x02,0x01,0x00,0x0D};
  for(int i=0;i<5;i++)
  {
    try {
      LibSerial::DataBuffer buffer(txbuf, txbuf + sizeof(txbuf));
      serial_imu_.Write(buffer);
    } catch (const std::exception& e) {
      RCLCPP_ERROR_STREAM(this->get_logger(), "Write error in set_output_1000HZ: " << e.what());
    }
    rclcpp::sleep_for(std::chrono::milliseconds(10));
  }
}

void DmImu::save_imu_para()
{
  uint8_t txbuf[4]={0xAA,0x03,0x01,0x0D};
  for(int i=0;i<5;i++)
  {
    try {
      LibSerial::DataBuffer buffer(txbuf, txbuf + sizeof(txbuf));
      serial_imu_.Write(buffer);
    } catch (const std::exception& e) {
      RCLCPP_ERROR_STREAM(this->get_logger(), "Write error in save_imu_para: " << e.what());
    }
    rclcpp::sleep_for(std::chrono::milliseconds(10));
  }
}

void DmImu::exit_setting_mode()
{
  uint8_t txbuf[4]={0xAA,0x06,0x00,0x0D};
  for(int i=0;i<5;i++)
  {
    try {
      LibSerial::DataBuffer buffer(txbuf, txbuf + sizeof(txbuf));
      serial_imu_.Write(buffer);
    } catch (const std::exception& e) {
      RCLCPP_ERROR_STREAM(this->get_logger(), "Write error in exit_setting_mode: " << e.what());
    }
    rclcpp::sleep_for(std::chrono::milliseconds(10));
  }
}

void DmImu::restart_imu()
{
  uint8_t txbuf[4]={0xAA,0x00,0x00,0x0D};
  for(int i=0;i<5;i++)
  {
    try {
      LibSerial::DataBuffer buffer(txbuf, txbuf + sizeof(txbuf));
      serial_imu_.Write(buffer);
    } catch (const std::exception& e) {
      RCLCPP_ERROR_STREAM(this->get_logger(), "Write error in restart_imu: " << e.what());
    }
    rclcpp::sleep_for(std::chrono::milliseconds(10));
  }
}

void DmImu::publish_imu_data()
{
  imu_msgs_.header.stamp = this->now();

  // 使用tf2创建四元数
  tf2::Quaternion q;
  q.setRPY(data_.roll * M_PI / 180.0, data_.pitch * M_PI / 180.0, data_.yaw * M_PI / 180.0);
  imu_msgs_.orientation = tf2::toMsg(q);

  imu_msgs_.angular_velocity.x = data_.gyrox;
  imu_msgs_.angular_velocity.y = data_.gyroy;
  imu_msgs_.angular_velocity.z = data_.gyroz;

  imu_msgs_.linear_acceleration.x = data_.accx;
  imu_msgs_.linear_acceleration.y = data_.accy;
  imu_msgs_.linear_acceleration.z = data_.accz;

  imu_pub_->publish(imu_msgs_);

  // 发布pose消息
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "imu_link";
  pose.header.stamp = imu_msgs_.header.stamp;
  pose.pose.position.x = 0.0;
  pose.pose.position.y = 0.0;
  pose.pose.position.z = 0.0;
  pose.pose.orientation.w = imu_msgs_.orientation.w;
  pose.pose.orientation.x = imu_msgs_.orientation.x;
  pose.pose.orientation.y = imu_msgs_.orientation.y;
  pose.pose.orientation.z = imu_msgs_.orientation.z;
  
  imu_pose_pub_->publish(pose);

  // 发布euler数据
  std_msgs::msg::Float64MultiArray euler2_msg;
  euler2_msg.data.resize(9);
  euler2_msg.data[0] = data_.gyrox;
  euler2_msg.data[1] = data_.gyroy;
  euler2_msg.data[2] = data_.gyroz;
  euler2_msg.data[3] = data_.accx;
  euler2_msg.data[4] = data_.accy;
  euler2_msg.data[5] = data_.accz;
  euler2_msg.data[6] = data_.roll * M_PI / 180.0;
  euler2_msg.data[7] = data_.pitch * M_PI / 180.0;
  euler2_msg.data[8] = data_.yaw * M_PI / 180.0;
  euler2_pub_->publish(euler2_msg);
}

void DmImu::get_imu_data_thread()
{ 
  int error_num = 0;
  int packet_count = 0;
  int published_count = 0;
  LibSerial::DataBuffer buffer;
  
  RCLCPP_INFO(this->get_logger(), "IMU data thread started");
  
  while (rclcpp::ok() && !stop_thread_)
  {    
    if (!serial_imu_.IsOpen())
    {
      RCLCPP_WARN(this->get_logger(), "In get_imu_data_thread,imu serial port unopen");
      continue;
    }       

    try {
      // 读取单个数据包（16字节）
      buffer.clear();
      buffer.resize(16);
      
      serial_imu_.Read(buffer, 16);
      
      if (buffer.size() >= 4) {
        uint8_t header1 = buffer[0];
        uint8_t header2 = buffer[1]; 
        uint8_t slave_id = buffer[2];
        uint8_t reg_type = buffer[3];
        
        // 检查帧头
        if(header1 == 0x55 && header2 == 0xAA && slave_id == 0x01)
        {
          packet_count++;
          
          // 根据不同的寄存器类型处理数据
          if(reg_type == 0x01 && buffer.size() >= 16) // 加速度计数据
          {
            // 从字节4开始解析，假设数据为小端格式
            memcpy(&data_.accx, &buffer[4], 4);
            memcpy(&data_.accy, &buffer[8], 4);
            memcpy(&data_.accz, &buffer[12], 4);
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "ACC: x=%.3f, y=%.3f, z=%.3f", data_.accx, data_.accy, data_.accz);
            
            // 每次更新数据都发布
            publish_imu_data();
            published_count++;
          }
          else if(reg_type == 0x02 && buffer.size() >= 16) // 陀螺仪数据
          {
            memcpy(&data_.gyrox, &buffer[4], 4);
            memcpy(&data_.gyroy, &buffer[8], 4);
            memcpy(&data_.gyroz, &buffer[12], 4);
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "GYRO: x=%.3f, y=%.3f, z=%.3f", data_.gyrox, data_.gyroy, data_.gyroz);
            
            // 每次更新数据都发布
            publish_imu_data();
            published_count++;
          }
          else if(reg_type == 0x03 && buffer.size() >= 16) // 欧拉角数据
          {
            memcpy(&data_.roll, &buffer[4], 4);
            memcpy(&data_.pitch, &buffer[8], 4);
            memcpy(&data_.yaw, &buffer[12], 4);
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "EULER: roll=%.3f, pitch=%.3f, yaw=%.3f", data_.roll, data_.pitch, data_.yaw);
            
            // 每次更新数据都发布
            publish_imu_data();
            published_count++;
          }
          
          // 每1000个包输出一次统计信息
          if(packet_count % 1000 == 0)
          {
            RCLCPP_INFO(this->get_logger(), "Processed %d packets, Published %d IMU messages", 
                       packet_count, published_count);
          }
          
          error_num = 0;
        }
        else
        { 
          error_num++;
          if(error_num > 1200)
          {
            RCLCPP_WARN(this->get_logger(), "Header mismatch: 0x%02X 0x%02X 0x%02X 0x%02X (expected: 0x55 0xAA 0x01 0x01/02/03)", 
                       header1, header2, slave_id, reg_type);
            error_num = 0;
          }
        }
      }
      else
      {
        // 数据包太短
        error_num++;
        if(error_num > 1200)
        {
          RCLCPP_WARN(this->get_logger(), "Received packet too short: %zu bytes", buffer.size());
          error_num = 0;
        }
      }
    }
    catch (const LibSerial::ReadTimeout& e) {
      // 读取超时是正常的，继续下一次循环
      continue;
    }
    catch (const std::exception& e) {
      RCLCPP_ERROR_STREAM(this->get_logger(), "Serial read error: " << e.what());
      // 短暂等待后重试
      rclcpp::sleep_for(std::chrono::milliseconds(10));
    }
  }
  
  RCLCPP_INFO(this->get_logger(), "IMU data thread stopped");
}

}