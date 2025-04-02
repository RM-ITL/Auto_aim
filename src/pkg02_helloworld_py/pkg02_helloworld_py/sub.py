import serial
import struct
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

def crc16_modbus(data: bytes) -> int:
    """计算CRC-16/MODBUS校验值（与发送端一致）"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc

class SerialReaderNode(Node):
    def __init__(self):
        super().__init__('serial_reader')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'sensor_data', 10)
        self.ser = serial.Serial(
            port='/dev/ttyUSB0',  # 根据实际情况修改
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        self.buffer = bytearray()  # 数据缓冲区

    def read_and_publish(self):
        while True:
            # 读取所有可用数据（非阻塞读取）
            data = self.ser.read(self.ser.in_waiting or 1)
            self.buffer += data

            while True:
                # 在缓冲区中查找包头0xFF
                start_pos = -1
                for i in range(len(self.buffer)):
                    if self.buffer[i] == 0xFF:
                        start_pos = i
                        break

                # 未找到包头
                if start_pos == -1:
                    self.buffer.clear()
                    break










                # 检查数据包完整性
                if len(self.buffer) < start_pos + 28:
                    self.buffer = self.buffer[start_pos:]  # 保留可能的数据包头
                    break

                # 提取候选数据包
                candidate_packet = self.buffer[start_pos:start_pos+28]

                # 校验CRC
                crc_received = struct.unpack('<H', candidate_packet[-2:])[0]
                crc_calculated = crc16_modbus(candidate_packet[:26])

                if crc_calculated == crc_received:
                    # 解析数据包头
                    header, bitfield = struct.unpack_from('>BB', candidate_packet, 0)

                    # 解析位域
                    detect_color = bitfield & 0x01
                    reset_tracker = (bitfield >> 1) & 0x01
                    reserved = (bitfield >> 2) & 0x3F

                    # 解析浮点数据
                    floats = struct.unpack_from('>6f', candidate_packet, 2)
                    roll, pitch, yaw, aim_x, aim_y, aim_z = floats

                    # 打包数据并发布
                    msg = Float32MultiArray()
                    msg.data = [roll, pitch, yaw, aim_x, aim_y, aim_z]
                    self.publisher_.publish(msg)

                    # 打印解析结果
                    self.get_logger().info(f"发布数据包: {msg.data}")

                    # 移除已处理数据
                    self.buffer = self.buffer[start_pos+28:]
                else:
                    # CRC校验失败，跳过当前包头
                    self.buffer = self.buffer[start_pos+1:]

                # 继续处理剩余数据
                continue

def main(args=None):
    rclpy.init(args=args)
    serial_reader_node = SerialReaderNode()

    try:
        serial_reader_node.read_and_publish()
    except KeyboardInterrupt:
        pass
    finally:
        serial_reader_node.ser.close()
        serial_reader_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
