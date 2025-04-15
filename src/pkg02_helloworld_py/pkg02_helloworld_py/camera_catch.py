#! /usr/bin/env python3

# -- coding: utf-8 --
import numpy as np
import sys
import threading
import os
import cv2
import cv_bridge
import termios
import time
import ctypes
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ctypes import *
import concurrent.futures
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

sys.path.append("/home/guo/ITL_sentry_auto/src/pkg02_helloworld_py/MvImport")
from MvCameraControl_class import *

g_bExit = False
DEBUG = False  # 全局调试标志 - 设置为False以禁用调试输出

# 定义一个用于封装相机状态和线程管理的类
class CameraManager:
    def __init__(self, cam, data_buf, nPayloadSize, node):
        self.cam = cam
        self.data_buf = data_buf
        self.nPayloadSize = nPayloadSize
        self.node = node
        self.bridge = cv_bridge.CvBridge()
        
        # 创建一个优化的QoS配置，用于零拷贝通信
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 创建发布者
        self.publisher = node.create_publisher(Image, 'image_topic', qos)
        
        # 创建线程池用于异步处理
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
    def start(self):
        """启动相机采集线程"""
        self.capture_thread = threading.Thread(target=self.capture_thread_func)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
    def capture_thread_func(self):
        """相机图像采集线程函数"""
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        
        # 加载动态链接库
        libc = ctypes.CDLL('libc.so.6')
        
        img_buff = None
        
        while not g_bExit:
            ret = self.cam.MV_CC_GetOneFrameTimeout(self.data_buf, self.nPayloadSize, stFrameInfo, 1000)
            if ret == 0:
                if DEBUG:
                    self.node.get_logger().info(f"获取帧: 宽[{stFrameInfo.nWidth}], 高[{stFrameInfo.nHeight}], 帧号[{stFrameInfo.nFrameNum}]")
                
                # 获取一份数据副本，避免在异步处理时数据被覆盖
                frame_data = (c_ubyte * stFrameInfo.nFrameLen)()
                ctypes.memmove(frame_data, self.data_buf, stFrameInfo.nFrameLen)
                
                # 创建一个帧信息的副本
                frame_info = MV_FRAME_OUT_INFO_EX()
                ctypes.memmove(byref(frame_info), byref(stFrameInfo), sizeof(stFrameInfo))
                
                # 提交图像处理任务到线程池
                self.thread_pool.submit(self.process_image, frame_data, frame_info)
            else:
                if DEBUG:
                    self.node.get_logger().warn(f"无数据[0x{ret:x}]")
                    
            # 避免CPU占用过高
            time.sleep(0.001)
            
    def process_image(self, pData, stFrameInfo):
        """异步处理图像并发布"""
        try:
            time_start = time.time()
            
            # 设置转换参数
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            
            is_color = IsImageColor(stFrameInfo.enPixelType)
            
            # 计算转换后的缓冲区大小，确保足够大
            if is_color == 'mono':
                stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
                # 为缓冲区添加安全边界，确保空间充足
                nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight * 2
                encoding = "mono8"
            elif is_color == 'color':
                stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed
                # BGR8需要3倍空间，添加安全边界
                nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight * 4
                encoding = "bgr8"
            else:
                self.node.get_logger().error("不支持的图像格式!")
                return
                
            # 设置转换参数
            stConvertParam.nWidth = stFrameInfo.nWidth
            stConvertParam.nHeight = stFrameInfo.nHeight
            stConvertParam.pSrcData = cast(pData, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
            
            # 分配足够大的目标缓冲区
            dst_buffer = (c_ubyte * nConvertSize)()
            stConvertParam.pDstBuffer = cast(dst_buffer, POINTER(c_ubyte))
            stConvertParam.nDstBufferSize = nConvertSize
            
            # 执行像素格式转换
            ret = self.cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                self.node.get_logger().error(f"转换像素格式失败! ret[0x{ret:x}]")
                return
                
            # 检查转换后的数据大小是否在预期范围内
            if stConvertParam.nDstLen > nConvertSize:
                self.node.get_logger().error(f"转换后数据大小超出缓冲区容量: {stConvertParam.nDstLen} > {nConvertSize}")
                return
                
            # 将数据转换为numpy数组并reshape
            img_data = np.frombuffer(dst_buffer, count=int(stConvertParam.nDstLen), dtype=np.uint8)
            
            if is_color == 'mono':
                img_data = img_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
            else:
                img_data = img_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, 3))
                
            # 调整图像大小 - 保持与原代码相同的尺寸
            if is_color == 'mono':
                img_data = cv2.resize(img_data, (1200, 1024), interpolation=cv2.INTER_AREA)
            else:
                img_data = cv2.resize(img_data, (1280, 1024), interpolation=cv2.INTER_AREA)
                
            # 创建ROS图像消息并发布
            msg = self.bridge.cv2_to_imgmsg(img_data, encoding=encoding)
            msg.header.stamp = self.node.get_clock().now().to_msg()
            self.publisher.publish(msg)
            
            if DEBUG:
                time_end = time.time()
                self.node.get_logger().info(f'图像处理耗时: {time_end - time_start:.4f}s')
                
        except Exception as e:
            self.node.get_logger().error(f"图像处理错误: {str(e)}")

def IsImageColor(enType):
    """判断图像格式是彩色还是黑白"""
    dates = {
        PixelType_Gvsp_RGB8_Packed: 'color',
        PixelType_Gvsp_BGR8_Packed: 'color',
        PixelType_Gvsp_YUV422_Packed: 'color',
        PixelType_Gvsp_YUV422_YUYV_Packed: 'color',
        PixelType_Gvsp_BayerGR8: 'color',
        PixelType_Gvsp_BayerRG8: 'color',
        PixelType_Gvsp_BayerGB8: 'color',
        PixelType_Gvsp_BayerBG8: 'color',
        PixelType_Gvsp_BayerGB10: 'color',
        PixelType_Gvsp_BayerGB10_Packed: 'color',
        PixelType_Gvsp_BayerBG10: 'color',
        PixelType_Gvsp_BayerBG10_Packed: 'color',
        PixelType_Gvsp_BayerRG10: 'color',
        PixelType_Gvsp_BayerRG10_Packed: 'color',
        PixelType_Gvsp_BayerGR10: 'color',
        PixelType_Gvsp_BayerGR10_Packed: 'color',
        PixelType_Gvsp_BayerGB12: 'color',
        PixelType_Gvsp_BayerGB12_Packed: 'color',
        PixelType_Gvsp_BayerBG12: 'color',
        PixelType_Gvsp_BayerBG12_Packed: 'color',
        PixelType_Gvsp_BayerRG12: 'color',
        PixelType_Gvsp_BayerRG12_Packed: 'color',
        PixelType_Gvsp_BayerGR12: 'color',
        PixelType_Gvsp_BayerGR12_Packed: 'color',
        PixelType_Gvsp_Mono8: 'mono',
        PixelType_Gvsp_Mono10: 'mono',
        PixelType_Gvsp_Mono10_Packed: 'mono',
        PixelType_Gvsp_Mono12: 'mono',
        PixelType_Gvsp_Mono12_Packed: 'mono'}
    return dates.get(enType, '未知')

def press_any_key_exit():
    """等待按键退出函数"""
    fd = sys.stdin.fileno()
    old_ttyinfo = termios.tcgetattr(fd)
    new_ttyinfo = old_ttyinfo[:]
    new_ttyinfo[3] &= ~termios.ICANON
    new_ttyinfo[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
    try:
        os.read(fd, 7)
    except:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        # 初始化SDK
        MvCamera.MV_CC_Initialize()
        
        # 记录SDK版本
        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        self.get_logger().info(f"SDK版本[0x{SDKVersion:x}]")
        
        # 枚举设备
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
        
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            self.get_logger().error(f"枚举设备失败! ret[0x{ret:x}]")
            return
            
        if deviceList.nDeviceNum == 0:
            self.get_logger().error("未找到设备!")
            return
            
        self.get_logger().info(f"找到 {deviceList.nDeviceNum} 个设备!")
        
        # 选择第一个设备
        nConnectionNum = 0
        
        # 创建相机实例
        self.cam = MvCamera()
        
        # 选择设备并创建句柄
        stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
        
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            self.get_logger().error(f"创建句柄失败! ret[0x{ret:x}]")
            return
            
        # 打开设备
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.get_logger().error(f"打开设备失败! ret[0x{ret:x}]")
            return
            
        # 探测网络最佳包大小(只对GigE相机有效)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE or stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    self.get_logger().warn(f"设置包大小失败! ret[0x{ret:x}]")
            else:
                self.get_logger().warn(f"获取包大小失败! ret[0x{nPacketSize:x}]")
                
        # 设置触发模式为off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            self.get_logger().error(f"设置触发模式失败! ret[0x{ret:x}]")
            return
            
        # 获取数据包大小
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            self.get_logger().error(f"获取有效负载大小失败! ret[0x{ret:x}]")
            return
            
        nPayloadSize = stParam.nCurValue
        
        # 开始取流
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.get_logger().error(f"开始采集失败! ret[0x{ret:x}]")
            return
            
        # 创建数据缓冲区
        self.data_buf = (c_ubyte * nPayloadSize)()
        
        # 创建并启动相机管理器
        self.camera_manager = CameraManager(self.cam, byref(self.data_buf), nPayloadSize, self)
        self.camera_manager.start()
        
        self.get_logger().info("相机已初始化并开始采集图像")
        
    def __del__(self):
        """析构函数，确保资源释放"""
        global g_bExit
        g_bExit = True
        
        # 停止采集
        if hasattr(self, 'cam'):
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                self.get_logger().error(f"停止采集失败! ret[0x{ret:x}]")
                
            # 关闭设备
            ret = self.cam.MV_CC_CloseDevice()
            if ret != 0:
                self.get_logger().error(f"关闭设备失败! ret[0x{ret:x}]")
                
            # 销毁句柄
            ret = self.cam.MV_CC_DestroyHandle()
            if ret != 0:
                self.get_logger().error(f"销毁句柄失败! ret[0x{ret:x}]")
                
        # 反初始化SDK
        MvCamera.MV_CC_Finalize()

def main(args=None):
    rclpy.init(args=args)
    
    camera_node = CameraNode()
    
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        camera_node.get_logger().info("用户中断，正在关闭...")
    finally:
        # 清理资源
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()