#! /usr/bin/env python

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

sys.path.append("/home/shaobing/rm_ws/src/pkg02_helloworld_py/MvImport")
from MvCameraControl_class import *


g_bExit = False

# 定义一个结构体，用于表示转换参数
class ConvertParam(ctypes.Structure):
    _fields_ = [
        ("pDstBuffer", ctypes.POINTER(ctypes.c_ubyte)),
        ("nDstLen", ctypes.c_size_t)
    ]

# 判读图像格式是彩色还是黑白
def IsImageColor(enType):
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

# 为线程定义一个函数
def work_thread_simple(cam=0, pData=0, nDataSize=0):
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    while True:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret == 0:
            print ("get one frame: Width[%d], Height[%d], PixelType[0x%x], nFrameNum[%d]"  % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.enPixelType, stOutFrame.stFrameInfo.nFrameNum))
            cam.MV_CC_FreeImageBuffer(stOutFrame)
        else:
            print ("no data[0x%x]" % ret)
        if g_bExit == True:
                break

 #实现GetImagebuffer函数取流，HIK格式转换函数
def work_thread(cam=0, pData=0, nDataSize=0):
    node = Node('image_publisher')
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    print("work_thread started!\n")
    img_buff = None
    # 加载动态链接库
    libc = ctypes.CDLL('libc.so.6')
    bridge = cv_bridge.CvBridge()
    publisher_ = node.create_publisher(Image, 'image_topic', 10)
    
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            print ("MV_CC_GetOneFrameTimeout: Width[%d], Height[%d], nFrameNum[%d]"  % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
            time_start = time.time()
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            if IsImageColor(stFrameInfo.enPixelType) == 'mono':
                print("mono!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
                nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight
            elif IsImageColor(stFrameInfo.enPixelType) == 'color':
                print("color!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed  # opecv要用BGR，不能使用RGB
                nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight* 3
            else:
                print("not support!!!")
            if img_buff is None:
                img_buff = (c_ubyte * stFrameInfo.nFrameLen)()
            # ---
            stConvertParam.nWidth = stFrameInfo.nWidth
            stConvertParam.nHeight = stFrameInfo.nHeight
            stConvertParam.pSrcData = cast(pData, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
            stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
            stConvertParam.nDstBufferSize = nConvertSize
            ret = cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                print("convert pixel fail! ret[0x%x]" % ret)
                del stConvertParam.pSrcData
                sys.exit()
            else:
                print("convert ok!!")
                # 转OpenCV
                # 黑白处理
                if IsImageColor(stFrameInfo.enPixelType) == 'mono':
                    img_buff = (c_ubyte * stConvertParam.nDstLen)()
                    # 使用 memcpy 复制内存
                    libc.memcpy(ctypes.byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                    img_buff = np.frombuffer(img_buff,count=int(stConvertParam.nDstLen), dtype=np.uint8)
                    img_buff = img_buff.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    print("mono ok!!")
                    # 发布图像消息，不显示
                    msg = bridge.cv2_to_imgmsg(img_buff, encoding="passthrough")
                    publisher_.publish(msg)
                # 彩色处理
                if IsImageColor(stFrameInfo.enPixelType) == 'color':
                    img_buff = (c_ubyte * stConvertParam.nDstLen)()
                    libc.memcpy(ctypes.byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                    img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)
                    img_buff = img_buff.reshape(stFrameInfo.nHeight,stFrameInfo.nWidth,3)
                    print("color ok!!")
                    # 发布图像消息，不显示
                    msg = bridge.cv2_to_imgmsg(img_buff, encoding="bgr8")
                    publisher_.publish(msg)

                time_end = time.time()
                print('time cos:', time_end - time_start, 's') 
        else:
            print ("no data[0x%x]" % ret)
        if g_bExit == True:
                break

def press_any_key_exit():
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
    
def main(args=None):
    rclpy.init(args=args)
    
    # ch:初始化SDK | en: initialize SDK
    MvCamera.MV_CC_Initialize()

    SDKVersion = MvCamera.MV_CC_GetSDKVersion()
    print ("SDKVersion[0x%x]" % SDKVersion)

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
    
    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print ("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print ("find no device!")
        sys.exit()

    print ("Find %d devices!" % deviceList.nDeviceNum)

    # 自动选择第一个设备
    nConnectionNum = 0
    
    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print ("No valid device found!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()
    
    # ch:选择设备并创建句柄| en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print ("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print ("open device fail! ret[0x%x]" % ret)
        sys.exit()
    
    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE or stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
            if ret != 0:
                print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print ("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:获取数据包大小 | en:Get payload size
    stParam =  MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    
    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print ("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    nPayloadSize = stParam.nCurValue

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print ("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    data_buf = (c_ubyte * nPayloadSize)()

    try:
        hThreadHandle = threading.Thread(target=work_thread, args=(cam, byref(data_buf), nPayloadSize))
        hThreadHandle.start()
    except:
        print ("error: unable to start thread")
        
    print ("Camera started. Press Ctrl+C to exit.")
    
    # 使用主线程等待用户输入来停止程序
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        g_bExit = True
        
    hThreadHandle.join()

    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print ("stop grabbing fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print ("close deivce fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print ("destroy handle fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    del data_buf

    # ch:反初始化SDK | en: finalize SDK
    MvCamera.MV_CC_Finalize()
    
if __name__ == '__main__':
    main()
