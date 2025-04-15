哨兵基础自瞄框架

# Camera Catch Driver - 多线程优化版本

## 🚀 最新更新 (2025/4/15)

### 核心优化
- **多线程架构**：图像采集/处理/发布全流程并行流水线
- **异步处理**：实现真正的非阻塞图像处理
- **ROS零拷贝**：极致通信性能优化
- **OpenVINO加速**：新增Intel集成GPU推理支持

### ⚡ 性能突破
| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| CPU占用率 | 100% | 70% |
| 检测延迟 | 1.5s | 实时 |
| 推理设备 | CPU-only | Intel iGPU |
| 模型精度 | FP32 | INT8量化 |

## 技术架构

## ✨ 关键技术
- 🧵 **多线程解耦**：采集/处理/发布独立线程
- 🔥 **OpenVINO加速**：
  - 支持Intel集成GPU推理
  - INT8量化压缩权重
  - 资源占用降低40%
- 📦 **ROS零拷贝**：消除数据复制开销
- ⚙️ **INT8量化**：保持精度同时减少计算负载

## 硬件兼容性
✔ 11代Intel Core i7及更新平台  
✔ 支持Intel Iris Xe集成显卡  
✔ 推荐搭配ROS2 Humble版本

## 优化对比测试（原版vs 改版GPU推理检测）

![727d3596712dd726e8797312f59bb1b](https://github.com/user-attachments/assets/be866e0a-f95e-4e8a-931e-1c13aaf62ea2)

![95464d135a0bc4ccc205fdda171724a](https://github.com/user-attachments/assets/2fa74bff-edee-469e-bc8c-58ba730b1fd5)


## 11代i7测试效果
![05c36400cda273e32f3a87c2f71dfeb](https://github.com/user-attachments/assets/3d491a4c-4aa2-44da-9e15-cd9d1a544c2f)

