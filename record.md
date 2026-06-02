#6.1  
-建立了分支hou  
##当前问题  
-四版收敛抖动
-小陀螺的yaw输出有问题  
-速度标定  

#6.2
你的推断大体是合理的，但表述上要更精确一点：

  不是“识别模型是四板模型”导致问题，而是“当前普通目标的跟踪模型从一开始就是四板整车 EKF，而它长期只被单板观测驱动”，这和你看到的“估计抖
  动、最后缩成一团”是吻合的。

  判断
  /home/hou/下载/上海科技大学2026视觉技术方案开源.pdf 的核心观点和你的现象基本一致：

  - 文档在 5B.1 明确说了：低速或不旋转时，整车模型里的旋转参数没有足够观测支撑，强行估计会发散。
  - 文档在 9.4.4 推荐的是双模型：低速用装甲板模型，高速再上整车模型，中间用 BOTH 过渡，并且要有 rotate_counter 一类的可信度门槛。
  - 文档在 9.4.3 还强调：装甲板切换时，不应该把整车模型整体打掉，而应该保留整车中心，只处理板级切换。

  你当前代码和这套思路的差异很大：

  - 普通目标一上来就走四板 Target，没有单独的装甲板模型分支，src/core/auto_aim/tracker/src/tracker.cpp:317。
  - single_plate_mode 只是个规划标志，不会改变 EKF 状态更新方式，src/core/auto_aim/target/predictor/include/target.hpp:28 src/core/
    auto_aim/target/predictor/src/target.cpp:188。
  - 即使已经长期单板观测，update_ypda() 仍然在更新完整 11 维整车状态，src/core/auto_aim/target/predictor/src/target.cpp:233。

  为什么你的推断成立
  数学上也能解释“缩成一团”：

  - 四板模型里 x[8]=r，x[9]=l，x[10]=h，src/core/auto_aim/target/predictor/src/target.cpp:336。
  - 对 id=0/2 的板，观测只和 r、z 相关；对 id=1/3 的板，观测只和 r+l、z+h 相关，src/core/auto_aim/target/predictor/src/target.cpp:376
    src/core/auto_aim/target/predictor/src/target.cpp:394。
  - 这意味着如果长时间只看到同一类板，l/h 要么完全不可观，要么只能以组合量被观测，无法稳定分解回 r、l、h。
  - 当前代码没有“观测不足就降级模型”的机制，所以这些参数会漂、塌缩，最后触发 diverged() 的硬阈值，src/core/auto_aim/target/predictor/
    src/target.cpp:333。
  - 一旦发散，tracker 直接 lost，再重建目标，src/core/auto_aim/tracker/src/tracker.cpp:108。这就是你日志里反复 tracking -> lost ->
    detecting -> tracking 和 yaw 突变的根源。

  还有一个次要放大器：

  - 当前关联不是按马氏距离，而是同名同类型里取 center.x 最小的板，src/core/auto_aim/tracker/src/tracker.cpp:345。四板退化时，这会进一步
    加重切板抖动。

  修改建议
  优先级从高到低建议这样做：

  1. 按 PDF 的思路做双模型。

  - 低速/长期单板：用装甲板模型。
  - 高速且已观察到稳定旋转：用整车模型。
  - 中间过渡区：两模型同时跑，按可信度融合。
  - 不要让普通目标默认从第一帧就只跑 11 维整车模型。

  2. 给整车模型加“可信度门槛”。

  - 至少要有 rotate_counter。
  - 至少要统计最近窗口内看到过多少不同板 ID。
  - 没看到足够旋转、没看到足够不同板时，整车模型不参与瞄准输出。

  3. 单板观测阶段不要继续更新 r/l/h。

  - 这是最直接的止血方案。
  - 可以冻结 x[8:10]，或者切到一个降阶观测模型，只更新中心和平面 yaw。
  - 至少不要在 single_plate_mode 下还继续用完整 h_jacobian() 去拉 r/l/h。

  4. 板切换不要等价成整车丢失。

  - 保留整车中心状态。
  - 只重置板级模型或板索引。
  - 这是 PDF 9.4.3 的关键点，而你当前实现更接近“整车一起丢”。

  5. 观测关联改成马氏距离/预测一致性，不要用 center.x。

  - 现在这个规则在四板靠拢时非常脆弱。
  - 至少要把 yaw/pitch/distance 对预测的 innovation 纳入评分。

  6. 规划层要接受“当前整车模型不可信”。

  - 不可信时直接退回单板瞄准，不要继续从四板虚拟几何里选最近板。
  - 否则上游状态一塌，yaw 还是会跳。

  一句话总结
  你的推断方向是对的：问题核心就是“单板长期观测下，四板整车 EKF 的几何参数不可观，继续硬估计导致发散”。
  真正该改的不是检测器，而是跟踪架构：从“单一四板整车模型”改成“装甲板模型 + 整车模型”的双模型自适应。
