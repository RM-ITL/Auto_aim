#!/usr/bin/env python3
"""定靶转云台 —— world_yaw 抖动误差源「三选一」诊断 + 图像<->IMU 偏移 Δt 标定。

背景
----
打一块**静止**装甲板、只**来回转云台**时，可算出的装甲板世界方位角本应恒定。
它残余的抖动来自三类互不相同的系统性误差，各自和一个不同的自变量线性相关：

    world_yaw 抖动 ≈ Δt·ω  +  k_θ·θ  +  k_α·α  +  随机噪声
                     └─时间偏移┘  └─外参/标定┘ └─动态/模糊┘

  1. 时间偏移 Δt   ——误差 ∝ ω_gimbal(带符号角速率)。图像↔IMU 到达延迟差，
                     斜率即 Δt(秒)，填进 Gimbal.image_imu_offset 即可消除。
  2. 外参/q_calib ——误差 ∝ θ_gimbal(云台指向角)。相机-IMU 旋转标定有偏，
                     误差随云台**指向**而非速率变化。→ 查 rotation_matrix_*/q_calib。
  3. 动态/运动模糊——误差 ∝ α(角加速度) 或 |ω|(速率大小)。曝光时序/模糊/插值滞后。

本工具对同一段数据，把 world_yaw 抖动分别回归到上述每个自变量，比较各自 R²
(可解释比例)，**谁最高谁就是你的主误差源**；再做一次三者联合回归，给出扣掉
外参/动态后的「干净」Δt。若四项 R² 都低，则残差主要是 PnP/检测随机噪声——
Δt 这条路收益有限，应转去做 optimize_yaw 插值 / 亚像素角点。

deep_node 每帧发布同帧对(raw_armor_yaw, raw_gimbal_yaw, raw_armor_valid)，
两者同帧，故管线延迟在回归里抵消，不污染诊断。

用法
----
1. 固定一块装甲板于 2~3.5m 正前方：ros2 run test_pipeline deep_node
   (首次诊断先确保 config 里 Gimbal.image_imu_offset = 0.0)
2. 手动/指令让云台在 yaw 上来回扫 ±(15~30°)、~0.5-1Hz，持续 20s+。
3. 另开终端：python3 tools/latency/measure_image_imu_offset.py
4. Ctrl-C 结束，看「主误差源」判定与建议。

只读 /debug，不改任何模块，不需要下位机配合。
"""

import sys

import numpy as np
import rclpy
from rclpy.node import Node

try:
    from autoaim_msgs.msg import Debug
except ImportError:
    sys.stderr.write(
        "无法导入 autoaim_msgs.msg.Debug — 请先 source 工作空间:\n"
        "  source install/setup.bash\n"
    )
    raise

MIN_OMEGA = 0.5     # rad/s，回归有效性判据：需有足够帧 |ω| 超过它才算「云台在动」
MAX_ABS_DT_MS = 60.0  # 合理 Δt 上界；超出多半是数据/装配问题，仅告警
R2_STRONG = 0.5     # 单项 R² 高于此 → 该项是显著误差源
R2_WEAK = 0.2       # 最高项 R² 仍低于此 → 残差主要是随机噪声(PnP/检测)


def _unwrap(a):
    return np.unwrap(np.asarray(a, dtype=float))


def _fit1(y, x):
    """单变量线性回归 y = slope·x + b，返回 (slope, R²)。"""
    if np.std(x) < 1e-9:
        return 0.0, 0.0
    slope, b = np.polyfit(x, y, 1)
    y_hat = slope * x + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(slope), float(r2)


def _fit_multi(y, cols):
    """多元线性回归 y ~ Σ cols + 1，返回 (系数数组, 联合R², 残差RMS)。"""
    X = np.column_stack(list(cols) + [np.ones_like(y)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    resid_rms = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    return coef, r2, resid_rms


def _diagnose(t, armor_yaw, gimbal_yaw, valid, dt):
    """把 world_yaw 抖动回归到 ω/θ/α/|ω| 四个自变量，返回各自拟合与联合结果。"""
    grid = np.arange(t[0], t[-1], dt)
    valid_u = np.interp(grid, t, np.asarray(valid, dtype=float)) > 0.5
    armor_u = np.interp(grid, t, _unwrap(armor_yaw))
    gimbal_u = np.interp(grid, t, _unwrap(gimbal_yaw))

    omega = np.gradient(gimbal_u, dt)     # 角速率(带符号) → 时间偏移
    alpha = np.gradient(omega, dt)        # 角加速度 → 动态/滞后
    theta = gimbal_u - np.mean(gimbal_u)  # 指向角(去均值) → 外参/标定
    absom = np.abs(omega)                 # 速率大小 → 运动模糊

    if valid_u.sum() < 30 or (valid_u & (absom > MIN_OMEGA)).sum() < 20:
        return None

    m = valid_u
    y = armor_u[m] - np.mean(armor_u[m])

    dt_slope, r2_omega = _fit1(y, omega[m])
    _, r2_theta = _fit1(y, theta[m])
    _, r2_alpha = _fit1(y, alpha[m])
    _, r2_absom = _fit1(y, absom[m] - np.mean(absom[m]))

    # 联合回归：扣掉 θ/α 后 ω 的偏系数即「干净」Δt。
    coef, r2_multi, resid_rms = _fit_multi(y, [omega[m], theta[m], alpha[m]])

    return {
        "n": int(m.sum()),
        "omega_span": (float(np.min(omega[m])), float(np.max(omega[m]))),
        "theta_span_deg": float(np.rad2deg(np.ptp(theta[m]))),
        "rms0_deg": float(np.rad2deg(np.sqrt(np.mean(y ** 2)))),
        # 各单项：(名称, R², 物理含义, 建议)
        "singles": [
            ("ω 角速率", r2_omega, "时间偏移 Δt"),
            ("θ 指向角", r2_theta, "外参/q_calib 标定"),
            ("α 角加速度", r2_alpha, "动态/插值滞后"),
            ("|ω| 速率大小", r2_absom, "运动模糊"),
        ],
        "dt_ms_single": dt_slope * 1e3,       # 单变量拟合给的 Δt
        "dt_ms_multi": coef[0] * 1e3,         # 联合回归给的「干净」Δt
        "r2_multi": r2_multi,
        "resid_rms_deg": float(np.rad2deg(resid_rms)),
    }


class Collector(Node):
    def __init__(self):
        super().__init__("image_imu_offset_probe")
        self.t, self.armor_yaw, self.gimbal_yaw, self.valid = [], [], [], []
        self.create_subscription(Debug, "debug", self._cb, 50)
        self.get_logger().info("正在采集 /debug ... 定靶并来回转云台, Ctrl-C 结束")

    def _cb(self, m: Debug):
        # 用节点时钟作到达时间戳；deep_node 按帧率发布，近似均匀采样。
        self.t.append(self.get_clock().now().nanoseconds * 1e-9)
        self.armor_yaw.append(m.raw_armor_yaw)
        self.gimbal_yaw.append(m.raw_gimbal_yaw)
        self.valid.append(bool(m.raw_armor_valid))


def _estimate(t, armor_yaw, gimbal_yaw, valid, dt):
    """回归 raw_armor_yaw ~ c0 + Δt·ω_gimbal，返回斜率 Δt(秒) 及拟合质量。"""
    grid = np.arange(t[0], t[-1], dt)
    # 有效帧掩码插值到网格（>0.5 视为有效），无效段不参与。
    valid_u = np.interp(grid, t, np.asarray(valid, dtype=float)) > 0.5
    armor_u = np.interp(grid, t, _unwrap(armor_yaw))
    gimbal_u = np.interp(grid, t, _unwrap(gimbal_yaw))

    # 中心差分求云台角速率 ω_gimbal。
    omega = np.gradient(gimbal_u, dt)

    # 只在「云台确实在动」且「该帧有效」的样本上回归。
    move = valid_u & (np.abs(omega) > MIN_OMEGA)
    if move.sum() < 30:
        return None

    x = omega[move]
    y = armor_u[move]
    y = y - np.mean(y)  # 去掉常数项（静止靶的真实世界方位）

    # 一次多项式拟合：y = slope·x + b，slope 即 Δt。
    slope, b = np.polyfit(x, y, 1)
    y_hat = slope * x + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    resid_rms0 = np.rad2deg(np.sqrt(np.mean(y ** 2)))       # 校正前残差
    resid_rms1 = np.rad2deg(np.sqrt(np.mean((y - y_hat) ** 2)))  # 拟合后残差
    return {
        "dt_ms": slope * 1e3,
        "r2": r2,
        "n": int(move.sum()),
        "omega_span": (float(np.min(x)), float(np.max(x))),
        "resid_rms0_deg": float(resid_rms0),
        "resid_rms1_deg": float(resid_rms1),
    }


def analyze(c: Collector):
    n = len(c.t)
    if n < 50:
        print(f"\n样本太少 ({n} 条), 无法诊断。请延长采集时间。")
        return
    if sum(c.valid) < 30:
        print(f"\n有效帧太少 ({sum(c.valid)} 条)。确认相机能持续检测到那块静止装甲板。")
        return
    t = np.asarray(c.t)
    dts = np.diff(t)
    dt = max(float(np.median(dts[dts > 0])), 0.002)

    print("\n========= world_yaw 抖动 误差源「三选一」诊断 =========")
    print(f"样本 {n} 条, 时长 {t[-1] - t[0]:.1f}s, 发布率 ≈ {1.0 / dt:.0f} Hz")

    r = _diagnose(t, c.armor_yaw, c.gimbal_yaw, c.valid, dt)
    if r is None:
        print("运动样本不足 — 云台要更明显地来回转 (|ω|>0.5rad/s), 且全程盯住静止靶后重测。")
        print("======================================================")
        return

    lo, hi = r["omega_span"]
    print(f"回归样本 {r['n']} 条 | ω 范围 [{lo:+.2f},{hi:+.2f}]rad/s | θ 扫过 {r['theta_span_deg']:.0f}° | "
          f"待解释抖动 RMS {r['rms0_deg']:.3f}°")
    print("\n  各误差源单独回归 (R² = world_yaw 抖动被该项解释的比例):")
    singles = sorted(r["singles"], key=lambda s: s[1], reverse=True)
    for name, r2, meaning in singles:
        bar = "█" * int(round(r2 * 20))
        print(f"    {name:<12} R²={r2:5.3f} {bar:<20} → {meaning}")

    best_name, best_r2, best_meaning = singles[0]
    print(f"\n  联合回归(ω+θ+α) 总解释率 R²={r['r2_multi']:.3f}, "
          f"剩余噪声地板 RMS={r['resid_rms_deg']:.3f}° (≈PnP/检测随机噪声)")

    print("\n  >>> 判定: ", end="")
    if best_r2 < R2_WEAK:
        print("残差主要是随机噪声，无显著系统性误差源 <<<")
        print("  → 图像↔IMU 偏移不是你的主要矛盾，image_imu_offset 保持 0 即可。")
        print("  → 转去做收益更高的观测端项：optimize_yaw 三点抛物线插值、亚像素角点修正。")
    elif best_meaning == "时间偏移 Δt":
        print(f"主误差源 = 时间偏移 Δt (R²={best_r2:.3f}) <<<")
        dt_ms = r["dt_ms_multi"]
        print(f"  → Δt(联合回归, 已扣外参/动态) ≈ {dt_ms:+.2f} ms ({dt_ms * 1e-3:+.5f} s)")
        if abs(dt_ms) > MAX_ABS_DT_MS:
            print(f"  ⚠ |Δt|>{MAX_ABS_DT_MS:g}ms 偏大, 先确认定靶且云台纯 yaw 转动。")
        print(f"  → 填进 config: Gimbal.image_imu_offset: {dt_ms * 1e-3:+.5f}")
        print("    重测该项 R² 应塌陷、Δt→0 即验证成功。")
    elif best_meaning.startswith("外参"):
        print(f"主误差源 = 外参/q_calib 标定 (R²={best_r2:.3f}) <<<")
        print("  → 误差随云台指向而非速率变化。查 solver.coord_converter 的")
        print("    rotation_matrix_camera_to_gimbal / _gimbal_to_imu 与 Gimbal.q_calib。")
        print("  → 标定未修正前，填 image_imu_offset 无意义(Δt 会被外参污染)。")
    else:
        print(f"主误差源 = 动态/运动模糊 ({best_name}, R²={best_r2:.3f}) <<<")
        print("  → 误差随角加速度/速率大小变化。考虑：降低曝光时间、加曝光中点参考、")
        print("    或减小云台运动激烈度后重测；此时 Δt 亦不可信。")
    print("======================================================")


def main():
    rclpy.init()
    node = Collector()
    last_note = node.get_clock().now()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            now = node.get_clock().now()
            if (now - last_note).nanoseconds * 1e-9 >= 3.0:
                if len(node.t) == 0:
                    node.get_logger().warn(
                        "仍未收到 /debug — 确认 deep_node 已启动 (ros2 topic hz /debug 自查)")
                last_note = now
    except KeyboardInterrupt:
        pass
    finally:
        analyze(node)
        node.destroy_node()
        # Ctrl-C 时 rclpy 默认信号处理可能已关掉 context，避免重复 shutdown 报错。
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
