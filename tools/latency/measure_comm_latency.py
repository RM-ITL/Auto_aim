#!/usr/bin/env python3
"""comm_latency 在线测量工具 —— 估计「电控解包 → 电机到位」的控制滞后。

原理
----
planner 线程每回合向 /debug 发布:
    msg.yaw          上位机规划的目标 yaw (指令)
    msg.yaw_gimbal   下位机反馈的当前 yaw (响应)
跟踪来回运动/小陀螺目标时, 反馈是指令的「延迟 + 平滑」版本。
把指令整体前移 τ 后与反馈做最佳对齐 (最小化均方误差),
使误差最小的 τ 即 comm_latency 的估计。pitch 同理。

用法
----
1. 正常启动自瞄:      ros2 run test_pipeline deep_node
2. 让云台稳定跟踪一个「来回平移」或「小陀螺」目标 (信号必须有足够运动, 20s+).
3. 另开终端:          python3 tools/latency/measure_comm_latency.py
4. Ctrl-C 结束, 打印:
   - yaw / pitch 的最佳滞后 τ、残差 RMS、「延迟贡献占比」(指令↔反馈, 控制滞后);
   - offset_rms = RMS(target_yaw - yaw_gimbal) (目标真值↔枪管实际, 命中诚实指标),
     并按自转角速度 |ω| 拆成 低速档/高速档, 分别对应 low/high_speed_delay_time。

只读 /debug, 不改任何模块, 不需要下位机配合。
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

MAX_LAG_S = 0.15   # 搜索的最大滞后 (s), 150ms 足够覆盖笨重云台
MOVE_STD_DEG = 1.0  # 反馈角滑窗标准差低于此值视为静止段, 不参与估计
DECISION_OMEGA = 5.0  # rad/s, 自转角速度高/低速分档阈值 (对齐 Planner.decision_speed),
#                       用于把 offset_rms 拆成 低速档/高速档 分别验证两个 delay_time


def _wrap(x):
    """把角度差包裹到 [-pi, pi], 避免 ±pi 处跳变污染 RMS。"""
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


class Collector(Node):
    def __init__(self):
        super().__init__("comm_latency_probe")
        self.t, self.cmd_yaw, self.fb_yaw = [], [], []
        self.cmd_pitch, self.fb_pitch, self.fire = [], [], []
        self.tgt_yaw, self.omega = [], []  # 目标参考 yaw (真值) 与自转角速度, 用于 offset_rms
        self.create_subscription(Debug, "debug", self._cb, 50)
        self.get_logger().info("正在采集 /debug ... 让云台跟踪运动目标, Ctrl-C 结束")

    def _cb(self, m: Debug):
        # 用节点时钟作为到达时间戳; planner 按其回合率发布, 采样近似均匀。
        self.t.append(self.get_clock().now().nanoseconds * 1e-9)
        self.cmd_yaw.append(m.yaw)
        self.fb_yaw.append(m.yaw_gimbal)
        self.cmd_pitch.append(m.pitch)
        self.fb_pitch.append(m.pitch_gimbal)
        self.fire.append(1.0 if m.fire else 0.0)
        self.tgt_yaw.append(m.target_yaw)
        self.omega.append(m.omega)


def _best_lag(t, cmd, fb, dt):
    """在均匀网格上搜索使 fb(k) ≈ cmd(k-n) 均方误差最小的滞后 n*·dt。"""
    grid = np.arange(t[0], t[-1], dt)
    cmd_u = np.interp(grid, t, cmd)
    fb_u = np.interp(grid, t, fb)

    # 只在「运动段」评估: 反馈角滑窗标准差达标的样本才计入。
    win = max(3, int(0.05 / dt))
    move = np.zeros(len(fb_u), dtype=bool)
    for i in range(len(fb_u)):
        s = fb_u[max(0, i - win): i + win]
        move[i] = np.std(np.rad2deg(s)) > MOVE_STD_DEG
    if move.sum() < win * 4:
        return None

    max_n = int(MAX_LAG_S / dt)
    errs = []
    for n in range(max_n + 1):
        a = fb_u[n:]
        b = cmd_u[: len(fb_u) - n] if n > 0 else cmd_u
        msk = move[n:]
        if msk.sum() < win * 4:
            errs.append(np.inf)
            continue
        errs.append(float(np.mean((a[msk] - b[msk]) ** 2)))
    errs = np.array(errs)
    n_star = int(np.argmin(errs))
    rms0 = np.sqrt(errs[0])                       # 零滞后残差 (含纯延迟误差)
    rms_best = np.sqrt(errs[n_star])              # 最佳对齐后残差 (不可约跟踪误差)
    return {
        "lag_ms": n_star * dt * 1e3,
        "rms0_deg": np.rad2deg(rms0),
        "rms_best_deg": np.rad2deg(rms_best),
        "lag_share": 1.0 - (rms_best / rms0 if rms0 > 1e-9 else 1.0),
    }


def _offset_rms(t, tgt_yaw, fb_yaw, omega, dt):
    """诚实瞄准误差: RMS(target_yaw - yaw_gimbal), 即枪管实际指向与目标参考的角度偏差。

    与 _best_lag 的「指令↔反馈」控制残差不同, 这里用 target_yaw (目标真值参考) 作基准,
    骗不过云台在零点附近的小幅震荡 —— 这才是命中率的诚实指标。
    只在运动段统计, 并按 |omega| 拆成低速档 / 高速档, 分别对应 low/high_speed_delay_time。
    """
    grid = np.arange(t[0], t[-1], dt)
    tgt_u = np.interp(grid, t, tgt_yaw)
    fb_u = np.interp(grid, t, fb_yaw)
    om_u = np.abs(np.interp(grid, t, omega))
    err = np.rad2deg(np.abs(_wrap(tgt_u - fb_u)))

    win = max(3, int(0.05 / dt))
    move = np.zeros(len(fb_u), dtype=bool)
    for i in range(len(fb_u)):
        s = fb_u[max(0, i - win): i + win]
        move[i] = np.std(np.rad2deg(s)) > MOVE_STD_DEG
    if move.sum() < win * 4:
        return None

    def _rms(mask):
        return float(np.sqrt(np.mean(err[mask] ** 2))) if mask.sum() else None

    lo = move & (om_u < DECISION_OMEGA)
    hi = move & (om_u >= DECISION_OMEGA)
    return {
        "all": _rms(move),
        "low": _rms(lo), "n_low": int(lo.sum()),
        "high": _rms(hi), "n_high": int(hi.sum()),
    }


def _report_offset(r):
    if r is None:
        print("[OFFSET] 运动样本不足 — 无法评估 offset_rms")
        return
    print(f"[OFFSET] offset_rms(target_yaw - yaw_gimbal) 整体 = {r['all']:.3f}°  (3m小装甲预算 ~1.14°)")
    lo = f"{r['low']:.3f}°" if r["low"] is not None else "样本不足"
    hi = f"{r['high']:.3f}°" if r["high"] is not None else "样本不足"
    print(f"         低速档 |ω|<{DECISION_OMEGA:g} rad/s: {lo} (n={r['n_low']}) -> 验证 low_speed_delay_time")
    print(f"         高速档 |ω|>={DECISION_OMEGA:g} rad/s: {hi} (n={r['n_high']}) -> 验证 high_speed_delay_time")


def _report(name, r):
    if r is None:
        print(f"[{name}] 运动样本不足 — 请让云台更明显地来回运动后重测")
        return
    print(
        f"[{name}] comm_latency ≈ {r['lag_ms']:.1f} ms | "
        f"零滞后RMS {r['rms0_deg']:.3f}° -> 对齐后RMS {r['rms_best_deg']:.3f}° | "
        f"延迟贡献占比 {r['lag_share'] * 100:.0f}%"
    )


def analyze(c: Collector):
    n = len(c.t)
    if n < 50:
        print(f"\n样本太少 ({n} 条), 无法估计。请延长采集时间。")
        return
    t = np.asarray(c.t)
    dts = np.diff(t)
    dt = float(np.median(dts[dts > 0]))
    dt = max(dt, 0.002)  # 下限 2ms, 防止个别抖动把网格撑爆
    print("\n================ comm_latency 测量结果 ================")
    print(
        f"样本 {n} 条, 时长 {t[-1] - t[0]:.1f}s, "
        f"发布率 ≈ {1.0 / dt:.0f} Hz (dt≈{dt * 1e3:.1f}ms)"
    )
    _report("YAW  ", _best_lag(t, np.asarray(c.cmd_yaw), np.asarray(c.fb_yaw), dt))
    _report("PITCH", _best_lag(t, np.asarray(c.cmd_pitch), np.asarray(c.fb_pitch), dt))
    print("------------------------------------------------------")
    _report_offset(_offset_rms(
        t, np.asarray(c.tgt_yaw), np.asarray(c.fb_yaw), np.asarray(c.omega), dt))
    print(f"开火占比 fire_rate = {np.mean(c.fire) * 100:.0f}%")
    print("------------------------------------------------------")
    print("下一步: 把 YAW 的 comm_latency 值 (加上 ~2ms send_latency) 填进")
    print("        config 的 Planner.high_speed_delay_time / low_speed_delay_time,")
    print("        重测应看到 offset_rms 下降 (比 fire_rate 上升更可信)。")
    print("======================================================")


def main():
    rclpy.init()
    node = Collector()
    # 用带超时的 spin_once 循环, 而不是 rclpy.spin() + 自定义信号处理:
    # spin() 无消息时卡在 C 层等待, Python 的 Ctrl-C 要等它返回才生效, 会「停不下来」。
    # spin_once(timeout_sec) 周期性返回, 让 KeyboardInterrupt 能被及时捕获。
    last_note = node.get_clock().now()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            now = node.get_clock().now()
            if (now - last_note).nanoseconds * 1e-9 >= 3.0:
                if len(node.t) == 0:
                    node.get_logger().warn(
                        "仍未收到 /debug 消息 — 确认 deep_node 已启动且云台在跟踪目标 "
                        "(ros2 topic hz /debug 可自查)")
                last_note = now
    except KeyboardInterrupt:
        pass
    finally:
        analyze(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
