#!/usr/bin/env python3
"""Control a Mongo ONNX policy using keys in the MuJoCo viewer window."""
from queue import Empty, SimpleQueue

import numpy as np

from sim2sim import parser as sim_parser, positive, run

HELP = """
点击 MuJoCo 窗口后操作（英文输入法）：
  W / S   前进速度 + / -       每次 0.1 m/s
  A / D   左转 / 右转          每次 0.2 rad/s
  I / K   高度目标 + / -       每次 0.01 m
  U / J   摆臂放下 / 抬起      每次 0.1 比例
  X       速度、转向指令归零（策略继续平衡）
  H       显示帮助与当前指令
  Esc     退出并保存记录；也可以关闭窗口
按键逐次累加，松开后保持指令；按 X 停车。按住是否重复取决于窗口按键事件。
高度按训练配置限速，摆臂沿用控制器限速。\n"""


class KeyboardCommands:
    """Queue viewer-thread key events; update commands only on the simulation thread."""
    def __init__(self, args):
        self.args = args
        self.keys = SimpleQueue()
        self.quit_requested = False

    def configure(self, cfg, args):
        ranges = cfg.get('commands', {}).get('ranges', {})
        self.vx_bounds = self._bounds(ranges.get('lin_vel_x', [-1.5, 1.5]), include_zero=True)
        self.wz_bounds = self._bounds(ranges.get('ang_vel_z', [-4., 4.]), include_zero=True)
        self.height_bounds = self._bounds(cfg.get('height_range', [.29, .45]))
        self.command = np.array([args.vx, args.vy, args.wz], dtype=float)
        self.command[0] = np.clip(self.command[0], *self.vx_bounds)
        self.command[2] = np.clip(self.command[2], *self.wz_bounds)
        if not self.height_bounds[0] <= args.height <= self.height_bounds[1]:
            raise ValueError(f'--height must be within training range {self.height_bounds}')
        self.height = self.height_goal = args.height
        self.fraction = args.front_fraction
        self.height_delta = cfg.get('height_command_max_rate_m_s', .03) * cfg.get('sim', {}).get('dt', .005) * cfg.get('decimation', 4)
        if not np.isfinite(self.height_delta) or self.height_delta <= 0:
            raise ValueError('height_command_max_rate_m_s must be finite and positive')
        print(HELP, flush=True)
        print(f'范围：vx={self.vx_bounds} m/s, wz={self.wz_bounds} rad/s, 高度={self.height_bounds} m', flush=True)
        print(f'步长：速度 {args.velocity_step:g} m/s，转向 {args.yaw_step:g} rad/s，'
              f'高度 {args.height_step:g} m，摆臂 {args.arm_step:g}', flush=True)
        self.show()

    @staticmethod
    def _bounds(values, include_zero=False):
        low, high = map(float, values)
        if not np.isfinite([low, high]).all() or low > high:
            raise ValueError(f'Invalid command range: {values}')
        return (min(0., low), max(0., high)) if include_zero else (low, high)

    def key_callback(self, keycode):
        self.keys.put(keycode)

    def show(self):
        print(f'vx={self.command[0]:+.2f} m/s  vy={self.command[1]:+.2f} m/s  '
              f'wz={self.command[2]:+.2f} rad/s  '
              f'高度目标={self.height_goal:.3f} m（当前 {self.height:.3f}）  '
              f'摆臂={self.fraction:.1f}（0 抬起 / 1 放下）', flush=True)

    def sample(self):
        changed = False
        while True:
            try:
                key = self.keys.get_nowait()
            except Empty:
                break
            key = chr(key).upper() if 0 <= key < 128 else key
            if key == 256:  # GLFW_KEY_ESCAPE
                self.quit_requested = True
                break
            if key in ('W', 'S'):
                self.command[0] += self.args.velocity_step * (1 if key == 'W' else -1)
            elif key in ('A', 'D'):
                self.command[2] += self.args.yaw_step * (1 if key == 'A' else -1)
            elif key in ('I', 'K'):
                self.height_goal += self.args.height_step * (1 if key == 'I' else -1)
            elif key in ('U', 'J'):
                self.fraction += self.args.arm_step * (1 if key == 'U' else -1)
            elif key == 'X':
                self.command[:] = 0
            elif key == 'H':
                print(HELP, flush=True)
            else:
                continue
            self.command[0] = np.clip(self.command[0], *self.vx_bounds)
            self.command[2] = np.clip(self.command[2], *self.wz_bounds)
            self.height_goal = float(np.clip(self.height_goal, *self.height_bounds))
            self.fraction = float(np.clip(self.fraction, 0., 1.))
            changed = True
        self.height += float(np.clip(self.height_goal - self.height, -self.height_delta, self.height_delta))
        if changed:
            self.show()
        return self.command.copy(), self.height, self.fraction


def parser():
    p = sim_parser()
    p.description = __doc__
    p.set_defaults(duration=300.)
    p.add_argument('--velocity-step', type=positive, default=.1)
    p.add_argument('--yaw-step', type=positive, default=.2)
    p.add_argument('--height-step', type=positive, default=.01)
    p.add_argument('--arm-step', type=positive, default=.1)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.headless:
        p.error('键盘控制需要图形窗口，请去掉 --headless')
    return run(args, command_source=KeyboardCommands(args))


if __name__ == '__main__':
    raise SystemExit(main())
