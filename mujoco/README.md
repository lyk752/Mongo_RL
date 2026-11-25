# Mongo MuJoCo sim2sim 使用说明

本目录用于将 Mongo 在 Isaac Lab 中训练的平地策略转移到 MuJoCo。

下面的命令均从项目根目录 `/home/inin/workspace/Mongo_RL` 执行。MuJoCo 运行环境不需要 Isaac Sim、Isaac Lab 或 PyTorch；从训练 checkpoint 导出 ONNX 时需要训练环境。目录仍依赖仓库中的模型、网格和共享电机保护函数，迁移到其他机器时应保留完整仓库结构。

## 1. 环境配置与快速启动

### 1.1 安装运行环境

本机已有 `mujoco` Conda 环境；首次安装可执行：

```bash
conda create -n mujoco python=3.10 -y
conda activate mujoco
cd /home/inin/workspace/Mongo_RL
python -m pip install -r mujoco/requirements.txt
```

依赖以 [requirements.txt](requirements.txt) 为准：

| 依赖 | 版本要求 | 用途 |
| --- | --- | --- |
| Python | 推荐 3.10 | 本机独立验证环境 |
| MuJoCo | `3.16.0` | 刚体、接触、闭链约束及窗口 |
| NumPy | `>=1.26,<3` | 控制计算及数据处理 |
| ONNX Runtime | `>=1.20,<2` | CPU Actor 推理 |
| PyYAML | `>=6,<7` | 读取训练／播放配置 |
| Matplotlib | `>=3.8,<4` | 动态测试及评估绘图 |

图形模式需要可用的桌面显示环境。远程终端或无桌面服务器优先使用 `--headless`；键盘控制必须使用图形窗口。

### 1.2 使用当前默认策略

```bash
# 图形窗口：前进 0.4 m/s，高度指令 0.355 m
python mujoco/sim2sim.py --vx 0.4 --height 0.355 --duration 20

# 无界面：零速度指令，观察平衡和漂移
python mujoco/sim2sim.py --headless --height 0.355 --duration 20

# 图形窗口：通过键盘调整指令
python mujoco/keyboard_control.py --height 0.355 --duration 300
```

## 2. 机器人模型来源与路径

### 2.1 文件来源

| 路径，相对项目根目录 | 内容与用途 |
| --- | --- |
| [`model/wl0901.xml`](../model/wl0901.xml) | 仓库提供的原始 Mongo MJCF，包含刚体、关节、质量、惯量和网格引用；是 MuJoCo 模型生成的直接输入 |
| [`model/meshes/`](../model/meshes/) | 机器人 STL 网格；显示和部分碰撞几何复用这些文件 |
| [`model/urdf/`](../model/urdf/) | 机器人 URDF 资产，用于追溯机器人结构；MuJoCo 运行器不直接读取 URDF |
| [`model/usd/Mongo0909.usd`](../model/usd/Mongo0909.usd) | Isaac 侧 Mongo 机器人资产 |
| [`model/usd/Mongo0909_caster.usda`](../model/usd/Mongo0909_caster.usda) | Isaac 侧前球轮物理修正包装层；MuJoCo 不直接加载 USD |
| [`model/readme.md`](../model/readme.md) | 原始模型坐标系、惯量和网格导出说明 |
| [`model/motor_config_24v.json`](../model/motor_config_24v.json) | 新训练与 MuJoCo 默认使用的 24 V 电机参数和控制频率 |
| [`mujoco/models/mongo.xml`](models/mongo.xml) | `build_model.py` 生成的可运行 MuJoCo 机器人模型 |

模型生成关系为：

```text
model/wl0901.xml + model/meshes/ + model/motor_config_24v.json
                            │
                     mujoco/build_model.py
                            ↓
                  mujoco/models/mongo.xml
                            │
          env.yaml + 可选 physics_env0.json
                            ↓
                    本次运行的实际模型
```

`build_model.py` 保留原模型的刚体结构和惯性参数，并加入浮动基座、平地、七个力矩执行器和四个闭链连接约束。驱动轮采用半径 `0.13 m`、半宽 `0.025 m` 的圆柱接触，前球轮采用半径 `0.0303 m` 的球体接触；机器人自碰撞关闭。前球轮关节为被动球关节。

仅在修改原始模型、网格引用、默认电机参数或模型生成逻辑后重新生成：

```bash
python mujoco/build_model.py
```

该命令会重新写入 `mujoco/models/mongo.xml`。单纯切换策略权重无需重新生成机器人模型。运行时会按配置调整物理步长和可映射的动力学参数，并将网格路径解析到本仓库。

### 2.2 配置如何生效

| 配置来源 | 使用条件 | 主要影响 |
| --- | --- | --- |
| `models/mongo.xml` | 所有运行 | 机器人结构、几何和约束 |
| `config.json` + `motor_config_24v.json` | `--zero-action` 且未指定 `--env-config` | 当前默认控制接口、频率和电机参数 |
| `--env-config` 指定的 YAML | 显式指定，或由策略路径推断 | 观测、动作缩放、反馈类型、频率、PD、限矩、armature 等 |
| `deployment_config.json` | 显式指定，或在 ONNX 同目录找到 | 校验策略与配置的接口一致性；不代替 `env.yaml` |
| `--physics-snapshot` 指定的 JSON | 手动启用 | 覆盖某个 Isaac 实际样本的质量、质心、惯量、摩擦、增益、限矩和速度保护参数 |

策略模式必须有对应 YAML，不会在缺失时静默使用 `config.json`。运行器也不会仅因为文件存在就自动加载 `physics_env0.json`，需要显式传入 `--physics-snapshot`。

当前新默认配置为 `mongo_24v_speed_guard_v2`：物理／电机更新 `500 Hz`，策略更新 `50 Hz`，即 `sim.dt=0.002`、`decimation=10`。实际运行频率以所加载 YAML 为准。

| 主动电机 | Kp / Kd | 持续限矩 N·m | 空载转速上限 rad/s | armature kg·m² |
| --- | --- | ---: | ---: | ---: |
| 四个髋部 MG6012-i36，24 V | 40 / 4 | 25 | 4.712389 | 0.120528 |
| 前摆臂 MG5010-i10 | 40 / 2 | 4 | 33.510322 | 0.0085 |
| 两个轮电机 MF9025-16T | 0 / 1 | 2.42 | 74.351026 | 0 |

摆臂参数暂采用 MG5010E-i10 手册参考值。新配置的轮速度目标上限为 `51.31268 rad/s`。开启速度保护时，每个物理步根据真实物理转速，在空载上限的 `90%～100%` 区间逐渐降低同方向驱动力矩，到达上限后禁止继续同向驱动，保留反向制动。该保护是工程近似，尚非实测力矩—转速曲线；外力仍可造成短时超速，不直接裁剪引擎速度。

历史 YAML 缺少 `speed_limit_enabled` 时保留原固定限矩行为。不能通过仅更换 ONNX、保留另一轮的配置来判断新模型效果。

## 3. 导入、归档与切换训练权重

### 3.1 区分 checkpoint 与部署模型

| 文件 | 用途 |
| --- | --- |
| `model_XXXX.pt` | 训练 checkpoint，包含网络和优化器等状态；MuJoCo 不直接加载 |
| `exported/policy.onnx` | MuJoCo 实际执行的 Actor |
| `exported/policy.pt` | Play 导出的 TorchScript Actor，与训练 checkpoint 不同；本运行器不使用 |
| `exported/action_processor.pt` | 导出的动作处理模块；MuJoCo 在 `controller.py` 内实现相同的动作处理，不直接加载它 |
| `exported/deployment_config.json` | 动作顺序、观测维度、频率和速度保护等部署元数据 |
| `exported/physics_env0.json` | 导出时某个 Isaac 环境的实际物理样本，可选择导入 |
| `params/env.yaml` | 与导出策略配套的运行配置 |

推荐每个 checkpoint 使用独立目录，避免多次 Play 导出覆盖同一个 `exported/`：

```text
mujoco/policies/<策略名称>/
├── params/
│   ├── env.yaml                # 此次 Play 的 effective_env.yaml
│   ├── training_env.yaml       # 原训练配置，保留用于追溯
│   └── agent.yaml
├── exported/
│   ├── policy.onnx
│   ├── policy.pt
│   ├── action_processor.pt
│   ├── deployment_config.json
│   └── physics_env0.json
└── play_audit/                 # 此次 Play 的实际配置和短播放记录
```

### 3.2 从指定 checkpoint 导出并归档

**在 Isaac 训练环境中执行：**

```bash
conda activate lab
cd /home/inin/workspace/Mongo_RL

RUN="$PWD/logs/rsl_rl/mongo_flat_direct/2026-10-26_21-35-54"
CHECKPOINT="$RUN/model_1000.pt"
BUNDLE="$PWD/mujoco/policies/flat_2026-10-26_21-35-54_model_1000"
mkdir -p "$BUNDLE/params"

python scripts/rsl_rl/play.py \
  --task Mongo-Flat-Play-v0 --num_envs 1 --headless \
  --checkpoint "$CHECKPOINT" --max_steps 2 \
  --export_dir "$BUNDLE/exported" \
  --plot_dir "$BUNDLE/play_audit"

# 将此次 Play 实际生效的配置交给 MuJoCo。
cp "$BUNDLE/play_audit/effective_env.yaml" "$BUNDLE/params/env.yaml"
cp "$RUN/params/env.yaml" "$BUNDLE/params/training_env.yaml"
cp "$RUN/params/agent.yaml" "$BUNDLE/params/agent.yaml"
```

这里保留 `--plot_dir`，因此实际配置会保存到明确的 `play_audit/` 路径。此命令不要同时加 `--no_plot_export`，否则审计文件会改存到训练目录下的 `play_audits/<时间>/`。


### 3.3 临时切换：使用 `--policy`

**切换回 MuJoCo 环境：**

```bash
conda activate mujoco
cd /home/inin/workspace/Mongo_RL

POLICY="mujoco/policies/flat_2026-10-26_21-35-54_model_1000/exported/policy.onnx"
python mujoco/sim2sim.py --policy "$POLICY" \
  --headless --vx 0.4 --height 0.355 --duration 20

python mujoco/keyboard_control.py --policy "$POLICY" --height 0.355
```

给定 `.../<策略名称>/exported/policy.onnx` 后，运行器会寻找：

- `.../<策略名称>/params/env.yaml`：必需；找不到时要求手动指定。
- `.../<策略名称>/exported/deployment_config.json`：存在时自动校验。

若文件不按上述结构保存，显式传入路径：

```bash
python mujoco/sim2sim.py \
  --policy /path/to/policy.onnx \
  --env-config /path/to/env.yaml \
  --deployment-config /path/to/deployment_config.json \
  --headless --duration 20
```

命令行相对路径以当前终端工作目录为基准。`--policy` 仅覆盖本次运行，不修改默认选择。

### 3.4 修改默认策略：编辑 `policy.json`

确认新部署包完整后，将 [policy.json](policy.json) 改为对应路径，例如：

```json
{
  "policy": "policies/flat_2026-10-26_21-35-54_model_1000/exported/policy.onnx",
  "training_run": "logs/rsl_rl/mongo_flat_direct/2026-10-26_21-35-54",
  "checkpoint": "model_1000.pt"
}
```

`policy` 相对 **`mujoco/` 目录**解析，也可填写绝对路径；`training_run` 和 `checkpoint` 用于记录来源，不参与定位模型。修改后，`sim2sim.py`、`keyboard_control.py` 和 `evaluate_flat.py` 在未指定 `--policy` 时都会使用新默认策略。



### 3.5 导入实际物理参数样本

普通 YAML 模式使用名义参数，不会执行训练中的随机化事件。若要与某一次 Isaac 实际机器人参数进行对照，显式导入对应快照：

```bash
POLICY="mujoco/policies/flat_2026-10-26_21-35-54_model_1000/exported/policy.onnx"
SNAPSHOT="mujoco/policies/flat_2026-10-26_21-35-54_model_1000/exported/physics_env0.json"

python mujoco/sim2sim.py --policy "$POLICY" \
  --physics-snapshot "$SNAPSHOT" \
  --headless --duration 20 --output mujoco/outputs/snapshot_trial

python mujoco/check_physics_snapshot.py \
  --snapshot "$SNAPSHOT" --run mujoco/outputs/snapshot_trial
```

输出目录必须尚不存在，重复试验时换一个名称。快照的时间步长、decimation 和关节集合必须匹配当前配置。快照仅代表一个实际参数样本；数值参数检查通过，不代表两种引擎的接触和运动轨迹完全等价。

## 4. 常用运行方式

### 4.1 固定指令

```bash
# 后退，前摆臂放到当前高度允许的上限
python mujoco/sim2sim.py --headless \
  --vx -0.4 --height 0.355 --front-fraction 1 --duration 20

# 原地左转；去掉 --headless 可查看窗口
python mujoco/sim2sim.py --headless \
  --wz 1.0 --height 0.355 --duration 20

# 不加载策略，只检查模型与 PD 链路
python mujoco/sim2sim.py --zero-action --headless --duration 0.2
```

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--policy` | 指定 ONNX；省略时使用 `policy.json` | 默认部署包 |
| `--env-config` | 配套环境 YAML | 从策略目录推断 |
| `--deployment-config` | 部署接口元数据 | 从 ONNX 同目录推断 |
| `--physics-snapshot` | 实际 Isaac 动力学参数样本 | 不导入 |
| `--reset-snapshot` | 同时恢复一次 Play reset 的动力学、初始姿态／速度、摆臂目标和 Actor 历史 | 不导入 |
| `--vx` / `--vy` | 机身线速度指令，m/s | 0 / 0 |
| `--wz` | 偏航角速度指令，rad/s | 0 |
| `--height` | 高度指令，m | 0.34 |
| `--front-fraction` | 前摆臂在允许区间内的比例，0 抬起、1 放下 | 0 |
| `--duration` | 仿真时长，秒 | 20；键盘模式为 300 |
| `--headless` | 不启动图形窗口 | 默认启动窗口 |
| `--output` | 指定新的输出目录 | `mujoco/outputs/<北京时间>/` |
| `--zero-action` | 不加载策略，使用零动作 | 关闭；不可与 `--policy` 同用 |


### 4.2 键盘控制

```bash
python mujoco/keyboard_control.py --height 0.355 --duration 300
```

点击 MuJoCo 窗口、切换英文输入法后操作：

| 按键 | 功能 | 默认单次变化 |
| --- | --- | --- |
| W / S | 增加／减少前进速度，可调到后退 | 0.1 m/s |
| A / D | 增加左转／右转指令 | 0.2 rad/s |
| I / K | 升高／降低高度目标 | 0.01 m |
| U / J | 放下／抬起前摆臂 | 0.1 比例 |
| X | 所有线速度、偏航速度指令归零，策略继续工作 | — |
| H | 打印帮助和当前指令 | — |
| Esc | 退出并保存记录 | — |

按键逐次累加，松开后保持指令；按 X 停车，长按是否重复取决于窗口事件。速度／高度范围读取配套 YAML，速度范围额外允许零指令；高度按训练配置平滑变化，摆臂按控制器设置限速。

步长通过 `--velocity-step`、`--yaw-step`、`--height-step`、`--arm-step` 修改。键盘模式继承固定指令模式的模型和配置参数，不支持 `--headless`。

### 4.3 与 Isaac Play 一致的动态命令测试

[`dynamic_test.py`](dynamic_test.py) 与 `scripts/rsl_rl/play.py --script dynamic` 共用指令生成器和绘图程序。相同参数和策略周期下，两端使用相同的阶段顺序、采样时刻、正弦周期及高度／速度／偏航幅值。

```bash
# --policy 可省略，省略时仍读取 policy.json。
python mujoco/dynamic_test.py --headless \
  --policy "$POLICY" \
  --script_duration 8 \
  --script_height_center 0.38 --script_height_amp 0.07 \
  --script_vel_max 1.5 --script_yaw_max 4 \
  --output mujoco/outputs/dynamic_trial
```

去掉 `--headless` 可在窗口中观察。脚本支持 `--env-config`、`--deployment-config` 和 `--physics-snapshot`，加载方法与 `sim2sim.py` 一致。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--script_duration` | 8 s | 基础阶段时长，同时决定正弦基础周期 |
| `--script_coupled_duration` | 基础时长的 2 倍 | 最后耦合阶段时长，不改变基础周期 |
| `--script_height_center` | 0.37 m | 中位高度 |
| `--script_height_amp` | 0.08 m | 高度幅值，默认范围 0.29～0.45 m |
| `--script_vel_max` | 0.8 m/s | 前进／后退速度正弦峰值 |
| `--script_yaw_max` | 2 rad/s | 偏航速度正弦峰值 |

这些参数也接受连字符写法，例如 `--script-duration`。总时长由脚本决定，继承的 `--duration`、固定 `--vx/--wz/--height/--front-fraction` 不控制动态序列。测试范围应与所加载策略的训练范围相符。

15 个阶段为：前摆臂完整往返一次；抬臂高度正弦扫描；中／高／低三个高度分别放臂和抬臂直行，共 6 阶段；相同组合原地转向，共 6 阶段；最后抬臂执行线速度、高度和双倍频偏航耦合运动。默认总时长 `14×8+16=128 s`。每阶段按 `max(1, round(阶段时长/策略周期))` 取整，阶段内部从 `t=0` 采样，与 Play 一致。

除普通仿真文件外，输出与 Play 同名的 `tracking_data.csv`、`phases.csv`，以及以下 10 张图：

```text
body_command_tracking.png            # 线速度、偏航、高度、摆臂目标与反馈
body_roll_pitch.png                  # 机身 roll/pitch
front_wheel_arm_diagnostics.png       # 前轮接触力峰值、净空、摆臂误差
all_joint_position.png               # 所有关节位置
all_joint_velocity.png               # 所有关节速度
all_joint_acceleration.png           # 所有关节加速度
policy_actions.png                   # 原始六维策略动作
applied_torque.png                   # 七电机实际下发力矩
normalized_torque_delta.png          # 按限矩归一化的力矩一阶差分
normalized_torque_second_diff.png    # 按限矩归一化的力矩二阶差分
```

另有 `script_config.json` 保存计划阶段、参数和完成标记，`phase_metrics.csv` 保存各阶段速度、高度和摆臂的 MAE、偏差及 P95。`tracking_data.csv` 的 `time_s` 沿用 Play：标记本步指令时间，状态为本步执行后的值；普通 `tracking.csv` 保留物理仿真时间，两者相差一个策略周期。


### 4.4 配对 reset 快照：统一两端初始条件后比较

Play 现在会在播放期间被记录环境的每次 reset 后，等首帧观测生成完毕再保存实际快照。普通播放的自动复位和动态测试的每次相机复位都会记录；`--plot_env_index` 决定监测哪个环境，默认环境 0。即使使用 `--no_plot_export`，仍在播放审计目录保存快照。

```text
<Play 输出目录>/
├── effective_env.yaml
├── global/tracking_data.csv
├── tracking/tracking_data.csv          # --camera both 时产生
└── reset_snapshots/
    ├── index.json                     # reset 编号、相机上下文、仿真步数
    ├── reset_000001.json               # 当次实际动力学及完整初始状态
    ├── reset_000001.policy_input.npy   # reset 后首次实际送入 Actor 的 155 维输入
    ├── reset_000002.json
    └── reset_000002.policy_input.npy
```

每份 JSON 除实际质量、质心、惯量、摩擦、PD 和电机限制外，还包含相对环境原点的根位姿、根速度、全部关节状态、摆臂目标、上一动作和 5 帧 Actor 历史。用 `index.json` 中的 `context` 选择对应相机，不能把 global 的快照用于 tracking 播放比较。

以下示例假定已按第 3 节导出策略，并将 Play 的输出放到 `$BUNDLE/play_audit`：

```bash
AUDIT="$BUNDLE/dynamic_audit"

conda activate lab
python scripts/rsl_rl/play.py \
  --task Mongo-Flat-v0 --checkpoint "$CHECKPOINT" --num_envs 1 --headless \
  --script dynamic --camera both --script_duration 8 \
  --script_height_center 0.38 --script_height_amp 0.07 \
  --script_vel_max 1.5 --script_yaw_max 4 \
  --plot_dir "$AUDIT" --export_dir "$BUNDLE/exported"

conda activate mujoco
RESET="$AUDIT/reset_snapshots/reset_000001.json"
python mujoco/dynamic_test.py --headless \
  --policy "$BUNDLE/exported/policy.onnx" \
  --env-config "$AUDIT/effective_env.yaml" \
  --reset-snapshot "$RESET" \
  --script_duration 8 \
  --script_height_center 0.38 --script_height_amp 0.07 \
  --script_vel_max 1.5 --script_yaw_max 4 \
  --output mujoco/outputs/paired_global

python mujoco/compare_dynamic.py \
  --isaac "$AUDIT/global" \
  --mujoco mujoco/outputs/paired_global \
  --reset-snapshot "$RESET" \
  --output mujoco/outputs/paired_global_comparison
```

Play 必须使用同样的动态参数重新测试；第 3 节的 `--max_steps 2` 普通播放导出只是接口检查，不是上述完整动态数据。`--reset-snapshot` 会同时导入动力学和状态，不再需要单独指定 `--physics-snapshot`；如同时指定，两者必须指向同一文件。只有 `--physics-snapshot` 时仍只导入动力学，不改变默认初始状态。

比较脚本检查：实际参数与快照一致性、根位姿／速度和关节状态恢复误差、首次 Actor 输入、全部阶段和指令时序，以及力矩归一化。输出 `comparison.json`、`phase_metrics.csv` 和 `comparison.png`；对齐检查失败返回非零。参数对齐通过不代表两种物理引擎的接触、摩擦与约束求解轨迹相同。

新数据的归一化力矩一阶／二阶差分均使用每个执行器的实际 `effort_limit`，不使用 PhysX 求解器的 `joint_effort_limits`。两端 CSV 增加 `effort_limit_nm/<关节名>` 共 7 列，当前共 107 列，便于重算核查。速度统计统一为基座质心在机身坐标系中的速度。

为了审计初始输入，MuJoCo 输出 `initial_policy_observation.npy` 和 `initial_state_check.json`。Actor 的初始历史与首帧传感器缓存由快照初始化，首步只更新当前指令字段；后续正常采样物理状态。当前导入支持球关节零角度的 reset，遇到非零球关节初始角度会明确拒绝，避免未经坐标转换直接套用。

### 4.5 批量平地评估与绘图

```bash
# 此脚本仅支持 --policy 和 --output；部署包应遵守第 3 节的目录结构。
python mujoco/evaluate_flat.py \
  --policy "$POLICY" --output mujoco/outputs/flat_sweep_trial

python mujoco/analyze_flat.py mujoco/outputs/flat_sweep_trial
```

评估包含 3 个高度、2 个摆臂位置下的直行、转向、耦合运动和长时间驻停，共 108 个固定工况，另加一轮动态指令测试。输出 `case_metrics.csv`、`dynamic_phases.csv`、`evaluation_summary.json` 及各工况完整记录；绘图输出速度跟踪、驻停漂移和动态响应图片。

当前批量脚本固定使用高度 `0.29/0.355/0.45 m`、线速度最高 `±1.5 m/s` 和偏航速度最高 `±4 rad/s`，不会自动适应新的训练范围。其 `saturation_pct/*` 指标仍按旧力矩上限 `35/15/10 N·m` 计算，分析新 24 V 策略时应结合 `tracking.csv` 与 `resolved_physics.json` 中的实际限矩重新统计。批量脚本当前不提供 `--physics-snapshot` 参数；需要导入快照的单项对照请使用 `sim2sim.py`。

## 5. 主要文件功能

| 文件 | 功能 |
| --- | --- |
| [`sim2sim.py`](sim2sim.py) | 主入口：选择策略、加载配置、推理与仿真、窗口和故障终止、CSV／JSON 输出 |
| [`dynamic_test.py`](dynamic_test.py) | 与 Isaac Play 共用 15 阶段动态指令和绘图，导出同名曲线及逐阶段指标 |
| [`keyboard_control.py`](keyboard_control.py) | 通过 MuJoCo 窗口回调调整运动、高度和摆臂指令 |
| [`controller.py`](controller.py) | 观测拼接、历史帧、动作处理、目标裁剪、前摆臂控制、逐物理步 PD 和速度保护 |
| [`build_model.py`](build_model.py) | 从原始 MJCF 构建独立验证模型，定义关节顺序和默认电机组 |
| [`physics.py`](physics.py) | 映射 YAML 的动力学参数，导入实际物理快照，处理接触摩擦映射 |
| [`feedback.py`](feedback.py) | 根据关节位置差分计算编码器速度，连续轮关节处理跨圈角差 |
| [`config.json`](config.json) | 无策略检查模式的当前默认控制配置 |
| [`policy.json`](policy.json) | 三个运行入口的默认 ONNX 选择 |
| [`evaluate_flat.py`](evaluate_flat.py) | 批量平地工况及动态指令评估 |
| [`analyze_flat.py`](analyze_flat.py) | 生成评估图表，可选与 Isaac 固定速度结果比较 |
| [`reset_state.py`](reset_state.py) | 导入配对初始状态／历史，统一根质心速度统计 |
| [`compare_dynamic.py`](compare_dynamic.py) | 核验配对快照、初始状态、Actor 输入和归一化，生成两引擎对照指标与曲线 |
| [`check_physics_snapshot.py`](check_physics_snapshot.py) | 将保存的 MuJoCo 模型及控制参数与 Isaac 快照逐项核对 |
| [`requirements.txt`](requirements.txt) | 最小运行依赖 |
| [`models/`](models/) | 生成的机器人 MJCF |
| [`policies/`](policies/) | 按 checkpoint 归档的部署包 |
| [`outputs/`](outputs/) | 单次仿真、批量评估及检查结果 |
| [`tests/`](tests/) | 控制接口、键盘、配置兼容性、速度保护等测试 |
| [`../scripts/playback.py`](../scripts/playback.py) | 两端共用的动态命令生成器和曲线导出器，无 Isaac／PyTorch 依赖 |
| [`../source/mongo_world/mongo_world/motor_limits.py`](../source/mongo_world/mongo_world/motor_limits.py) | Isaac 与 MuJoCo 共用的电机速度保护公式；不依赖 Isaac 运行时 |

## 6. 策略接口与仿真边界

Actor 输入为 `[1,155]`：每帧 31 维，5 帧按旧到新堆叠，复位后以首帧填满历史。每帧内容为速度指令 3、高度指令 1、机身角速度 3、投影重力 3、主动关节绝对角度 7、关节速度 7、上一动作 6、摆臂绝对目标 1。默认将驱动轮位置置零，clip/scale 读取 YAML。

观测中的主动关节顺序为：

```text
left_hip_joint, right_hip_joint,
left_parallel_1_joint, right_parallel_1_joint,
front_hip_joint, left_podarthrum_joint, right_podarthrum_joint
```

Actor 输出 `[1,6]`：四个后腿关节的位置动作和两个轮速度动作。新默认前四维乘 `0.5 rad` 并加默认角度，末两维乘 `10 rad/s`；实际缩放及限制从配置读取。前摆臂独立接受高度相关目标，不占 Actor 动作维度。上一动作记录的是滤波／变化率限制后、关节目标裁剪前的值。

`joint_velocity_source=native` 使用引擎速度；`position_delta` 使用每个物理步的位置后向差分，Actor 关节速度与 PD 阻尼使用同一差分反馈。速度保护始终读取引擎真实速度。当前确定性验证器支持低通和动作变化率限制，明确拒绝已启用的观测／动作延迟和噪声，避免误判为已复现训练接口。

MuJoCo 不执行训练的 domain randomization 或随机 reset；默认使用安静的固定装配姿态，根位姿可由 YAML 初始状态覆盖。MuJoCo 与 PhysX 的接触、闭链求解、动态摩擦、恢复系数及初始化存在差异，PhysX 的 `enable_external_forces_every_iteration` 也没有直接对应的 MuJoCo 开关。

## 7. 输出、终止条件与检查

每次运行创建新的输出目录，已有目录不覆盖。

| 输出文件 | 内容 |
| --- | --- |
| `tracking.csv` | 本步命令、仿真后速度／高度／姿态、六维动作、七个主动电机状态与力矩、反馈速度及闭链误差 |
| `summary.json` | 退出状态、时长、跟踪 MAE、最大俯仰、闭链误差、位移、MuJoCo 警告和策略／模型哈希 |
| `resolved_env.json` | 本次读取的配套环境配置 |
| `resolved_physics.json` | 实际映射的动力学、PD、限矩、速度保护、反馈类型和接触映射说明 |
| `model.mjb` | 参数覆盖后的完整编译模型，读取时需匹配 MuJoCo 版本 |
| `model.xml` | 参数可由 MJCF 表达时保存；球关节三个轴参数不同则省略 |
| `source_model.xml` | 参数快照覆盖前的模型，含本机网格绝对路径 |
| `initial_state_check.json` | 导入 reset 快照后的位姿、速度和历史审计 |
| `initial_policy_observation.npy` | 首步实际送入 Actor 的 155 维输入 |
| `parameter_check.json` | 执行 `check_physics_snapshot.py` 后生成的参数核对结果 |

基座相对地面高度低于 `0.15 m`、roll／pitch 超过 `60°`、闭链误差超过 `0.02 m`、出现非有限数值或 MuJoCo 警告时提前停止。正常完成、关闭窗口、按 Esc 的退出码为 0；仿真故障提前停止为 2；参数／加载错误返回非零。

`duration_completed` 只表示完成指定时长，评估运动效果还应检查速度偏差、零指令漂移、姿态、力矩和闭链误差。

可执行以下接口检查：

```bash
python mujoco/sim2sim.py --help
python mujoco/keyboard_control.py --help
python -m unittest discover -s mujoco/tests -v
```
