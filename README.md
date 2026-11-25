# Mongo_RL

MuJoCo sim2sim 已独立放在 [`mujoco/`](mujoco/README.md)，使用 Conda 环境 `mujoco`。模型适配、ONNX 闭环运行和验证结果见该目录说明及 [验证报告](mujoco/VALIDATION.md)。
本文描述**当前源码的默认配置**。运行时的 CLI / Hydra 覆盖以训练目录内的 `params/env.yaml` 和 `params/agent.yaml` 为准。修改奖励、观测或控制接口后，不能仅凭 checkpoint 文件名判断模型兼容性。

阅读导航：

- [工程概览](#1-工程概览)、[环境与依赖](#2-环境与依赖)、[目录结构](#3-目录结构与文件职责)
- [运行与训练命令](#4-从检查到训练的操作流程)、[机器人与复位](#5-机器人仿真与复位配置)
- [动作空间](#6-动作空间与控制链路)、[观测空间](#7-观测空间)、[任务指令](#8-任务指令)
- [43 项奖励](#9-奖励设计43-项逐项说明)、[网络与 PPO](#10-网络与-ppo-训练配置)、[随机化开关](#11-随机化噪声与延迟的有效状态)
- [训练产物与曲线](#12-训练产物与曲线说明)、[Play 与导出](#13-play动态评估录像与部署导出)、[验证与排查](#14-验证排查与实验维护)

## 1. 工程概览

| 项目 | 默认配置 |
| --- | --- |
| 训练任务 / 播放任务 | `Mongo-Flat-v0` / `Mongo-Flat-Play-v0` |
| 环境类型 | `DirectRLEnv`，无限平面，无地形课程 |
| 默认并行环境 | 32；正式训练示例显式设置 4096 |
| 物理 / 策略频率 | 500 Hz / 50 Hz，`dt=0.002 s`、`decimation=10` |
| 单回合上限 | 20 s，即约 1000 个策略步 |
| 策略动作 | 6 维：4 个后腿位置偏移、2 个驱动轮速度 |
| 前摆臂 | 环境独立下发位置目标，不占策略动作维度 |
| Actor / Critic 输入 | Actor 5 帧 × 31 = 155 维；Critic 单帧 77 维 |
| Actor / Critic 网络 | 独立 MLP，隐藏层 `[256, 128, 64]`，ELU |
| 算法 | 普通 PPO，非 DreamWaQ / HIM / NP3O |
| rollout / 总迭代 / 保存间隔 | 每环境 24 步 / 20000 次 / 每 500 次 |
| 日志根目录 | `logs/rsl_rl/mongo_flat_direct/` |
| 当前启用的随机化 | 质量、质心、摩擦、执行器增益、几何一致的初始状态 |
| 刚体角速度上限 | 100 rad/s |
| 当前关闭的功能 | 观测/动作延迟、传感器/动作噪声、输出力矩噪声、动作低通、动作变化率限制、周期推扰 |

## 2. 环境与依赖

### 2.1 运行前提

需要具备 Isaac Sim、Isaac Lab、Isaac Lab RL / Tasks 和 RSL-RL 的 Python 环境。GPU 物理仿真及录像需要可用的 NVIDIA 驱动、CUDA 和 Isaac Sim 渲染环境。仓库内的 `setup.py` **不会安装完整的 Isaac Sim / Isaac Lab**。

包元数据要求 Python `>=3.10,<3.12`。本机已有 Conda 环境名为 `lab`；下面的版本来自该环境的已安装包元数据，用作复现线索，并非所有版本组合都已完成仿真验证。

| 依赖 | 本机 `lab` 环境版本 | 用途 / 声明 |
| --- | --- | --- |
| Isaac Sim | `5.1.0.0` | 仿真、PhysX、渲染 |
| `isaaclab` | `0.54.4` | 已安装分发包版本；不等同于 Isaac Lab 仓库发布标签 |
| `isaaclab-rl` / `isaaclab-tasks` | `0.5.2` / `0.11.16` | RSL-RL 包装、任务配置工具 |
| `rsl-rl-lib` | `3.0.1` | `OnPolicyRunner`、Actor-Critic、PPO |
| PyTorch / torchvision | `2.7.0+cu128` / `0.22.0+cu128` | 包声明分别为 `>=2.7.0` / `>=0.22.0` |
| NumPy | `1.26.0` | 包声明 `>=1.26.0` |
| Gymnasium | `1.2.1` | 注册环境、录像包装器 |
| Hydra / OmegaConf | `1.3.4` / `2.3.1` | 训练配置覆盖与解析 |
| Matplotlib | `3.10.3` | Play PNG 曲线；包声明 `>=3.8.0` |
| TensorBoard | `2.21.0` | 默认训练日志与曲线查看 |
| ONNX / ONNX Script | `1.22.0` / `0.7.1` | Play 的 ONNX 导出；不在本库 `setup.py` 的依赖列表中 |
| MoviePy / imageio-ffmpeg | `2.2.1` / `0.6.0` | 录像编码，需按实际 Gymnasium 环境补齐 |
| psutil / prettytable | `7.2.2` / `3.3.0` | 运行辅助、环境列表 |


### 2.2 本机安装与环境检查

以下命令从本机项目根目录执行：

```bash
conda activate lab
cd /home/inin/workspace/Mongo_RL

# 在已经配好的 Isaac 环境中安装本项目，保留现有 torch 等依赖版本。
python -m pip install --no-deps -e source/mongo_world -e source/mongo_tasks

# 查看关键依赖及 CUDA 状态；不会启动 Isaac Sim。
python -c "import torch; from importlib.metadata import version; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('RSL-RL:', version('rsl-rl-lib'))"
nvidia-smi

# 查看 CLI；--help 不进入仿真循环。
python scripts/rsl_rl/train.py --help
python scripts/rsl_rl/play.py --help
```

`--no-deps` 适用于已有完整依赖的环境。需要解析本项目声明依赖时，可去掉它；应先确认不会改变 Isaac 环境要求的 PyTorch / NumPy 版本。ONNX 导出或录像出现缺包错误时，再在同一环境中补齐对应依赖。

入口自带源码路径引导，因此可编辑安装不是直接运行脚本的必要条件。模型默认读取 `model/usd/Mongo0909_caster.usda`；移动项目时应保留整个 `model/` 的相对目录关系。可用环境变量选择其他模型：

```bash
export MONGO0909_USD_PATH="/absolute/path/to/compatible_robot.usda"
```

覆盖模型必须保留任务要求的关节、刚体名称和闭链结构。当前路径解析器先尝试该变量，文件不存在时会继续尝试仓库默认模型。

## 3. 目录结构与文件职责

```text
Mongo_RL/
├── README.md
├── LICENSE / THIRD_PARTY_NOTICES.md
├── SOURCE_MANIFEST.json                  # 原始源码哈希及配置来源，不参与运行
├── play.py                               # 根目录入口，转交 scripts/rsl_rl/play.py
├── model/
│   ├── usd/
│   │   ├── Mongo0909.usd                  # 原始转换模型
│   │   ├── Mongo0909_caster.usda          # 默认入口，引用原模型并覆盖前球轮物理属性
│   │   └── urdf/                         # USD 的相对资源依赖
│   ├── urdf/ / meshes/                    # URDF 与网格资源
│   └── scene.xml / wl0901.xml             # 保留的模型资料，RL 不以 XML 为入口
├── mujoco/                              # MuJoCo 模型、ONNX sim2sim、平地验证
├── scripts/
│   ├── mongo_bootstrap.py                # 将本地 source 包加入 sys.path
│   ├── list_envs.py                      # 启动仿真应用并列出 Mongo 任务
│   ├── view_robot.py                     # 零动作或随机动作查看机器人
│   ├── smoke_test.py                     # 实际 reset/step 与有限值检查
│   ├── record_video.sh                   # 从 checkpoint 录制短视频
│   └── rsl_rl/
│       ├── cli_args.py                   # 共享 CLI、版本兼容及 agent 配置覆盖
│       ├── train.py                      # 训练、加载权重、续训、配置快照
│       └── play.py                       # 推理、动态测试、录像、导出及诊断曲线
├── source/
│   ├── mongo_world/
│   │   ├── setup.py
│   │   └── mongo_world/assets/mongo.py    # 模型路径、默认姿态、关节分组、执行器
│   └── mongo_tasks/
│       ├── setup.py
│       └── mongo_tasks/
│           ├── mdp/events.py             # 质量/摩擦/增益等随机化实现
│           └── direct/mongo/
│               ├── agents/rsl_rl_ppo_cfg.py  # 网络、PPO、rollout、保存周期
│               └── flat/
│                   ├── __init__.py        # 两个 Gymnasium 任务 ID 注册
│                   ├── env_cfg.py         # 环境/仿真/指令/奖励参数及随机化开关
│                   ├── env.py             # 动作、观测、奖励、终止、复位的实际执行
│                   ├── cfg_utils.py       # 维度、关节顺序、clip/scale 表
│                   ├── action_pipeline.py # 可选动作低通/slew 及部署处理器
│                   ├── reset_randomization.py # 后腿闭链 IK、复位几何工具
│                   ├── link_clearance.py  # 固定速度评估专用碰撞网格/净空诊断
│                   ├── reward_state.py    # 指令稳态门控、虚拟腿几何角速度历史
│                   └── rewards.py         # 43 项奖励纯 PyTorch 公式，可离线验证
├── tests/                                # 奖励、接口、兼容、PPO、Play 离线测试
├── logs/rsl_rl/                           # 训练/播放产物，运行后生成
└── outputs/                              # Hydra 输出，运行时可能生成
```

两个任务 ID 指向同一个环境类和配置类。`-Play-` 后缀本身不会切换配置；由 `play.py` 显式设置 `env_cfg.play=True` 等字段。日常修改建议按职责定位：机器人增益改 `assets/mongo.py`，奖励权重改 `env_cfg.py`，公式改 `rewards.py`，动作/观测实际拼接改 `env.py`，网络/PPO 改 `rsl_rl_ppo_cfg.py`。

## 4. 从检查到训练的操作流程

### 4.1 环境注册、可视化与仿真检查

```bash
# 需要 Isaac Sim 环境；不是纯离线导入。
python scripts/list_envs.py

# 零动作查看：后腿回到默认位置目标、驱动轮速度目标为零。
python scripts/view_robot.py --task Mongo-Flat-v0 --num_envs 1 --device cuda:0

# 随机动作查看，不加载策略。
python scripts/view_robot.py --task Mongo-Flat-v0 --num_envs 1 --random --device cuda:0

# 16 个环境、256 个控制步的仿真检查。
python scripts/smoke_test.py --headless --device cuda:0 --num_envs 16 --steps 256
```

`smoke_test.py` 检查 6 维动作、155/77 维观测、奖励/观测/关节有限值，并要求不发生数值安全复位。它不能替代策略效果验证；随机动作也不能保证长期站稳。

### 4.2 短训练与正式训练

```bash
# 建议先验证环境、PPO 更新、日志及 checkpoint 写入。
python scripts/rsl_rl/train.py \
  --task Mongo-Flat-v0 --num_envs 16 --max_iterations 2 \
  --device cuda:0 --headless --seed 42 --run_name smoke

# 正式训练；并行数需按实际显存调整。
python scripts/rsl_rl/train.py \
  --task Mongo-Flat-v0 --num_envs 4096 --max_iterations 20000 \
  --device cuda:0 --headless --seed 42 --run_name flat_v1

# 训练时录像：每 2000 个向量环境控制步触发一次，每段 200 步（约 4 s）。
python scripts/rsl_rl/train.py \
  --task Mongo-Flat-v0 --num_envs 4096 --max_iterations 20000 \
  --device cuda:0 --headless --video --video_interval 2000 --video_length 200
```

录像会启用相机和渲染，增加开销。长训练通常可直接使用 checkpoint 另行录制。`--video_interval` 的单位是环境控制步，不是 PPO 迭代数，也不乘并行环境数量。

训练入口支持 Hydra 覆盖，例如：

```bash
python scripts/rsl_rl/train.py \
  --task Mongo-Flat-v0 --num_envs 1024 --max_iterations 1000 --headless \
  env.episode_length_s=20.0 env.commands.ranges.lin_vel_x='[-1.5,1.5]' \
  env.commands.ranges.ang_vel_z='[-4.0,4.0]' \
  agent.algorithm.entropy_coef=0.005
```

CLI 中专门提供的参数会在入口中覆盖相应配置，例如 `--num_envs`、`--max_iterations` 和 `--device`。对长期实验应保存完整命令，并查看最终 YAML。Play 使用 `parse_env_cfg`，不执行训练入口的 Hydra 配置覆盖流程；不要直接将训练的 `env.xxx=...` 参数复制给 Play 并假设它已生效。

### 4.3 加载权重与完整续训

先将下面的变量改为实际存在的 checkpoint 路径；`<run>` 为占位符：

```bash
export MONGO_CKPT="/home/inin/workspace/Mongo_RL/logs/rsl_rl/mongo_flat_direct/<run>/model_500.pt"

# 只加载 Actor/Critic 权重：重新初始化优化器，迭代编号从 0 开始，新建日志目录。
python scripts/rsl_rl/train.py \
  --task Mongo-Flat-v0 --num_envs 4096 --headless \
  --checkpoint "$MONGO_CKPT" --max_iterations 2000 --run_name finetune

# 加载权重、优化器和迭代编号，追加到原 checkpoint 目录。
python scripts/rsl_rl/train.py \
  --task Mongo-Flat-v0 --num_envs 4096 --headless \
  --checkpoint "$MONGO_CKPT" --resume_training --resume_same_log_dir \
  --max_iterations 2000
```

`--resume_training` 要求 `--checkpoint`；`--resume_same_log_dir` 要求二者同时存在。只使用 `--resume_training` 时会新建日志目录。`--resume` / `--load_run` 当前只是共享 CLI 占位参数，不会自动选择 checkpoint。

在本机 RSL-RL 3.0.1 中，`--max_iterations` 传给 `runner.learn(num_learning_iterations=...)`，续训时表示**再训练多少次迭代**，不是最终编号上限。加载编号 500 后，入口从 501 开始。相同日志目录续训会重写 `params/*.yaml`，需要保留历史配置时应先归档或改用新目录。

完整续训并不恢复仿真状态、未完成的 rollout、随机数状态或所有调度器内部状态，不能视为逐步精确重放。加载权重和 Play 都要求相同网络参数形状；当前默认输入为 Actor 155 / Critic 77 维

### 4.4 常用参数速查

| 参数 | 适用入口 | 含义 |
| --- | --- | --- |
| `--task` | train / play / view / smoke | train、view 应显式传入；Play 默认 `Mongo-Flat-Play-v0`，smoke 默认训练任务 |
| `--num_envs` | 同上 | 覆盖并行数；普通 Play 不传则沿用配置 32，动态 Play 不传则设置 1 |
| `--device cuda:0` | 同上 | 仿真与训练/推理设备；支持形式由 AppLauncher 决定 |
| `--headless` | 同上 | 关闭交互窗口；不等同于禁用录像渲染 |
| `--seed` | train / play | 默认继承 agent seed，本机为 42；`-1` 随机生成种子 |
| `--experiment_name` / `--run_name` | train | 修改日志根目录名称 / 时间戳后的运行名后缀 |
| `--logger tensorboard` | train | 默认 TensorBoard；也可选择已配置依赖和账号的 wandb / neptune |
| `--log_project_name` | train | 外部 logger 的项目名 |
| `--clip_actions` | train / play | RSL-RL 包装器的策略动作裁剪，默认继承配置，本机为 `None` |
| `--disable_fabric` | play / view | 禁用 Fabric，改用 USD I/O，供兼容性排查 |
| `--video` | train / play | 启用录像并自动启用相机 |

## 5. 机器人、仿真与复位配置

### 5.1 模型与关节组织

机器人基座为 `wlbody`，左右驱动轮刚体为 `left_wheel`、`right_wheel`，前球轮为 `front_wheel`。左右腿根横向偏移分别为 `+0.09205 m`、`-0.09205 m`，驱动轮半径 `0.13 m`，前球轮半径 `0.0303 m`。

| 分组 | 关节 | 控制方式 |
| --- | --- | --- |
| 后腿主动关节 | `left/right_hip_joint`、`left/right_parallel_1_joint` | 策略输出位置偏移，共 4 个 |
| 前摆臂 | `front_hip_joint` | 独立的位置指令，共 1 个 |
| 驱动轮 | `left/right_podarthrum_joint` | 策略输出轮角速度，共 2 个 |
| 闭链被动关节 | `left/right_knee_joint`、`left/right_parallel_2/3/4_joint` | 无主动驱动力矩，共 8 个 |
| 前球轮 | `front_podarthrum_joint.*` | 球形关节，通常暴露为 3 个自由度，零 PD 驱动 |

模型有 7 个主动电机关节，但策略只输出 6 个动作。前球轮接触和驱动轮接触均允许；机身和其他腿部刚体接触属于非期望接触。

### 5.2 默认姿态与执行器

默认基座位置为 `(0, 0, 0.34) m`，默认关节速度为零。位置偏移和观测中的关节相对角度始终以资产默认姿态为基准，不以每次随机复位姿态为零点。

| 关节 | 默认角度 |
| --- | --- |
| 左/右 hip | `-3°` |
| 左/右 parallel_1 | `30°` |
| 前摆臂 | `28°` |
| 左/右驱动轮 | `0°` |
| 左/右 knee | 约 `19.90°` |
| 左/右 parallel_2 / parallel_4 | 约 `-19.90°` / `-19.92°` |
| 左/右 parallel_3 | 约 `6.81°` |
| 前球轮自由度 | `0°` |

默认前摆臂 `28°` 是资产参考姿态；实际 reset 会按高度条件重新采样前摆臂目标并设置初始角度，不保证从 `28°` 开始。

主动电机使用 `SpeedLimitedPDActuatorCfg`，被动关节使用零力矩的 `IdealPDActuatorCfg`：

24V 电机与部署频率统一读取 [`model/motor_config_24v.json`](model/motor_config_24v.json)。
主动电机使用持续限矩基线；摆臂暂采用 MG5010E-i10 参考参数。配置版本为 `mongo_24v_speed_guard_v2`。
每个 2 ms 物理步按真实物理转速执行速度保护：低于下表速度上限的 90% 时保留完整限矩，
90%–100% 区间将同方向驱动力矩线性降至 0，达到或超过上限后禁止继续同方向驱动；
反向制动力矩仍允许达到持续限矩。90% 起点是工程保护近似，尚非实测力矩—转速曲线。
不直接裁剪引擎速度，外力或碰撞可能造成短时超速。Isaac 与 MuJoCo 共用同一力矩保护函数。
PD 反馈仍按 `joint_velocity_source` 配置，电机速度保护始终使用引擎真实速度。

| 组 | Kp | Kd | 力矩上限 N·m | 速度上限 rad/s | armature |
| --- | ---: | ---: | ---: | ---: | ---: |
| 后腿 `legs_act` | 40 | 4 | 25 | 4.712389 | `930e-7 × 36² = 0.120528` |
| 前摆臂 `front_arm` | 40 | 2 | 4 | 33.510322 | `850e-7 × 10² = 0.0085` |
| 驱动轮 `wheel` | 0 | 1 | 2.42 | 74.351026 | `0` |
| 闭链被动关节 `legs_inact` | 0 | 0.01 | 0 | 100 | `0.0001` |
| 前球轮 `front_caster` | 0 | 0 | 0 | 100 | `0.0001` |

Kp 的单位为 N·m/rad，Kd 为 N·m·s/rad，armature 为转动惯量。驱动轮使用纯速度 PD，名义力矩近似 `tau = Kd × (omega_target - omega)`，再受执行器力矩限制。前球轮名义静摩擦 / 动摩擦 / 黏性摩擦分别为 `0.015 / 0.010 / 0.001`；这些是被动阻力参数，不是策略电机。


### 5.3 仿真与接触

| 参数 | 默认值 / 含义 |
| --- | --- |
| 地面 | `plane`，静/动摩擦 1.0，恢复系数 0，材质 combine 为 multiply |
| 环境间距 | 4 m，`replicate_physics=True` |
| 机器人 | 浮动基座，重力开启，自碰撞关闭 |
| 求解器迭代 | TGS，position 12、velocity 1，`enable_external_forces_every_iteration=True` |
| 刚体阻尼 | 线/角阻尼均为 0 |
| 最大去穿透速度 | 5 m/s |
| PhysX GPU 接触缓冲 | rigid patch `2**20`，rigid contact `2**23` |
| 接触传感器 | 每 0.002 s 更新，历史长度 10，覆盖一个策略周期，记录 air time |
| 相机 | 默认世界视角 `(7.5,7.5,7.5)` 看向原点，1280×720 |

接触缓冲设置用于大批量平面环境，避免接触容量不足。更改并行数、模型接触面或求解器参数后，应重新检查实际仿真行为。


### 5.4 几何一致的随机复位

默认 `reset_randomization_enabled=True`。先采样虚拟腿长和腿角，用闭链 IK 同时计算主动与被动关节角，再结合机身姿态、轮半径和净空确定根高度，避免直接独立采样闭链各关节造成约束冲击。

| 项目 | 采样范围 |
| --- | --- |
| 虚拟腿长 | `[0.16, 0.275] m`，左右在公共样本上各加 `[-0.005,0.005] m` 偏差 |
| 虚拟腿角 | `[-20°,15°]`，左右各加 `[-2°,2°]` 偏差；从机身向下朝后方向计角 |
| 根高度 | 几何推导后限制到 `[0.29,0.45] m` |
| 驱动轮离地净空 | `[0.002,0.008] m` |
| 根 roll / pitch / yaw | `±3°` / `±3°` / `±180°` |
| 本体系线速度 x / y / z | `±0.20 / ±0.05 / 0 m/s` |
| 本体系角速度 x / y / z | `±0.10 / ±0.10 / ±0.25 rad/s` |
| 驱动轮初始角度 | `[-2π,2π] rad` |
| 后腿主动关节速度 | `[-0.50,0.50] rad/s` |
| 驱动轮速度 | `[-10,10] rad/s` |
| 被动闭链及前摆臂速度 | 0，避免人为注入闭链约束冲击 |

左右偏差叠加后，腿长和腿角还会裁剪回表中的主范围。复位后的高度目标和当前高度指令同步到实际根高度，避免回合开始立即产生高度指令阶跃。IK 使用 `acos` 裁剪裕量 `1e-6` 并检查可达性；扩大腿长范围时需同时验证几何可达性。

### 5.5 终止条件

| 条件 | 当前实现 |
| --- | --- |
| 时间上限 | 20 s 左右，作为 `time_out` 返回 |
| 非期望接触 | 当前接触力范数超过 1 N |
| roll / pitch 过大 | 任一绝对角超过 40° |
| 持续判定 | 接触或姿态异常连续存在 20 个控制步才终止，约 0.4 s |
| 数值安全 | 主动关节/基座速度存在非有限值，或主动关节速度 >500 rad/s、基座角速度 >200 rad/s、线速度 >100 m/s，立即复位 |

奖励中的非期望接触阈值是 **3 N，检查接触历史**；终止中的阈值是 **1 N，检查当前接触**，二者不是同一判定。数值异常时所有奖励分项置零，避免异常状态污染训练。


## 6. 动作空间与控制链路

### 6.1 6 维策略动作

| 索引 | 关节 | 动作含义 |
| --- | --- | --- |
| 0 | `left_hip_joint` | 相对默认角度的位置偏移 |
| 1 | `right_hip_joint` | 同上 |
| 2 | `left_parallel_1_joint` | 同上 |
| 3 | `right_parallel_1_joint` | 同上 |
| 4 | `left_podarthrum_joint` | 左驱动轮角速度目标 |
| 5 | `right_podarthrum_joint` | 右驱动轮角速度目标 |

```text
后腿位置目标 q_target = q_default + 0.5 × a_leg       [rad]
驱动轮速度目标 omega_target = 10 × a_wheel           [rad/s]
```

`leg_action_encoding="raw"`，网络输出不经过 tanh。默认 RSL-RL `clip_actions=None`，不将策略输出限制在 `[-1,1]`；`a=1` 表示缩放后的参考量，不是动作空间硬边界。最终目标另行裁剪：

| 后腿关节 | 下限 rad | 上限 rad |
| --- | ---: | ---: |
| 左/右 hip | -3.08923278 | 0.25708700 |
| 左/右 parallel_1 | -0.46408305 | 3.08923278 |

这些是保留机械止挡内约 3° 裕量的工作范围。轮速度目标限幅为 `±51.31268 rad/s`（额定工作速度）；执行器名义空载速度参数为 `74.35103 rad/s`，需区分目标裁剪值与执行器参数。默认采用速度控制，力矩控制分支仅作为可选实现保留。

### 6.2 每步执行顺序

```text
Actor 动作
  → RSL-RL 包装器可选裁剪
  → 非有限值清理
  → 可选低通 / slew（当前关闭）
  → 后腿位置偏移、轮速度映射
  → 机械工作范围 / 轮速度目标裁剪
  → 可选动作噪声、物理步延迟（当前关闭）
  → 500 Hz 执行器和物理仿真，保持 10 个物理步
```

前摆臂位置目标独立写入 `front_hip_joint`。Actor 的“上一步动作”保存 `_actions`，即动作条件处理后、机械目标裁剪前的归一化动作；并不是由实际关节位置或最终裁剪目标反算的动作。

低通关闭时不生效的保留参数为 previous/current 权重 `0.7/0.3`；slew 关闭时不生效的限制为腿每策略步 `0.10`、轮 `0.20`。开启后必须同时核对训练、Play 和部署处理器，避免控制链路不一致。

## 7. 观测空间

### 7.1 Actor：5 帧堆叠，共 155 维

每帧 31 维，按 `[o(t-4), o(t-3), o(t-2), o(t-1), o(t)]` 从旧到新拼接；50 Hz 下最早帧距当前 80 ms。回合复位后用首个有效观测填满该环境的 5 帧，不继承上一回合历史。

关节观测顺序为 `[左hip, 右hip, 左parallel_1, 右parallel_1, 前摆臂, 左轮, 右轮]`。下表为单帧内索引，从 0 开始；角度、速度采用 SI 单位后执行表中的 clip/scale。最新帧位于总输入的 124–154 列。

| 索引 | 维度 | 原始量 | 裁剪 | 乘法缩放 |
| --- | ---: | --- | --- | ---: |
| 0–2 | 3 | 本体系速度指令 `vx,vy,wz` | ±100 | 1 |
| 3 | 1 | 当前下发高度 `height_cmd` | `[0,1] m` | 5 |
| 4–6 | 3 | 本体系角速度 | ±100 rad/s | 0.5 |
| 7–9 | 3 | 本体系投影重力 | ±100 | 1 |
| 10–16 | 7 | 主动关节绝对角度 `joint_pos` | ±100 rad | 1 |
| 17–21 | 5 | 后腿四关节与前摆臂角速度 | ±200 rad/s | 0.1 |
| 22–23 | 2 | 驱动轮角速度 | ±200 rad/s | 0.1 |
| 24–29 | 6 | 上一步归一化策略动作 `_actions` | ±100 | 1 |
| 30 | 1 | 当前下发前摆臂绝对目标角 `front_arm_target_cmd` | ±10 rad | 1 |

轮位置对应索引 15、16 强制置零，保留槽位但不输入绝对转角。关节角度与前摆臂目标均使用 articulation 坐标中的弧度，不减资产默认角度，前摆臂目标也不除以 `leg_action_scale`。动作控制仍使用 `q_target = q_default + 0.5 × action`，与观测表示独立。

Actor 不直接读取机身线速度、真实高度、世界坐标位置或轮里程计位移。当前 `num_obs_hist=5`、`use_frame_stack=True`、`num_privileged_obs_hist=1`。环境创建前会重新解析观测维度，使 Hydra 的历史帧覆盖与 Gym 空间保持一致。这里的手工 clip/scale 与神经网络的运行均值方差归一化是两个概念；后者默认关闭。


### 7.2 Critic：77 维单帧

Critic 采用同序的 31 维**干净当前状态**，再附加本体系机身线速度 3 维、根高度 1 维和 42 维特权块。即使后续开启 Actor 观测噪声/延迟，Critic 的基础状态流仍取当前真实状态。

| 索引 | 内容 | 裁剪 / 缩放 |
| --- | --- | --- |
| 0–30 | Actor 同序的干净基础状态 | 与基础观测相同 |
| 31–33 | 根线速度 `root_lin_vel_b` | ±100，×1 |
| 34 | 真实根高度 | `[-10,10] m`，×5 |

特权块配置容量为 `7×3 + 4 + 2 + 2×3 + 2 + 1 + 2×3 = 42`。**当前延迟全部关闭时，实际拼接会省去两个延迟块，再在特权块末尾补 6 个零**，实际布局为：

| 默认索引 | 维度 | 特权信息 | 裁剪 / 缩放 |
| --- | ---: | --- | --- |
| 35–41 | 7 | 主动关节 stiffness | ±100，×1 |
| 42–48 | 7 | 主动关节 damping | ±100，×1 |
| 49–55 | 7 | 主动关节实际施加力矩，含前摆臂 | ±100 N·m，×0.05 |
| 56–61 | 6 | 左/右轮相对基座的线速度，转到本体系后将 y 置零 | ±100，×1 |
| 62–63 | 2 | 左/右驱动轮接触状态，历史峰值力 >5 N 为 1 | `[0,1]`，×1 |
| 64 | 1 | 基座质量 / 默认质量 | ±10，×1 |
| 65–70 | 6 | 两轮材质的静摩擦、动摩擦、恢复系数 | ±100，×1 |
| 71–76 | 6 | 维度补零 | 0 |

如果观测延迟与动作延迟都开启，索引 56–59 为 4 组观测延迟、60–61 为 2 组动作延迟，单位为物理步；后续轮速度、接触、质量和材质依次后移，最后材质位于 71–76。只开启其中一类时按实际拼接长度移动。

因此切换延迟开关会改变部分 Critic 列的含义，虽然总维度仍可能是 77。不能假定关闭延迟仅仅把原位置的延迟列置零。关闭整个特权块时 Critic 变为 35 维，网络输入也必须匹配。

## 8. 任务指令

### 8.1 速度与高度

| 指令 | 训练范围 | 更新规则 |
| --- | --- | --- |
| `vx` | `[-1.5,1.5] m/s` | 每 4 s 重采样 |
| `vy` | 固定 0 | 同速度生成器 |
| `wz` | `[-4,4] rad/s` | 每 4 s 重采样 |
| 高度目标 `height_goal_cmd` | `[0.29,0.45] m` | 每 4 s 重采样 |
| 当前高度指令 `height_cmd` | 向目标靠近 | 最大 `0.03 m/s`，每策略步最多 `0.0006 m` |

速度生成器有 10% 的静止环境采样比例，无 heading 模式，从首次 rollout 就使用完整速度范围，没有逐步扩大的指令课程。高度采用绝对根高度，当前不通过腿长估计高度，也不启用额外高度观测裁剪。

奖励计算会将 `abs(vx_cmd)<0.1` 和 `abs(wz_cmd)<0.1` 分别按零指令处理；这只改变部分奖励的跟踪目标与静止掩码，不会把 Actor 观测中的小指令全部改写为零。

### 8.2 高度条件前摆臂

下式以高度米值 `h` 输入、角度度值输出：

```text
lower = -15.11°
upper_raw(h) = asin(clamp((1000h - 145.3) / 240, -1, 1)) × 180/π - 27° - 8°
upper(h) = clamp(upper_raw(h), lower, 50°)
front_arm_goal ~ Uniform(lower, upper(h))
```

上限裕量 `front_arm_command_upper_margin_deg=8°`，绝对最大下放角 `front_arm_command_upper_limit_deg=50°`，训练和 Isaac Play 共用。较小角度对应抬起，较大角度对应下放。高度 0.29 / 0.355 / 0.45 m 对应上限约 2.08° / 25.90° / 50°。每 4 s 采样新目标；目标角以最大 `30°/s` 接近 goal，即每控制步最多 `0.6°`，最终物理接口使用弧度。

前摆臂的目标与策略输出相互独立；策略通过观测中的目标角和真实运动状态学习保持平衡。默认没有专门的前摆臂跟踪奖励，前摆臂位置环由独立 PD 电机执行。

## 9. 奖励设计：43 项逐项说明

### 9.1 汇总方式与符号

所有项包括终止项都按以下方式计算：

```text
r_k = weight_k × term_k × step_dt
reward = sum(r_k),  step_dt = 0.02 s
```

正权重奖励更大项值，负权重惩罚更大项值；权重 `0` 或 `-0.0` 当前贡献恒为零，但公式和日志键保留。不存在额外总奖励正值裁剪。非有限奖励会清零；数值安全复位环境的所有项均置零。

下表公式均为**未加权、未乘 dt 的 term**，使用以下符号：

- `q,dq,ddq,tau`：关节位置、速度、加速度、施加力矩；`L` 为 4 个后腿策略关节，`W` 为 2 个驱动轮，`M=L∪W`。
- `a_t`：环境中的归一化策略动作，`Δa=a_t-a_(t-1)`，`Δ²a=a_t-2a_(t-1)+a_(t-2)`。默认条件处理关闭时对应送入环境的动作；每次 reset 清空历史。
- `v`、`omega`：本体系根线/角速度；`pitch` 为机身俯仰角；`gx,gy` 为投影重力 x、y 分量。
- `c_v,c_w`：按 0.1 死区处理后的前向/偏航速度目标；`I_v=1[abs(vx_cmd)<0.1]`、`I_w=1[abs(wz_cmd)<0.1]`。
- `e_v=c_v-vx×cos(pitch)`，`e_w=c_w-omega_z`，`e_h=h-height_cmd`。
- `E_v=clip(e_v,±1.0)²`、`E_w=clip(e_w,±0.8)²`、`E_h=clip(e_h,±0.15)²`。仅指数跟踪核使用这些裁剪误差；平方惩罚使用未裁剪误差。
- `dx,dz`：左右驱动轮中心在本体系中的 x、z 坐标之差；`F_i` 为驱动轮接触历史峰值力范数。
- `C_both=1[两个驱动轮的 F_i 均>5 N]`，`B=2×exp(-vx_cmd²/3)+2`。

#### A. 终止、关节、力矩与能耗：9 项

| 奖励键 | 权重 | 未加权公式 | 作用与当前状态 |
| --- | ---: | --- | --- |
| `termination` | -200 | `1[故障终止]` | 惩罚失败；单次有效故障贡献 `-200×0.02=-4`。纯超时不计入 |
| `leg_joint_acc` | -5e-7 | `sum_L(ddq²)` | 抑制后腿高频加速度，不含前摆臂与被动关节 |
| `leg_joint_vel` | -0.005 | `sum_L(dq²)` | 抑制后腿过快运动 |
| `leg_joint_pair_pos_diff` | 0 | 左右 hip、parallel_1 角差 wrap 到 `[-π,π)` 后平方求和 | 对称姿态正则，当前关闭 |
| `joint_torque` | -0.0001 | `sum_M(tau²)` | 限制 6 个策略电机的力矩，不包含直接控制的前摆臂 |
| `wheel_acc` | -1e-8 | `sum_W(ddq²)` | 减少驱动轮加速度 |
| `wheel_vel` | -1e-5 | `sum_W(dq²)` | 限制过大轮速 |
| `wheel_power` | -0.0001 | `sum_W(max(tau×dq,0))` | 惩罚正机械功率，不对负功率加正向奖励 |
| `wheel_air_spin` | 0 | `sum_W(dq²×1[F_i≤5 N])` | 离地空转惩罚，当前关闭 |

#### B. 基座运动、动作平滑与静止轮速：6 项

| 奖励键 | 权重 | 未加权公式 | 作用 |
| --- | ---: | --- | --- |
| `lin_vel_z` | -0.5 | `vz²` | 抑制机身垂向跳动 |
| `ang_vel_xy` | -0.05 | `omega_x²+omega_y²` | 抑制滚转/俯仰角速度 |
| `action_rate` | -0.01 | `sum_6(Δa²)` | 约束相邻策略步动作变化 |
| `action_smoothness_leg` | -0.05 | `sum_前4(Δ²a²)` | 抑制后腿动作二阶变化 |
| `action_smoothness_wheel` | -0.01 | `sum_后2(Δ²a²)` | 抑制轮速动作二阶变化 |
| `stand_still_wheel_vel` | -0.10 | `I_zero×sum_W(max(abs(dq)-0.3,0)²)` | 仅全部速度指令严格为 0 时，抑制超过平衡调整余量的实际轮速；权重强度为原 -0.02 的 5 倍 |

动作平滑项使用归一化动作，不是物理关节目标、力矩或关节加速度。后腿和轮的缩放及电机动力学不同，应结合 Play 的实际关节和力矩曲线判断振荡。

`I_zero=1[vx_cmd==0 且 vy_cmd==0 且 wz_cmd==0]`，使用未经奖励死区处理的原始指令。当前 vy 指令固定为 0，因此主要由 vx、wz 同时为 0 触发；即使是很小的非零指令也不触发，原地转向不触发。该项与 `stand_still_deadzone_enabled` 和 0.1 指令死区无关，也不要求高度或摆臂指令保持不变。

`stand_still_wheel_vel_deadzone_rad_s=0.3` 是实际轮关节角速度的允许余量，单位 rad/s；在余量内此项为 0。可以设为 0 以惩罚所有非零实际轮速。权重 `env.rewards.stand_still_wheel_vel=-0.10`，同样乘 `step_dt=0.02`；例如零指令下两轮实际角速度分别为 1.3、-2.3 rad/s，此项每步贡献 -0.01。日志自动记录 `Reward/stand_still_wheel_vel` 和 `Episode/Reward/stand_still_wheel_vel`。

#### C. 姿态稳定：6 项

| 奖励键 | 权重 | 未加权公式 | 作用与当前状态 |
| --- | ---: | --- | --- |
| `flat_orientation_y` | 0 | `(2×gx)²` | 固定俯仰方向惩罚，当前关闭 |
| `flat_orientation_y_v` | -2 | `(B×gx)²` | 速度相关的俯仰方向惩罚；低速 B 更大 |
| `flat_orientation_y_exp` | 1 | `exp(-gx²/0.02)` | 机身俯仰接近水平时给奖励 |
| `flat_orientation_x` | 0 | `(4×gy)²` | 固定滚转方向惩罚，当前关闭 |
| `flat_orientation_x_v` | -2 | `(B×gy)²` | 速度相关的滚转方向惩罚 |
| `flat_orientation_x_exp` | 1 | `exp(-gy²/0.01)` | 机身滚转接近水平时给奖励 |

这里直接使用投影重力分量，不把欧拉角代入公式。键名中的 `x/y` 对应绕轴姿态含义，实际分别使用 `gy/gx`。B 使用原始 `vx_cmd²`。

#### D. 速度跟踪与静止：7 项

| 奖励键 | 权重 | 未加权公式 | 作用与当前状态 |
| --- | ---: | --- | --- |
| `track_lin_vel_xy` | 1.5 | `exp(-E_v/0.5)` | 前向速度跟踪；保留 xy 键名但实际只跟踪 vx |
| `track_lin_vel_xy_tight` | 0 | `exp(-E_v/0.1)` | 更窄速度指数核，当前关闭 |
| `track_lin_vel_xy_square` | -1 | `(0.25×e_v)²` | 平方速度误差，未使用误差裁剪 |
| `track_ang_vel_z` | 1 | `exp(-E_w/0.25)` | 偏航角速度跟踪 |
| `track_ang_vel_z_square` | -1 | `(0.5×e_w)²` | 未裁剪偏航误差平方 |
| `stand_still_lin_vel` | -1 | `(abs(vx)+abs(vy))×I_v` | 前向零指令时的平面漂移 L1 惩罚；原地转向时也可能生效 |
| `stand_still` | 0 | `(vx²+vy²)×I_v+omega_z²×I_w` | 平面速度与偏航静止正则，当前关闭 |

指数核分母直接为表中数值，不再将其平方。不要因参数名含 `sigma` 就替换成其他高斯核定义。

#### E. 高度跟踪：5 项

| 奖励键 | 权重 | 未加权公式 | 作用与当前状态 |
| --- | ---: | --- | --- |
| `track_height_exp` | 0 | `exp(-E_h/0.025)` | 中等宽度高度核，当前关闭 |
| `track_height_exp_soft` | 0 | `exp(-E_h/0.1)` | 宽松高度核，当前关闭 |
| `track_height_exp_tight` | 1 | `exp(-E_h/0.001)` | 当前有效的紧高度指数奖励 |
| `track_height_square` | -1 | `(10×e_h)²` | 高度误差平方；例如误差 0.1 m 时 term=1、单步贡献 -0.02 |
| `track_height_exp_both_wheels_contact` | 0 | `exp(-E_h/0.1)×C_both` | 两轮接触时奖励高度跟踪，当前关闭 |

#### F. 左右轮位置一致性与接触：5 项

| 奖励键 | 权重 | 未加权公式 | 作用与当前状态 |
| --- | ---: | --- | --- |
| `no_fork` | -1 | `1[abs(dx)>0.05]` | 惩罚左右轮前后错位超过 5 cm |
| `no_fork_square` | -1 | `(5×dx)²` | 连续惩罚左右轮前后错位 |
| `no_fork_exp` | 0 | `1-exp(-max(abs(dx)-0.05,0)/0.0001)` | 超阈值后的饱和惩罚，当前关闭 |
| `no_fork_z_exp` | 0 | `1-exp(-max(abs(dz)-0.05,0)/0.0001)` | 左右轮高度错位惩罚，当前关闭；未单独配置 z 参数时回退到 x 参数 |
| `undesired_contact` | -2 | `1[任一非允许刚体在接触历史中受力>3 N]` | 限制机身/腿部触地；允许左右驱动轮和前球轮接触 |

#### G. 稳态虚拟腿角速度与机身静止：4 项

| 奖励键 | 权重 | 未加权公式 | 作用 |
| --- | ---: | --- | --- |
| `virtual_leg_pitch_rate` | -0.1 | `g×sum_左右(max(abs(phi_dot)-0.1,0)/1.0)²` | 指令稳态时抑制两腿前后摆动，允许保留固定倾角；较初始 -0.05 增强一倍 |
| `stand_still_body_lin_vel` | -0.5 | `I_linear_zero×s_stop×[(vx/0.1)²+(vy/0.1)²]` | vx/vy 指令严格为零即启用，允许 wz 非零，约束原地转向平移 |
| `stand_still_body_ang_vel` | -0.1 | `I_zero×s_stop×(omega_z/0.2)²` | 直接约束零指令时的缓慢偏航 |
| `stand_still_position` | **0（关闭）** | `I_zero×g_position×[max(norm(p_xy-anchor_xy)-0.01,0)/0.05]²` | 保留诊断，默认不参与奖励 |

虚拟腿定义与固定速度测试一致：取真实轮中心相对根原点的位置，转到机身坐标系，`phi=atan2(-p_x,-p_z)`；在 50 Hz 控制步之间计算 `phi_dot=wrap(phi_t-phi_(t-1))/0.02`。使用位置差分，不依赖关节速度反馈；左右分别平方求和，避免反相摆动相互抵消。局部复位只清对应环境的历史，首步导数为零。

指令稳态判定每控制步执行一次，在下一步指令调度之前取样：

| 判定量 | 判为变化的绝对变化率阈值 |
| --- | ---: |
| `vx_cmd`、`vy_cmd` | `>0.05 m/s²` |
| `wz_cmd` | `>0.1 rad/s²` |

稳态只看线速度指令 `vx_cmd/vy_cmd` 和角速度指令 `wz_cmd` 的变化。任一变化率超过上表阈值或环境复位时清零安静时长；低于阈值的小变化视为安静。安静前 `command_steady_delay_s=0.4 s` 门控为 0，其后 `command_steady_ramp_s=0.6 s` 使用 `smoothstep(t)=t²(3-2t)` 从 0 恢复到 1。高度、前摆臂目标和实际机身运动不参与判定；因此速度指令保持稳定时，升降机身或摆臂不会放松摆腿约束。姿态项全阶段保持完整权重。恒定零速度也属于稳态。所有相关奖励共用该门控，保留的 `g_motion` 接口与日志名是 `g` 的别名。

`I_linear_zero` 要求原始 vx/vy 指令严格为零，不检查 wz；`I_zero` 要求 vx/vy/wz 全为零。机身平移成本在转向时启用，机身偏航、轮速静止和位移成本仍只在三者都为零时启用。机身速度项保留 `s_stop=0.25+0.75×g_motion`，停车或转向指令变化时保留 25%，之后恢复满额，高度/摆臂变化不能关闭它们。

位置项独立记录连续零指令时长：`stand_still_position_delay_s=0.6 s` 内不扣位移分；延时结束时记录当前世界坐标 XY 为停车锚点，因此不追罚减速阶段位移。之后用 `stand_still_position_ramp_s=0.4 s` 的 smoothstep 渐增至完整权重。锚点一旦记录便保持固定，不根据实际速度或持续漂移更新；任一原始速度指令非零立即关闭并清除有效锚点，重新停车重新计时。局部复位只清对应环境，改变高度、摆臂或实际偏航不重设锚点，Z 高度变化不计入位移。默认容差 `stand_still_position_tolerance_m=0.01 m`，尺度 `stand_still_position_scale_m=0.05 m`；完整权重下总位移 1 cm 内不扣分，6 / 11 / 21 cm 每 0.02 s 步分别扣 0.01 / 0.04 / 0.16。参数均可配置。

当前位置奖励权重为 0：上面的公式与锚点计时仅用于兼容和诊断，实际位置奖励恒为 0。


#### H. 前轮冲击：1 项

| 奖励键 | 权重 | 未加权公式 | 作用 |
| --- | ---: | --- | --- |
| `front_wheel_impact` | -1 | `max((F_peak-50)/100,0)²` | 允许轻微接触，对较大峰值接触力施加平方惩罚 |

`F_peak` 是前球轮在当前控制周期全部 10 个物理子步中的接触合力模长最大值，单位 N。接触传感器以 0.002 s 更新，历史长度为 10。阈值 `front_wheel_impact_force_threshold_n=50`、尺度 `front_wheel_impact_force_scale_n=100` 均可调，50 N 是初始训练阈值。接触力 ≤50 N 不扣分；100 / 150 / 500 / 1100 N 在 0.02 s 控制步分别扣 0.005 / 0.02 / 0.405 / 2.205。所有指令与阶段均生效，不裁剪大力成本，也不新增轻触终止。持续大载荷同样受罚；这是峰值力代理，并非接触冲量积分或力变化率。

记录 `Reward/front_wheel_impact` 和 `Metrics/front_wheel_contact_peak_mean_n`、`Metrics/front_wheel_contact_peak_max_n`、`Metrics/front_wheel_impact_fraction`（超过阈值环境比例）。这些是控制步统计，TensorBoard 聚合后的 max 曲线不是整段训练绝对最大值。


## 10. 网络与 PPO 训练配置

### 10.1 非对称 Actor-Critic

```text
Actor:  policy(155 = 5 × 31) → Linear(256) → ELU → Linear(128) → ELU → Linear(64) → ELU → Linear(6)
Critic: critic(77) → Linear(256) → ELU → Linear(128) → ELU → Linear(64) → ELU → Linear(1)
```

两条网络独立，不使用循环网络、历史编码器或地形编码器。Actor 输出 6 维高斯策略的均值，初始探索标准差为 1.0，由 PPO 学习；推理使用确定性均值动作。Critic 利用仿真特权状态估计价值，部署只需 Actor。

`empirical_normalization=False`；配置兼容层在需要时将 Actor/Critic 的 observation normalization 设为 False。Actor 输入的手工尺度仍然必须由环境或部署控制器复现。

### 10.2 参数表

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `num_steps_per_env` | 24 | 每次更新前，每环境采集 24 个控制步，约 0.48 s |
| `max_iterations` | 20000 | 默认训练更新次数 |
| `save_interval` | 500 | checkpoint 保存间隔 |
| `learning_rate` | 1e-4 | 固定学习率 |
| optimizer | Adam | 本机 RSL-RL 普通 PPO 实现使用的优化器 |
| `schedule` / `desired_kl` | fixed / 0.01 | 默认固定学习率；desired_kl 在 fixed 模式不调节学习率 |
| `num_learning_epochs` | 5 | 每个 rollout 重复优化轮数 |
| `num_mini_batches` | 4 | 每轮拆分为 4 个 minibatch |
| `clip_param` | 0.2 | PPO 概率比裁剪 |
| `value_loss_coef` | 4.0 | 价值损失系数 |
| `use_clipped_value_loss` | True | 使用裁剪后的价值损失 |
| `entropy_coef` | 0.005 | 鼓励策略探索 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE 参数 |
| `max_grad_norm` | 1.0 | 梯度范数裁剪 |
| seed / device / logger | 本机继承 42 / cuda:0 / tensorboard | 可由 CLI 覆盖 |

rollout 后由 Critic 和 GAE 计算优势与回报目标，再以 PPO 概率比裁剪限制更新幅度。当前优化目标为 `surrogate_loss + 4.0×value_loss - 0.005×entropy`；策略损失、价值损失和探索熵共同影响更新。

4096 个环境时，每迭代采样 `4096×24=98304` 条 transition，每 minibatch 约 24576 条，每迭代共 20 次 minibatch 更新。20000 次迭代理论采样约 19.66 亿条 transition；这不是墙钟时间估算。

默认使用 `schedule=fixed`、学习率 `1e-4`，防止旧 adaptive 调度将学习率升至 `1e-2`。完整续训恢复优化器后，也会重新应用配置中的固定学习率；仅加载权重微调则使用新优化器。显式切回 adaptive 时，仍使用本机 RSL-RL 的 `[1e-5,1e-2]` 边界。

## 11. 随机化、噪声与延迟的有效状态

配置文件中后面赋值为 `None` 的 EventTerm 不会执行。不能仅看到前面的参数定义就判断功能已经开启。

| 功能 | 当前状态 | 参数 |
| --- | --- | --- |
| 基座质量 | startup 开启 | 默认质量乘 `[0.9,1.3]` |
| 腿部 / 轮质量 | startup 开启 | 默认质量乘 `[0.9,1.1]` |
| 基座质心偏移 | startup 开启 | x ±0.04 m，y/z ±0.02 m |
| 基座材质 | startup 开启 | 静/动摩擦 `[0.01,0.1]`，恢复 `[0.02,0.2]`，64 buckets |
| 轮材质 | startup 开启 | 静摩擦 `[0.5,1.2]`，动摩擦 `[0.4,1.0]`，恢复 `[0.02,0.2]`，64 buckets |
| 主动腿关节摩擦 | startup 开启 | 静/动摩擦各加 `[0.25,1.0]`，黏性摩擦加 `[0.05,0.2]` |
| 驱动轮关节摩擦 | startup 开启 | 静摩擦加 `[0.05,0.25]`，黏性摩擦加 `[0,0.01]` |
| 被动闭链摩擦 | startup 开启 | 静摩擦加 `[0.05,0.1]`，黏性摩擦加 `[0.01,0.025]` |
| 前球轮摩擦 | startup 开启 | 名义静/动/黏性摩擦分别乘 `[0.67,1.5]` |
| 执行器 stiffness / damping | reset 开启 | 乘 `[0.75,1.25]`，事件最小重采样间隔计数 720，非每次 reset 必定重采样 |
| 机身/关节初始状态 | reset 开启 | 见几何复位表 |
| 基座惯量 | 关闭 | 保留 `[0.8,1.2]` scale 定义，但最终为 None |
| 电机输出比例/噪声 | 关闭 | `leg_effort_noise=None`、`wheel_effort_noise=None` |
| 周期速度推扰 | 关闭 | 保留每 5–10 s、x/y 速度 ±0.25 m/s 定义 |
| 周期外力/力矩 | 关闭 | 保留每 5–10 s、各轴力 ±10 N、力矩 ±1 N·m 定义 |
| 观测延迟 | 关闭 | 保留 4 组 `[3,10]` 物理步，即 6–20 ms，缓冲长度 10 |
| 动作延迟 | 关闭 | 保留腿/轮 `[5,5]` 物理步，即 10 ms；当前有效值为 0 |
| 传感器 / 动作噪声 | 关闭 | `self_obs_noise_cfg=None`、`self_act_noise_cfg=None` |
| 低通 / slew | 关闭 | 见动作空间章节 |

PPO 训练时的随机策略探索仍然开启，它不属于传感器或执行器噪声。Play 默认也会创建这些已启用的质量、摩擦、增益和初始状态随机化；普通播放不是自动切换到完全确定的名义机器人。

## 12. 训练产物与曲线说明

### 12.1 输出目录

```text
logs/rsl_rl/mongo_flat_direct/<YYYY-MM-DD_HH-MM-SS>[_run_name]/
├── params/
│   ├── env.yaml                          # 生效的环境配置，含 launch_checkpoint 来源
│   └── agent.yaml                        # 生效的网络、算法、runner 配置
├── events.out.tfevents.*                 # 默认 TensorBoard 标量曲线
├── model_0.pt / model_500.pt / ...        # 周期 checkpoint
├── git/                                  # RSL-RL 支持且检测到仓库时的代码状态信息
├── videos/train/                         # 仅 --video 训练时生成
├── videos/play/                          # 后续 Play 录像
├── exported/                             # 后续 Play 导出
└── play_plots/<timestamp>/               # 后续 Play CSV / PNG
```

训练本身默认保存 **TensorBoard event 数据，不自动生成奖励 CSV 或 PNG**。下面的训练曲线名称按本库环境与本机 RSL-RL 3.0.1 说明；其他版本 logger 可能调整名称。

```bash
tensorboard --logdir logs/rsl_rl/mongo_flat_direct --host 127.0.0.1 --port 6006
# 浏览器访问 http://127.0.0.1:6006
```

### 12.2 总体训练与优化曲线

| TensorBoard tag | 含义 | 读取建议 |
| --- | --- | --- |
| `Train/mean_reward` | 最近已结束回合的累计总 reward 均值，runner 缓冲上限 100 回合 | 同时受跟踪、存活时长及惩罚影响，不等于单步奖励 |
| `Train/mean_episode_length` | 已结束回合长度，单位为控制步 | 默认上限约 1000 步；下降时查看故障终止 |
| `Train/mean_reward/time` | 与 mean_reward 同值，以训练累计时间为横轴 | 默认 TensorBoard 下便于比较墙钟效率 |
| `Train/mean_episode_length/time` | 与 mean_episode_length 同值，以累计时间为横轴 | 初始化时随机化回合计数，初期可能短于正常回合 |
| `Loss/value_function` | Critic 价值回归损失 | 与奖励尺度有关，不能要求它跨实验绝对可比 |
| `Loss/surrogate` | PPO 裁剪策略目标损失 | 不应简单当作越小越好的监督学习误差 |
| `Loss/entropy` | 本机 PPO 返回的策略熵统计 | 虽位于 Loss 命名空间，它记录的是熵值 |
| `Loss/learning_rate` | 当前自适应学习率 | 结合策略变化、损失和效果判断更新强度 |
| `Policy/mean_noise_std` | 6 维动作分布标准差的均值 | 观察探索强度；不代表真实电机噪声 |
| `Perf/total_fps` | 所有环境合计 transition/s | 含采样和学习耗时，不是单机器人控制频率 |
| `Perf/collection time` | 每次 rollout 收集耗时，秒；tag 中有空格 | 判断物理仿真/渲染开销 |
| `Perf/learning_time` | 每次 PPO 更新耗时，秒 | 判断网络学习开销 |

默认没有专门保存 KL、clip fraction、梯度范数或“成功率”曲线；需要这些指标时应扩展 runner，而不是从现有 tag 推断。

### 12.3 奖励分解、误差与复位曲线

| tag / 前缀 | 数据定义 | 读取建议 |
| --- | --- | --- |
| `Reward/<奖励键>` | 当前控制步各环境的 `weight×term×0.02` 均值，再由 runner 汇总 | 43 项均保留；零权重项应为零，负惩罚靠近零通常表示该成本减小 |
| `Reward/total` | 当前控制步总奖励的环境均值 | 与所有 Reward 分项同尺度 |
| `Episode/Reward/<奖励键>` | reset 环境的该回合累计分项 / **配置回合时长 20 s** | 不是除以该回合实际存活时长，提前失败会影响比较 |
| `Episode/reward_total` | 同上，总累计奖励 / 20 s | 不等于 `Train/mean_reward`，二者口径不同 |
| `Metrics/height_abs_error_m` | 真实高度与当前下发高度指令的绝对误差均值 | 单位 m，使用平滑后的目标，不是尚未到达的 height_goal |
| `Metrics/lin_vel_abs_error_m_s` | `abs(vx_cmd-vx×cos(pitch))` 均值 | 单位 m/s；使用原始 vx 指令，与奖励死区处理存在差别 |
| `Metrics/command_steady_gate` | 仅按 vx/vy/wz 指令变化判定的稳态门控，环境均值 | 0 为过渡、1 为完全稳态；用于虚拟腿角速和静止项，姿态项不使用门控 |
| `Metrics/motion_steady_gate` | 与 `command_steady_gate` 相同 | 保留原日志名，使用同一个速度指令门控 |
| `Metrics/virtual_leg_pitch_rate_abs_rad_s` | 全环境左右虚拟腿角速度绝对值均值 | rad/s，位置几何差分；评估稳态振荡需再按指令阶段筛选 |
| `Metrics/zero_command_fraction` | 原始 vx/vy/wz 同时严格为零的环境比例 | 零指令速度指标需与该采样比例一起看 |
| `Metrics/zero_command_body_speed_m_s` | 仅零指令环境的平面速度范数均值 | 无零指令环境时记 0，不能解释成静止达标 |
| `Metrics/zero_linear_command_body_speed_m_s` | vx/vy 严格零指令环境的机身平面速度均值 | 包含原地转向 |
| `Metrics/stand_still_position_error_m` | 已建立停车锚点环境的 XY 距离均值 | 无锚点环境记 0；包括 1 cm 容差内误差 |
| `Metrics/stand_still_position_active_fraction` | 停车位移诊断门控大于零的环境比例 | 默认奖励权重为 0，门控非零不代表扣分 |
| `Metrics/stand_still_position_gate` | 停车位移门控的全环境均值 | 0–1 |
| `Metrics/zero_command_yaw_speed_rad_s` | 仅零指令环境的偏航角速度绝对值均值 | 无零指令环境时记 0；包含停车过渡和稳态 |
| `Metrics/front_arm_position_error_deg` | 前摆臂实际角与当前下发目标的绝对误差均值 | 单位度，用于判断独立 PD 跟踪 |
| `Episode/Reset/terminate` | 本次 reset 中故障终止环境个数 | 不是概率；并行数和 reset 频率变化会改变曲线尺度 |
| `Episode/Reset/time_out` | 本次 reset 中超时环境个数 | 稳定策略通常更多回合到达时间上限 |
| `Reset/root_height_m` | 被复位环境采样根高度均值 | 检查 reset 分布是否正确 |
| `Reset/mean_leg_length_m` | 初始虚拟腿长均值 | 检查几何采样 |
| `Reset/mean_abs_leg_angle_deg` | 初始虚拟腿角绝对值均值 | 检查姿态分布 |
| `Reset/mean_abs_roll_deg` / `Reset/mean_abs_pitch_deg` | 初始机身角绝对值均值 | 观察复位姿态随机化 |
| `Reset/mean_lin_speed_m_s` | 初始机身线速度范数均值 | 检查初始速度扰动 |

每控制步新建日志字典，避免 RSL-RL rollout 引用同一字典以及旧复位指标残留。reset 曲线只在复位时产生。日志数值经过环境均值与 runner 聚合，不能把某个计数 tag 直接解释成全实验失败率。当前训练日志没有前轮净空、前轮受力、每关节位置/速度/加速度和力矩时序；这些详细诊断由 Play 单独导出。

建议先联合观察 `Train/mean_reward`、回合长度、故障/超时计数与跟踪误差，再用 Reward 分解定位主要成本。总奖励提高但高度误差、姿态抖动或力矩峰值恶化时，应通过播放曲线进一步判断。

## 13. Play、动态评估、录像与部署导出

Play 会读取 checkpoint 同目录下 `params/env.yaml` 中的
`sim.physx.enable_external_forces_every_iteration`。`--external_force_iterations auto`
为默认：有记录时按记录恢复；旧快照缺少该字段时关闭此选项以兼容旧训练。

每次 Play 在曲线输出根目录保存 `effective_env.yaml`、`effective_agent.yaml` 和 `play_config.json`，
记录实际配置、checkpoint、启动参数及 PhysX 选项来源。
关闭曲线导出时改存到 checkpoint 目录的 `play_audits/`。
可用 `--export_dir <目录>` 分开保存不同 checkpoint 的策略导出。

### 13.1 播放

```bash
python scripts/rsl_rl/play.py \
 --task Mongo-Flat-v0 \
  --checkpoint /model_3000.pt \
  --script dynamic --script_duration 8 \
  --script_height_center 0.38 --script_height_amp 0.07 \
  --script_vel_max 1.5 --script_yaw_max 4 \
  --camera both --headless --video
```

动态模式禁用随机速度重采样与静止/heading 掩码，逐策略步写入命令，并刷新 Actor 当前观测中的指令槽位，避免新命令滞后一拍。每个相机 pass 开头 reset 一次，15 个阶段连续执行；不同相机 pass 不保证具有完全相同的随机初始状态。默认每个视角 128 s，`--camera both` 执行两遍，总计 256 s 仿真时间。

刷新后的最新帧同时写回环境内部历史，保证后续历史包含策略实际使用过的指令。动态速度脚本使用对称正弦，默认覆盖 `[-0.8,0.8] m/s`；当前训练范围为 `vx∈[-1.5,1.5] m/s`、`wz∈[-4,4] rad/s`。测试完整指令范围时指定 `--script_vel_max 1.5 --script_yaw_max 4`。

| 阶段 | 速度 / 高度 / 前摆臂 |
| --- | --- |
| `1_front_arm_cycle` | 机身静止、中心高度，前摆臂按余弦从抬起到放下再抬起 |
| `2_height_sweep_arm_up` | 机身静止、高度正弦扫描、前摆臂抬起 |
| `3a_walk_mid_arm_down` | 前后速度正弦、中位高度 0.355 m、前摆臂向下 |
| `3b_walk_mid_arm_up` | 同中位高度速度测试，前摆臂抬起 |
| `3c_walk_high_arm_down` | 前后速度正弦、最高高度 0.45 m、前摆臂向下 |
| `3d_walk_high_arm_up` | 同最高高度速度测试，前摆臂抬起 |
| `3e_walk_low_arm_down` | 前后速度正弦、最低高度 0.29 m、前摆臂向下 |
| `3f_walk_low_arm_up` | 同最低高度速度测试，前摆臂抬起 |
| `4a_spin_mid_arm_down` | 原地偏航正弦、中位高度 0.355 m、前摆臂向下 |
| `4b_spin_mid_arm_up` | 同中位高度偏航测试，前摆臂抬起 |
| `4c_spin_high_arm_down` | 原地偏航正弦、最高高度 0.45 m、前摆臂向下 |
| `4d_spin_high_arm_up` | 同最高高度偏航测试，前摆臂抬起 |
| `4e_spin_low_arm_down` | 原地偏航正弦、最低高度 0.29 m、前摆臂向下 |
| `4f_spin_low_arm_up` | 同最低高度偏航测试，前摆臂抬起 |
| `5_coupled_arm_up` | 前向速度与高度正弦、偏航双频正弦、前摆臂抬起；默认持续 16 s |

默认高度中心/幅值 `0.37/0.08 m`，即中位 0.37 m、最高 0.45 m、最低 0.29 m；修改中心/幅值后，三个固定高度随之变为 `center`、`center+amp`、`center-amp`。速度峰值 `0.8 m/s`，偏航峰值 `2 rad/s`。基础正弦周期等于 `--script_duration`，耦合阶段延长时保持该周期，默认运行两轮完整耦合周期；自定义耦合时长不必是周期整数倍。总时长为 `14×script_duration+script_coupled_duration`，按每阶段控制步数取整执行。动态脚本直接写入高度目标与当前高度指令，**绕过训练时的高度变化率限制**；前摆臂仍使用环境的目标限速。

每阶段打印跟踪绝对误差、偏差、部分跨环境 p95、俯仰/俯仰角速度、动作变化、前轮受力/净空和前摆臂误差。p95 统计是逐步计算后再在阶段内平均，不是整段所有时刻合并后的 p95。指标汇总输出在终端；需要保存终端报告时可将 stdout 重定向到文件。

| Play 参数 | 默认 / 用法 |
| --- | --- |
| `--script none/dynamic` | 默认普通播放 |
| `--script_duration` | 14 个基础阶段各 8 s，必须为正，同时决定基础正弦周期 |
| `--script_coupled_duration` | 最后耦合阶段时长，必须为正；不传则为 `2×script_duration`，默认 16 s |
| `--script_height_center` / `--script_height_amp` | 0.355 / 0.065 m，幅值非负 |
| `--script_vel_max` / `--script_yaw_max` | 0.8 m/s / 2 rad/s |
| `--camera global/tracking/both` | 动态模式使用，默认 global；全局相机位于 `(8.1,9.7,5.2) m`，看向 `(0,1.6,0) m`，视角进一步抬高、拉远，初始机器人位于画面偏左上；both 执行两遍 |
| `--viewer_robot` | 普通模式跟随机器人 0 的近景 |
| `--front_arm_upper_margin_deg` | 默认 0；在配置 8° 基础上增加评估裕量，绝对上限仍为 50°，不允许负数 |
| `--max_steps` | 普通模式退出步数，0 不限；动态模式按完整阶段执行，不使用该限制 |
| `--video_length` | 普通模式录像步数，默认 200；动态模式由脚本录制完整 pass |
| `--plot_env_index` | 时序采样的环境索引，默认 0 |
| `--plot_dir` | 自定义 CSV/PNG 根目录 |
| `--no_plot_export` | 关闭 CSV/PNG；不会关闭策略导出 |

### 13.2 Play 时序与 PNG 曲线

默认写入 checkpoint 同目录下 `play_plots/<timestamp>/`。动态测试在下面分别创建 `global/` 或 `tracking/`；普通播放直接写在时间戳目录。每次只记录 `--plot_env_index` 选中的一个环境，采样频率 50 Hz。

| 文件 | 内容与用途 |
| --- | --- |
| `tracking_data.csv` | 指令、真实状态、阶段、所有关节状态、动作及主动电机力矩的逐步记录 |
| `phases.csv` | 阶段起始时间与名称，便于对齐指令变化 |
| `body_command_tracking.png` | vx/vy/wz、高度、前摆臂的指令与真实值；高度另含 goal 曲线 |
| `body_roll_pitch.png` | 机身 roll/pitch，检查姿态偏差与振荡 |
| `front_wheel_arm_diagnostics.png` | 前轮接触力、净空与前摆臂有符号误差 `actual-command` |
| `all_joint_position.png` | 所有 articulation 关节位置，包括被动闭链与前球轮 |
| `all_joint_velocity.png` | 所有关节速度，检查被动关节/轮自由度异常运动 |
| `all_joint_acceleration.png` | 所有关节加速度，辅助识别高频抖动 |
| `policy_actions.png` | 6 维策略动作，物理映射前的归一化网络输出 |
| `applied_torque.png` | 7 个主动电机关节实际力矩，包含前摆臂 |
| `normalized_torque_delta.png` | 力矩一阶差分除以关节 effort limit，比较跨电机力矩跳变 |
| `normalized_torque_second_diff.png` | 力矩二阶差分除以 effort limit，检查力矩平滑度 |

力矩差分没有除以时间，因此不是 N·m/s 或 N·m/s²；reset 时差分历史清空。前轮力使用最近 3 个物理样本中的力范数峰值；净空为前轮中心高度减环境地面高度再减 `0.0303 m`，负值表示几何穿地趋势。缺失前轮传感器/刚体时诊断回退值可能为零，应先检查启动日志。

CSV 主要列族如下：

| 列 / 前缀 | 含义 |
| --- | --- |
| `step,time_s,phase` | 采样编号、相对时间、阶段名；第一条记录时间标记为 0 |
| `command_*` | 本次动作使用的速度、高度、前摆臂目标，另含高度 goal |
| `actual_*` | step 后的真实根速度、高度和前摆臂角度 |
| `roll_deg,pitch_deg` | 真实根姿态 |
| `joint_position_rad/*` / `joint_velocity_rad_s/*` / `joint_acceleration_rad_s2/*` | 按实际 joint name 导出的关节时序 |
| `policy_action/*` | 按策略动作关节顺序导出 |
| `applied_torque_nm/*` | 按 7 个主动关节导出 |
| `normalized_torque_delta/*` / `normalized_torque_second_diff/*` | effort limit 归一化力矩差分 |

数据配对方式是“step 前命令 + step 后状态”。普通播放遇到自动复位的那一步可能记录到复位后的状态，分析瞬态时应留意回合边界。动态模式不做普通故障复位，便于观察连续响应。

### 13.3 录像辅助脚本

```bash
# 默认从 mongo_flat_direct 的各 run 目录按修改时间选择最新 checkpoint。
bash scripts/record_video.sh

# 指定模型；默认 1 个环境、400 步、跟随机器人。
bash scripts/record_video.sh "$MONGO_CKPT"

# 修改录制环境数与长度；世界视角。
NUM_ENVS=4 STEPS=600 VIDEO_LEN=600 VIEWER_ROBOT=0 \
  bash scripts/record_video.sh "$MONGO_CKPT"

# 其他安装路径时，替换 Isaac Lab 启动器。
ISAACLAB_SH=/absolute/path/to/IsaacLab/isaaclab.sh \
  bash scripts/record_video.sh "$MONGO_CKPT"
```

脚本默认启动器是 `/home/inin/workspace/IsaacLab/isaaclab.sh`，录像位于 checkpoint 目录的 `videos/play/`。还支持 `TASK` 和 `EXTRA_ARGS` 环境变量。自动选择只查默认实验目录；使用自定义 `--experiment_name` 时建议显式指定 checkpoint。

### 13.4 导出与部署接口

每次 Play 会在 checkpoint 同目录的 `exported/` 写入下列文件，同名导出会被覆盖：

| 文件 | 内容 |
| --- | --- |
| `policy.pt` | Actor TorchScript，输入缩放后按时间拼接的 155 维观测，输出原始 6 维动作 |
| `policy.onnx` | 同策略的 ONNX 导出 |
| `action_processor.pt` | 有状态的可选低通/slew TorchScript，导出 batch_size=1，输入形状 `[1,6]` |
| `deployment_config.json` | 50 Hz 策略频率、动作顺序/缩放、滤波/限速开关、有效/保留延迟参数、上一动作语义，以及绝对角度观测语义、轮位置置零开关、历史帧数/顺序/复位填充规则 |

部署处理器启动或回合复位时调用 `reset()`。当前低通与 slew 都关闭，处理器对动作值等价于透传；它不包含机械限位、执行器 PD、前摆臂指令生成或仿真延迟。

部署侧仍需实现：与训练一致的每帧 31 维观测顺序、单位和 clip/scale；以 50 Hz 更新 5 帧历史，从旧到新拼接为 155 维，复位后用首帧填满；上一动作记忆；4 个后腿位置目标与 2 个轮速度目标映射；机械目标裁剪；独立前摆臂目标及限速。不能将 `policy.onnx` 的输出直接当作物理关节目标。当前导出 JSON 不是完整机器人控制器配置，PD、默认角度和机械限位还需核对源码/训练 YAML。

## 14. 验证、排查与实验维护

### 14.1 离线检查

```bash
conda activate lab
cd /home/inin/workspace/Mongo_RL
python -m unittest discover -s tests -v
python -m compileall -q source scripts tests
python scripts/rsl_rl/play.py --help
```

当前离线测试覆盖如下场景，包含裕量/绝对上限、reset 采样、平滑收回和真实 Play 注入/MuJoCo 公式兼容的 4 项测试；其余覆盖指令阶跃/持续斜坡/局部复位、高度和摆臂不参与稳态判定、角差跨 ±π、全阶段等权正负姿态项、前轮轻触阈值/平方成本/时间缩放与接触历史覆盖、零线速度转向时的机身平移成本、停车位移容差/延时锚定/平滑启用/退出重进/局部复位/世界平移不变性及求解器默认值。原有覆盖包括：原有 37 项奖励随机状态逐项比对、网络/PPO/原有奖励参数对齐、严格零指令轮速惩罚的触发条件、平衡余量与时间缩放、连杆净空的旋转/平移/地面高度计算、奖励汇总与数值安全屏蔽、USD 碰撞实例及缩放、前摆臂采样/限速、绝对角度观测及延迟 Actor/干净 Critic 的一致性、5 帧顺序与局部复位填充、高度指令采样范围、实际 CPU PPO 更新与 checkpoint 推理一致性、5 项 Isaac Lab 配置辅助函数兼容性，以及 Play 权重加载、绝对前摆臂目标刷新、诊断指标、多高度/摆臂组合和耦合阶段执行时长。未暴露 USD Python 库时，2 项 USD 几何测试跳过，其余可直接离线执行。
