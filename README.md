# robot_mj_ws

面向双臂和人形机器人操作任务的模块化 MuJoCo 仿真工程。第一版适配 Unitree G1
三指手，核心模块不依赖 ROS 2；后续可通过适配层连接 ROS 2 topic 或真实机器人。

当前里程碑已经建立从场景到真实接触验证的单臂抓取链路：

- 配置驱动的机器人、关节组和末端执行器描述；
- 配置驱动的桌面、障碍物和多类自由物体场景；
- position、velocity、torque 三种统一关节控制模式；
- MuJoCo 真值物体位姿接口，同时保留外部位姿更新入口；
- Pinocchio 6D IK、HPP-FCL、OMPL 与 TOPP-RA；
- 分阶段碰撞策略、三指力限位闭合和不焊接物体的物理试抬；
- 原始 G1 模型与 mesh 放在仓库内，不依赖绝对路径。

后续里程碑依次加入 Pinocchio、HPP-FCL、OMPL、TOPP-RA、抓取状态机、双臂协调、
阻抗/导纳控制和 ROS 2 bridge。完整边界与数据流见
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 安装

```bash
cd /home/zzz/robot_mj_ws
python3.11 -m venv .venv
.venv/bin/pip install -e .
```

从运动学与规划里程碑开始，再安装可选依赖：

```bash
.venv/bin/pip install -e '.[planning,dev]'
```

也可以先复用旧工程环境验证第一阶段：

```bash
cd /home/zzz/robot_mj_ws
PYTHONPATH=src /home/zzz/unitree_test/.venv/bin/python -m robot_mj.apps.view_scene
```

加 `--headless --seconds 0.1` 可以只编译场景并执行短时间无界面验证。

单独验证自碰撞、环境碰撞和边碰撞：

```bash
PYTHONPATH=src .venv/bin/python -m robot_mj.apps.check_collision
```

运行 OMPL 与 TOPP-RA 完整规划链路：

```bash
PYTHONPATH=src .venv/bin/python -m robot_mj.apps.plan_trajectory
```

运行完整抓取并在 MuJoCo 中用接触/摩擦试抬验收：

```bash
PYTHONPATH=src .venv/bin/python -m robot_mj.apps.grasp_pick
```

CI 或终端验证使用 `--headless`；只检查 IK、碰撞、OMPL 和 TOPP-RA
使用 `--plan-only --headless`。

运行不依赖 GUI 的测试：

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## 当前目录

```text
assets/                    可随仓库分发的机器人资源和许可证
configs/robots/            机器人关节组、末端和控制参数
configs/tasks/             桌面、障碍物、抓取候选和放置区
src/robot_mj/sim/          MuJoCo 后端
src/robot_mj/control/      三种关节控制模式
src/robot_mj/kinematics/   Pinocchio FK 与 Jacobian 后端
src/robot_mj/collision/    HPP-FCL 自碰撞、环境碰撞和分阶段接触
src/robot_mj/planning/     OMPL RRTConnect 与路径复检
src/robot_mj/trajectory/   TOPP-RA 同步时间参数化
src/robot_mj/robots/       通用描述及机器人适配器
src/robot_mj/tasks/        场景和任务接口
src/robot_mj/interfaces/   状态、命令、位姿输入等稳定接口
docs/                      架构与开发路线
tests/                     不启动 GUI 的自动化验证
```
