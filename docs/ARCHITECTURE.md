# 架构与数据流

## 设计原则

1. 任务层不直接访问 MuJoCo 数组，只使用命名关节状态、命令和位姿。
2. 规划器不包含 G1 关节名；机器人差异由 TOML 描述和适配器消化。
3. 几何路径、时间轨迹和控制执行是三个独立结果，便于定位问题。
4. 目标物在接近阶段仍是障碍物；只在指定 touch links 的闭爪阶段允许接触。
5. ROS 2 是边界适配器，不进入仿真、规划和控制算法的核心依赖。

## 完整 pipeline

```text
MuJoCo 真值位姿 / 外部 Pose
              │
              v
      场景语义与抓取目标
              │
       抓取候选生成/评分
              │
              v
 Pinocchio IK + HPP-FCL 状态/边碰撞
              │
              v
       OMPL 几何路径规划
              │
       shortcut / spline 后复检
              │
              v
       TOPP-RA 时间参数化
              │
              v
  position / velocity / torque 命令
              │
              v
 MuJoCo 动力学、接触、传感器与监控
              │
              v
  抓取确认 -> 搬运 -> 放置 -> 撤离
```

## 稳定接口

- `JointState`：名称、位置、速度、力矩和时间戳。
- `JointCommand`：关节名称、有序数值和控制模式。
- `Pose`：位置、wxyz 四元数、坐标系和时间戳。
- `RobotDescription`：关节组、末端执行器、touch bodies 和控制参数。

未来 ROS 2 bridge 只负责消息转换，建议 topic：

```text
/robot/joint_states
/robot/command/position
/robot/command/velocity
/robot/command/torque
/task/target_object_pose
/task/place_pose
/task/status
```

真实机器人适配器必须另外实现状态新鲜度检查、命令限幅、通信超时和急停，不能把
MuJoCo 的 `qfrc_applied` 语义直接映射到硬件。

## 里程碑

1. 仿真基础：本地模型、场景、配置、三模式控制和自动测试。
2. 运动学：Pinocchio FK/Jacobian、多初值有界 6D IK 已模块化，并与 MuJoCo 交叉验证。
3. 碰撞：HPP-FCL 活动组对全身/环境碰撞、边采样、允许碰撞矩阵和分阶段目标接触。
4. 规划：OMPL RRTConnect、路径简化、连续边采样和失败诊断已接入。
5. 轨迹：TOPP-RA 逐关节速度/加速度约束、2 ms 同步轨迹和碰撞复检已接入。
6. Pick-and-place：pregrasp、直线接近、闭爪接触、试抬、搬运和放置。
7. 柔顺控制：任务空间阻抗、导纳外环、接触状态与力/矩传感器滤波。
8. 双臂与 ROS 2：联合规划组、同步轨迹、跨臂碰撞和 topic bridge。
