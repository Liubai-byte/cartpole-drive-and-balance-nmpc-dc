# Cartpole Swing-up and Position Control

Control an inverted-pendulum cart with a single input: the horizontal force applied to the cart. Starting from a stationary cart and a pendulum hanging straight down, the controller must swing the pendulum up to the upright position while driving the cart to a specified target location, then hold both the cart position and the pendulum stable.

Two optimal control methods are implemented:

- **Direct Collocation (DC)**: offline trajectory planning. The cart-pendulum dynamics are transcribed into a nonlinear programming problem (NLP), which is solved to obtain a dynamically feasible trajectory from the initial hanging state to the final upright-and-positioned state.

- **Nonlinear Model Predictive Control (NMPC)**: real-time feedback control. A finite-horizon optimal control problem is solved in a receding-horizon fashion to track the planned trajectory and reject disturbances.

## Initial / Final Conditions

| State | Initial | Final |
|---|---|---|
| Cart position `x` | 0 | `x_target` |
| Cart velocity `ẋ` | 0 | 0 |
| Pendulum angle `θ` | π (hanging down) | 0 (upright) |
| Pendulum angular velocity `θ̇` | 0 | 0 |

# 倒立摆小车：摆起与定位控制

本项目实现倒立摆小车的摆起与定位任务。控制输入只有一个：作用在小车上的水平力。系统从静止、摆杆自然竖直下垂的状态出发，需要把摆杆摆动至倒立位置，同时驱动小车到达指定目标点并停车，最终保持摆杆稳定直立。

仓库包含两种最优控制方法：

- **直接配点法（DC）**：离线轨迹规划。将小车-摆杆动力学转换成有限维非线性规划问题（NLP），求解出一条从“下垂静止”到“直立且到位”的动力学可行轨迹。

- **非线性模型预测控制（NMPC）**：在线实时反馈控制。通过滚动时域求解有限时域最优控制问题，实时跟踪规划轨迹，并抑制扰动和模型误差。

## 初始 / 目标状态

| 状态 | 初始 | 目标 |
|---|---|---|
| 小车位置 `x` | 0 | `x_target` |
| 小车速度 `ẋ` | 0 | 0 |
| 摆杆角度 `θ` | π（下垂） | 0（直立） |
| 摆杆角速度 `θ̇` | 0 | 0 |
