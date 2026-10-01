import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 系统参数
mc = 1.0  # 小车质量 (kg)
mp = 0.1  # 摆杆质量 (kg)
l = 0.2   # 摆杆质心到支点的距离 (m)
g = 9.81  # 重力加速度 (m/s^2)

# NMPC参数
dt = 0.1          # 时间步长 (s)
N = 50            # 预测步长
T = 5.0           # 总时间 (s)
nx = 4            # 状态维度 [p, θ, v, ω]
nu = 1            # 控制输入维度 [F]

# 创建CasADi优化问题
opti = ca.Opti()

# 决策变量
X = opti.variable(nx, N+1)  # 状态轨迹
U = opti.variable(nu, N)    # 控制轨迹

# 参数（当前状态和参考轨迹）
x0 = opti.parameter(nx)     # 当前状态
p_ref = opti.parameter(1)   # 参考位置
theta_ref = opti.parameter(1)  # 参考角度

# 系统动力学（连续时间）
def cartpole_dynamics(x, u):
    p, theta, v, omega = x[0], x[1], x[2], x[3]
    sin_theta = ca.sin(theta)
    cos_theta = ca.cos(theta)
    
    denom = mc + mp * sin_theta**2
    dv_dt = (u + mp*sin_theta*(l*omega**2 + g*cos_theta)) / denom
    domega_dt = -(g * sin_theta * (mc + mp) + (u * cos_theta + mp * l * omega**2 * sin_theta * cos_theta)) / (l * denom)
    return ca.vertcat(v, omega, dv_dt, domega_dt)

# 离散时间动力学（欧拉法）
def discrete_dynamics(x, u):
    dx = cartpole_dynamics(x, u)
    return x + dt * dx

# 初始状态约束
opti.subject_to(X[:, 0] == x0)

# 动力学约束
for k in range(N):
    x_next = X[:, k] + dt * cartpole_dynamics(X[:, k], U[:, k])
    opti.subject_to(X[:, k+1] == x_next)

# 控制输入约束
# F_min, F_max = -100, 100
# opti.subject_to(opti.bounded(F_min, U, F_max))

# 状态约束（可选）
# opti.subject_to(opti.bounded(-2, X[0,:], 12))  # 位置约束
# opti.subject_to(opti.bounded(-ca.inf, X[1,:], ca.inf))  # 角度无约束

# 代价函数
Q = ca.diag([100, 500, 100, 100])  # 状态权重 [p, θ, v, ω]
R = 0.1                          # 控制权重

cost = 0
for k in range(N):
    # 参考轨迹：位置从0到10m，角度从0到π
    p_des = p_ref * (k*dt/5.0)  # 线性增加的位置参考
    theta_des = theta_ref * (k*dt/5.0)  # 线性增加的角度参考
    
    state_error = X[:, k] - ca.vertcat(p_des, theta_des, 0, 0)
    # cost += ca.mtimes([state_error.T, Q, state_error]) + R * U[:, k]**2
    cost +=  R * U[:, k]**2
    

# 终端代价
terminal_error = X[:, N] - ca.vertcat(p_ref, theta_ref, 0, 0)
cost += ca.mtimes([terminal_error.T, Q, terminal_error])

opti.minimize(cost)

# 设置求解器
opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
opti.solver('ipopt', opts)

# 仿真参数
tsim = np.arange(0, T, dt)
nsim = len(tsim)
x_sim = np.zeros((nx, nsim))
u_sim = np.zeros((nu, nsim))

# 初始状态 [位置, 角度, 速度, 角速度]
x_sim[:, 0] = [0, 0.1, 0, 0]  # 小角度偏移避免奇点

# NMPC仿真循环
for i in range(1, nsim):
    # 设置当前状态
    opti.set_value(x0, x_sim[:, i-1])
    opti.set_value(p_ref, 10.0)    # 最终位置参考
    opti.set_value(theta_ref, np.pi)  # 最终角度参考
    
    # 求解优化问题
    try:
        sol = opti.solve()
        u_opt = sol.value(U[:, 0])
    except:
        print(f"求解失败 @ t={i*dt:.1f}s")
        u_opt = np.array([0])
    
    u_sim[:, i] = u_opt
    
    # 模拟系统响应（使用RK45积分）
    def dynamics(t, y):
        return cartpole_dynamics(y, u_opt).full().flatten()
    
    sol = solve_ivp(dynamics, [0, dt], x_sim[:, i-1], method='RK45')
    x_sim[:, i] = sol.y[:, -1]

# 结果可视化
plt.figure(figsize=(12, 8))

# 小车位置
plt.subplot(3, 1, 1)
plt.plot(tsim, x_sim[0, :], 'b-', label='Position')
plt.plot(tsim, 2*tsim, 'r--', label='Reference')  # 线性参考轨迹
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)

# 摆杆角度
plt.subplot(3, 1, 2)
plt.plot(tsim, x_sim[1, :], 'b-', label='Angle')
plt.plot(tsim, np.pi*tsim/5, 'r--', label='Reference')  # 线性角度参考
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

# 控制输入
plt.subplot(3, 1, 3)
plt.plot(tsim, u_sim[0, :], 'g-')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.grid(True)

plt.tight_layout()
plt.show()