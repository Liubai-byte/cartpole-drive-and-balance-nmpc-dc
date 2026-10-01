import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#物理学公式
def cart_pole_dynamics(state, u):
    # state = [x, x_dot, theta, theta_dot]
    # u = force
    mc = 1.0  
    mp = 0.1 
    l = 0.5   
    g = 9.81 
    
    x, v, theta, omega = state
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    denom = mc + mp * sin_t**2
    omega_dot = (-u*cos_t - mp*l*omega**2*cos_t*sin_t - (mc+mp)*g*sin_t) / (-l * denom)
    v_dot = (u + mp*sin_t*(l*omega**2 - g*cos_t)) / denom
    #!
    return np.array([v, v_dot, omega, omega_dot])


#把时间分成 N 段，每一段的状态和力都是我们要找的未知数
N = 40       
T = 2.0        
dt = T / (N - 1)


# 初始：theta = 0 (下垂), 目标：theta = pi (直立)
start_state = np.array([0, 0, 0, 0])     
final_state = np.array([0, 0, np.pi, 0])  

# 初始猜测
x_init = np.linspace(0, 0, N)
v_init = np.linspace(0, 0, N)
th_init = np.linspace(0, np.pi, N) # 假设它匀速转上去
om_init = np.linspace(0, 0, N)
u_init = np.zeros(N) 

# 把所有变量打包成一个长向量，传给优化器
# 变量顺序：[所有x, 所有v, 所有theta, 所有omega, 所有u]
decision_vars_init = np.concatenate([x_init, v_init, th_init, om_init, u_init])

# ==========================================
# 约束 (Defects)
def constraints(vars):

    x = vars[0:N]
    v = vars[N:2*N]
    th = vars[2*N:3*N]
    om = vars[3*N:4*N]
    u = vars[4*N:5*N]
    
    # 边界约束：起点和终点必须对齐
    eq_cons = []
    # 起点
    eq_cons.extend([x[0]-start_state[0], v[0]-start_state[1], 
                    th[0]-start_state[2], om[0]-start_state[3]])
    # 终点
    eq_cons.extend([x[-1]-final_state[0], v[-1]-final_state[1], 
                    th[-1]-final_state[2], om[-1]-final_state[3]])
    
    # 动力学约束 (Defects): 下一刻的状态 必须等于 当前状态 + 物理变化
    # Trapezoidal Collocation
    for i in range(N-1):
        state_i = [x[i], v[i], th[i], om[i]]
        state_next = [x[i+1], v[i+1], th[i+1], om[i+1]]
        
        f_i = cart_pole_dynamics(state_i, u[i])
        f_next = cart_pole_dynamics(state_next, u[i+1])
        
        # 核心公式：x[k+1] - x[k] = 0.5 * dt * (f[k] + f[k+1])
        # 如果这个式子不为0，说明物理上有“缺陷”
        defect = np.array(state_next) - np.array(state_i) - 0.5 * dt * (f_i + f_next)
        eq_cons.extend(defect)
        
    return np.array(eq_cons)

# ==========================================
# 目标函数 (Cost Function)
def objective(vars):
    # 希望力越小越好
    u = vars[4*N:5*N]
    return np.sum(u**2)

# ==========================================
# 调用求解器
res = minimize(objective, decision_vars_init, method='SLSQP', 
               constraints={'type': 'eq', 'fun': constraints},
               options={'maxiter': 300, 'ftol': 1e-4})

if res.success:
    print("成功")
else:
    print("失败")

# ==========================================
# 可视化结果（表格）

x_opt = res.x[0:N]
th_opt = res.x[2*N:3*N]
u_opt = res.x[4*N:5*N]
time = np.linspace(0, T, N)

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(time, x_opt, 'b')
plt.ylabel('Cart Position (m)')
plt.title('Optimal Swing-up Trajectory')
plt.grid(True)



plt.subplot(3, 1, 2)
plt.plot(time, np.degrees(th_opt), 'g')
plt.ylabel('Pole Angle (deg)')
plt.yticks([0, 90, 180])
plt.grid(True)
# 180度意味着摆杆完全竖直向上

plt.subplot(3, 1, 3)
plt.plot(time, u_opt, 'r')
plt.ylabel('Control Force (N)')
plt.xlabel('Time (s)')
plt.grid(True)

plt.tight_layout()
plt.show()