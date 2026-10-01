import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy.optimize import minimize
from scipy.interpolate import interp1d



def cart_pole_dynamics(state, u):
    mc = 1.0  # 小车质量
    mp = 0.1  # 摆杆质量
    l = 0.5   # 摆杆长度
    g = 9.81  # 重力
    
    x, v, theta, omega = state
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    denom = mc + mp * sin_t**2
    omega_dot = (-u*cos_t - mp*l*omega**2*cos_t*sin_t - (mc+mp)*g*sin_t) / (l * denom)
    v_dot = (u + mp*sin_t*(l*omega**2 + g*cos_t)) / denom
    
    return np.array([v, v_dot, omega, omega_dot])

# 设置参数
N = 40          
T = 2.0         
dt = T / (N - 1)
start_state = np.array([0, 0, 0, 0])      
final_state = np.array([0, 0, np.pi, 0])  

# 初始猜测
x_init = np.linspace(0, 0, N)
v_init = np.linspace(0, 0, N)
th_init = np.linspace(0, np.pi, N) 
om_init = np.linspace(0, 0, N)
u_init = np.zeros(N)
decision_vars_init = np.concatenate([x_init, v_init, th_init, om_init, u_init])

def constraints(vars):
    x = vars[0:N]
    v = vars[N:2*N]
    th = vars[2*N:3*N]
    om = vars[3*N:4*N]
    u = vars[4*N:5*N]
    
    eq_cons = []
    # 边界约束
    eq_cons.extend([x[0]-start_state[0], v[0]-start_state[1], th[0]-start_state[2], om[0]-start_state[3]])
    eq_cons.extend([x[-1]-final_state[0], v[-1]-final_state[1], th[-1]-final_state[2], om[-1]-final_state[3]])
    
    # 动力学约束 (Trapezoidal Collocation)
    for i in range(N-1):
        state_i = [x[i], v[i], th[i], om[i]]
        state_next = [x[i+1], v[i+1], th[i+1], om[i+1]]
        f_i = cart_pole_dynamics(state_i, u[i])
        f_next = cart_pole_dynamics(state_next, u[i+1])
        defect = np.array(state_next) - np.array(state_i) - 0.5 * dt * (f_i + f_next)
        eq_cons.extend(defect)
    return np.array(eq_cons)

def objective(vars):
    u = vars[4*N:5*N]
    return np.sum(u**2) 

print("正在计算轨迹")
res = minimize(objective, decision_vars_init, method='SLSQP', 
               constraints={'type': 'eq', 'fun': constraints},
               options={'maxiter': 300, 'ftol': 1e-4}) #迭代300次，误差1e-4

#SLSQP：求解非线性约束优化问题的算法。SLSQP（Sequential Least Squares Programming）是一种基于序列二次规划的方法，适用于求解具有非线性约束的优化问题。它通过在每次迭代中构建一个二次近似模型来寻找最优解，并使用拉格朗日乘子法处理约束条件
# 在当前迭代点 x_k处，将非线性约束 ceq(x)=0和 cineq(x)>=0进行一阶泰勒展开，将其近似为线性约束。

# 提取解
x_sol = res.x[0:N]
th_sol = res.x[2*N:3*N]
time_sol = np.linspace(0, T, N)


fps = 30
total_frames = int(T * fps)
time_smooth = np.linspace(0, T, total_frames)


f_x = interp1d(time_sol, x_sol, kind='cubic')
f_th = interp1d(time_sol, th_sol, kind='cubic')

x_smooth = f_x(time_smooth)
th_smooth = f_th(time_smooth)


# 制作动画


fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.8, 0.8)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Optimal Cart-Pole Swing-Up")

# 定义绘图元素
# 地面
ax.plot([-2, 2], [-0.1, -0.1], 'k-', lw=2)

# 小车 (矩形)
cart_width = 0.2
cart_height = 0.1
cart = patches.Rectangle((0, 0), cart_width, cart_height, fc='blue', ec='black')
ax.add_patch(cart)

# 摆杆 (线)
pole_len = 0.5
pole_line, = ax.plot([], [], 'r-', lw=3)

# 摆锤 (圆点)
mass_point, = ax.plot([], [], 'ro', markersize=8)

# 时间显示
time_template = 'Time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    cart.set_xy((-cart_width/2, -cart_height/2))
    pole_line.set_data([], [])
    mass_point.set_data([], [])
    time_text.set_text('')
    return cart, pole_line, mass_point, time_text

def animate(i):
    # 当前时刻的位置
    curr_x = x_smooth[i]
    curr_th = th_smooth[i]
    
    # 1. 更新小车位置 (中心在 curr_x，矩形需要左下角坐标)
    cart.set_xy((curr_x - cart_width/2, -cart_height/2))
    
    # 2. 计算摆杆端点位置
    # 注意：这里 theta=0 是向下，theta=pi 是向上
    # 摆杆起点 (小车中心)
    pole_x0 = curr_x
    pole_y0 = 0
    # 摆杆终点
    # 既然定义向下为0度，那么 y = -L * cos(theta)
    # theta=0 -> y=-L (下), theta=pi -> y=L (上)
    pole_x1 = curr_x + pole_len * np.sin(curr_th)
    pole_y1 = -pole_len * np.cos(curr_th)
    
    # 更新摆杆
    pole_line.set_data([pole_x0, pole_x1], [pole_y0, pole_y1])
    
    # 更新摆锤
    mass_point.set_data([pole_x1], [pole_y1])
    
    # 更新时间
    time_text.set_text(time_template % time_smooth[i])
    
    return cart, pole_line, mass_point, time_text

ani = animation.FuncAnimation(fig, animate, frames=total_frames,
                              interval=1000/fps, blit=True, init_func=init)

# 如果你在 Jupyter Notebook 里，可以取消下面这行的注释来显示交互式动画
# from IPython.display import HTML
# HTML(ani.to_jshtml())

plt.show()

# 如果想保存成 gif，可以取消下面这行的注释 (需要安装 imagemagick 或 ffmpeg)
# ani.save('swingup.gif', writer='pillow', fps=fps)