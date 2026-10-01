import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# ==========================================
# 1. 物理模型与求解器
# ==========================================

def cart_pole_dynamics(state, u):
    mc = 1.0  # 小车质量
    mp = 0.1  # 摆杆质量
    l = 0.5   # 摆杆长度
    g = 9.81  # 重力
    
    x, v, theta, omega = state
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    denom = mc + mp * sin_t**2
    omega_dot = (-u*cos_t - mp*l*omega**2*cos_t*sin_t - (mc+mp)*g*sin_t) / (-l * denom)
    v_dot = (u + mp*sin_t*(l*omega**2 - g*cos_t)) / denom
    
    return np.array([v, v_dot, omega, omega_dot])

# --- 你修改的参数 ---
N = 40          
T = 5.0  # 时间改为 5秒
dt = T / (N - 1)

start_state = np.array([0, 0, 0, 0])      
final_state = np.array([10, 0, np.pi, 0]) # 终点改为 x=10

# --- 初始猜测 (Guess) ---
# 小提示：因为你要从0跑到10，猜测最好也是从0到10，这样求解器更容易收敛
x_init = np.linspace(0, 10, N) 
v_init = np.linspace(0, 0, N)   # 假设平均速度为 2m/s 可能更好，但设为0通常也能算出来
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
    
    # 动力学约束
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

print(f"正在计算从 x=0 到 x=10 的 {T}秒 轨迹...")
res = minimize(objective, decision_vars_init, method='SLSQP', 
               constraints={'type': 'eq', 'fun': constraints},
               options={'maxiter': 500, 'ftol': 1e-4}) # 稍微增加迭代次数

if not res.success:
    print("警告：优化可能未完全收敛 (但也可能只是精度未达标，结果仍可用)")
else:
    print("轨迹计算成功！正在生成动画...")

# 提取解
x_sol = res.x[0:N]
th_sol = res.x[2*N:3*N]
time_sol = np.linspace(0, T, N)

# ==========================================
# 2. 插值
# ==========================================
fps = 30
total_frames = int(T * fps)
time_smooth = np.linspace(0, T, total_frames)

f_x = interp1d(time_sol, x_sol, kind='cubic')
f_th = interp1d(time_sol, th_sol, kind='cubic')

x_smooth = f_x(time_smooth)
th_smooth = f_th(time_smooth)

# ==========================================
# 3. 制作动画 (修改版：摄像机跟随)
# ==========================================

fig, ax = plt.subplots(figsize=(10, 4)) # 宽一点的画布
ax.set_aspect('equal')
ax.grid(True)
ax.set_title(f"Swing-Up: 0m -> 10m in {T}s")

# 我们不在这里设置固定的 xlim，而是在 animate 里动态设置
ax.set_ylim(-1.0, 1.5) # y轴稍微留高一点

# 地面 (画一条很长的线)
ground_line, = ax.plot([], [], 'k-', lw=2)

# 小车
cart_width = 0.4  # 小车画大一点
cart_height = 0.2
cart = patches.Rectangle((0, 0), cart_width, cart_height, fc='blue', ec='black')
ax.add_patch(cart)

# 摆杆
pole_len = 0.6
pole_line, = ax.plot([], [], 'r-', lw=3)
mass_point, = ax.plot([], [], 'ro', markersize=8)

time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    cart.set_xy((-cart_width/2, -cart_height/2))
    pole_line.set_data([], [])
    mass_point.set_data([], [])
    ground_line.set_data([], [])
    time_text.set_text('')
    return cart, pole_line, mass_point, ground_line, time_text

def animate(i):
    curr_x = x_smooth[i]
    curr_th = th_smooth[i]
    
    # 1. 更新物体位置
    cart.set_xy((curr_x - cart_width/2, -cart_height/2))
    
    pole_x0 = curr_x
    pole_y0 = 0
    pole_x1 = curr_x + pole_len * np.sin(curr_th)
    pole_y1 = -pole_len * np.cos(curr_th)
    
    pole_line.set_data([pole_x0, pole_x1], [pole_y0, pole_y1])
    mass_point.set_data([pole_x1], [pole_y1])
    
    # 2. 关键修改：让摄像机跟随小车
    # 视窗范围：小车当前位置 前后各 2米
    view_width = 4.0
    ax.set_xlim(curr_x - view_width/2, curr_x + view_width/2)
    
    # 更新地面线，保证它始终铺满当前视窗
    ground_line.set_data([curr_x - view_width, curr_x + view_width], [-0.1, -0.1])
    
    time_text.set_text(f'Time = {time_smooth[i]:.1f}s | Pos = {curr_x:.1f}m')
    
    return cart, pole_line, mass_point, ground_line, time_text

# 注意：当需要动态改变坐标轴(xlim)时，通常建议 blit=False，
# 因为 blit 模式下改变背景(坐标轴)会比较麻烦。
ani = animation.FuncAnimation(fig, animate, frames=total_frames,
                              interval=1000/fps, blit=False, init_func=init)

plt.show()

ani.save('swingup_10m.gif', writer='pillow', fps=fps)