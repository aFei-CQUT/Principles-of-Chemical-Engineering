import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 给定数据
x = np.array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 1000, 1000])
t = np.array([52, 48, 47, 50, 46, 46, 49, 46, 50, 46, 129, 175])

# 计算系数 -Gc/S
def calculate_coefficient():
    # 计算表面积并使用给定的常数计算系数
    return 110.8 * 0.001 / (7.4 * 5.4 + 5.4 * 1.4 + 7.4 * 1.4) / 2 / 0.01

# 计算干燥速率 U 的函数
def calculate_drying_rate(x, t):
    return calculate_coefficient() * x * 0.000001 / t

# 打印调试信息
print("系数 -Gc/S:", calculate_coefficient())
print("表面积:", (7.4 * 5.4 + 5.4 * 1.4 + 7.4 * 1.4) * 2 * 0.01)

# 初始水质量 (单位: kg)
G = 118.8 * 0.001

# 计算干燥速率
u = calculate_drying_rate(x, t)

# 初始化 X 数组并将 x 转换为适当的单位 (单位: kg)
X = []
x = x * 0.000001

# 计算累积水质量并更新 X
for i in x:
    G -= i
    X.append(G)

# 将 X 和 u 进行排序
sorted_indices = np.argsort(X)
X_sorted = np.array(X)[sorted_indices]
u_sorted = np.array(u)[sorted_indices]

# 插值使曲线平滑
X_smooth = np.linspace(min(X_sorted), max(X_sorted), 300)
u_smooth = make_interp_spline(X_sorted, u_sorted)(X_smooth)

# 初始化图表
fig, ax = plt.subplots(1, 1, figsize=(12, 9))

# 设置字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制干燥速率曲线
ax.plot(X_smooth, u_smooth, label='干燥速率曲线')
ax.scatter(X_sorted, u_sorted, marker='^', color='red')

# 设置坐标轴外观
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.grid()

# 设置图表标题和标签
ax.set_title("干燥速率曲线", fontsize=20)
ax.set_ylabel("$U$ ($kg$/($m²·h$))", fontsize=16)
ax.set_xlabel("$X$ ($kg$ 水/$kg$ 绝干料)", fontsize=16)
plt.legend(fontsize=14)

# 保存并显示图表
plt.savefig("./拟合图结果/恒定干燥条件下干燥速率曲线.png")
plt.show()
