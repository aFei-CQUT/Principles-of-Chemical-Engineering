# This project is created by @aFei-CQUT
# ------------------------------------------------------------------------------------------------------------------------------------
# About @ aFei-CQUT
# - Interests&Hobbies: Programing,  ChatGPT,  Reading serious books,  Studying academical papers.
# - CurrentlyLearning: Mathmodeling，Python and Mathmatica (preparing for National College Mathematical Contest in Modeling).
# - Email:2039787966@qq.com
# - Pronouns: Chemical Engineering, Computer Science, Enterprising, Diligent, Hard-working, Sophomore,Chongqing Institute of Technology,
# - Looking forward to collaborating on experimental data processing of chemical engineering principle
# ------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# 定义数据
flow_rate = np.array([0, 0.81, 1.62, 2.38, 3.23, 4.2, 5, 5.63, 6.462, 7.2, 7.99, 8.8, 9.6])  # 流量 m^3/h
head = np.array([16.602, 16.602, 15.35, 14.33, 14.25, 13.15, 12.52, 11.9, 10.05, 6.78, 5.67, 4.71, 4.68])  # 扬程 H
power = np.array([0.246, 0.264, 0.288, 0.318, 0.342, 0.378, 0.402, 0.42, 0.438, 0.45, 0.462, 0.48, 0.462])  # 功率 kW
efficiency = np.array([0, 13.82, 23.4, 29.1, 36, 39.58, 42.23, 43.18, 40.2, 29.42, 27.17, 23.33, 26.37])  # 效率 %

# 定义二次函数
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

# 对扬程进行二次拟合
params_head, _ = curve_fit(quadratic, flow_rate, head)
head_fit = quadratic(flow_rate, *params_head)

# 对功率进行二次拟合
params_power, _ = curve_fit(quadratic, flow_rate, power)
power_fit = quadratic(flow_rate, *params_power)

# 对效率进行二次拟合
params_efficiency, _ = curve_fit(quadratic, flow_rate, efficiency)
efficiency_fit = quadratic(flow_rate, *params_efficiency)

# 创建图形
fig, ax1 = plt.subplots(dpi=200)

# 绘制扬程-流量散点图和拟合曲线
ax1.scatter(flow_rate, head, color='blue', label='扬程 $H$ 数据')
ax1.plot(flow_rate, head_fit, 'b-', label='扬程 $H$ 拟合')
ax1.set_xlabel('$Q/(m^3/h)$')
ax1.set_ylabel('$H/m$', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 创建第二个y轴用于绘制功率和效率
ax2 = ax1.twinx()
ax2.scatter(flow_rate, power, color='red', label='功率 $N$ 数据')
ax2.plot(flow_rate, power_fit, 'r--', label='功率 $N$ 拟合')
ax2.set_ylabel('$N/kW$', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 创建第三个y轴用于绘制效率
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # 偏移第三个y轴
ax3.scatter(flow_rate, efficiency, color='green', label='效率 $\eta$ 数据')
ax3.plot(flow_rate, efficiency_fit, 'g-.', label='效率 $\eta$ 拟合')
ax3.set_ylabel('$\eta/\%$', color='green')
ax3.tick_params(axis='y', labelcolor='green')

# 添加图例
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3)

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 显示图形
plt.title('离心泵特性曲线及二次拟合')
plt.tight_layout(rect=[0.05, 0.03,0.95, 0.93])  # 调整布局
plt.savefig(r'./拟合图结果/离心泵特性曲线及二次拟合', dpi=300)
plt.show()
