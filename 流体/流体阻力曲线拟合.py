# This project is created by @aFei-CQUT
# ------------------------------------------------------------------------------------------------------------------------------------
# About @ aFei-CQUT
# - Interests&Hobbies: Programing,  ChatGPT,  Reading serious books,  Studying academical papers.
# - CurrentlyLearning: Mathmodeling，Python and Mathmatica (preparing for National College Mathematical Contest in Modeling).
# - Email:2039787966@qq.com
# - Pronouns: Chemical Engineering, Computer Science, Enterprising, Diligent, Hard-working, Sophomore,Chongqing Institute of Technology,
# - Looking forward to collaborating on experimental data processing of chemical engineering principle
# ------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# 雷诺数（Re）和阻力系数（λ）的数据
Re = np.array([51410, 46270, 44003, 38005, 31671, 25728.56, 20601.49, 15414.4, 10300, 5130, 4660, 4100, 3640, 3077, 2611, 2051, 1491, 1025.64, 559.43])
lambda_ = np.array([0.0182, 0.0188, 0.01928, 0.0202, 0.0196, 0.0207, 0.0209, 0.02162, 0.01544, 0.0369, 0.0395, 0.0424, 0.0394, 0.0432, 0.0447, 0.0415, 0.0501, 0.0687, 0.1026])

# 使用 polyfit 进行多项式拟合
degree = 9  # 设置多项式的阶数
coefficients = np.polyfit(np.log(Re), np.log(lambda_), degree)
p = np.poly1d(coefficients)

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制散点图和拟合曲线（双对数坐标轴）
plt.scatter(np.log(Re), np.log(lambda_), label='数据点')
plt.plot(np.log(Re), p(np.log(Re)), color='red', label=r'拟合曲线: $\log(\lambda) = {:.5f} + {:.5f}\log(Re) + {:.5f}\log^2(Re)$'.format(coefficients[0], coefficients[1], coefficients[2]))
plt.xlabel('lg(Re)')
plt.ylabel('lg(λ)')
plt.title('雷诺数与阻力系数双对数拟合')
plt.grid(True)

# 调整图例位置（向上偏移）
plt.legend(loc=None, bbox_to_anchor=(1, 1.35))
plt.savefig(r'./拟合图结果/流体阻力双对数拟合.png', dpi=300)
plt.show()

print("拟合参数:")
print("a =", np.exp(coefficients[0]))
print("b =", coefficients[1])
print("c =", coefficients[2])
