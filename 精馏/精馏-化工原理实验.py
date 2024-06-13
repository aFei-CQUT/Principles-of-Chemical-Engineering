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
import sympy
import matplotlib.pyplot as plt
import pandas as pd

# 用 x 表示 y 的平衡线方程，命名 LaTeX 格式为：$y_{e}$
def y_e(x):
    y = α * x / (1 + (α - 1) * x)
    return y

# 用 y 表示 x 的平衡线方程，命名 LaTeX 格式为：$x_{e}$
def x_e(y):
    x = y / (α - (α - 1) * y)
    return x

# 精馏段操作方程，命名 LaTeX 格式为：$y_{n+1}$，取其中加号为 plus 首字母 p
def y_np1(x):
    y = R / (R + 1) * x + xD / (R + 1)
    return y

# 提馏段操作方程，命名 LaTeX 格式为：$y_{m+1}$，取其中加号为 plus 首字母 p
def y_mp1(x):
    y = (L + q * F) / (L + q * F - W) * x - W / (L + q * F - W) * xW
    return y

# $q$ 线方程，命名 LaTeX 格式为：$y_{q}$
def y_q(x):
    if q == 1:
        y = 0
    else:
        y = q / (q - 1) * x - 1 / (q - 1) * xF
    return y

# 待求量以 None 表示
F, D, W, xF, xD, xW, R = [100, None, None, 0.5, 0.97, 0.04, 2]
q, α = [1, 2.5]

# 提馏段操作方程须知 D、W
# 使用 Sympy 列出物料衡算的矩阵形式并求解
A = sympy.Matrix([[1, 1], [xD, xW]])
b = sympy.Matrix([F, xF * F])
D, W = A.solve(b)
L = R * D

# 调用函数计算相应数据数组
# x 数据数组
x_array = np.linspace(0, 1, 50)
# $y_{q}$ 数据数组
y_q_array = y_q(x_array)
# $y_{e}$ 数据数组
y_e_array = y_e(x_array)
# $y_{n+1}$ 数据数组
y_np1_array = y_np1(x_array)
# $y_{m+1}$ 数据数组
y_mp1_array = y_mp1(x_array)

# 确定 Q 点
xQ = ((R + 1) * xF + (q - 1) * xD) / (R + q)
yQ = (xF * R + q * xD) / (R + q)

# 逐板计算，求解每个塔板的平衡情况
# n 从 0 开始计，n=0 时有 $y_{0} = x_{D}$
yn = np.array([xD])
xn = np.array([])
NT = None
while x_e(yn[-1]) > xW:
    xn = np.append(xn, x_e(yn[-1]))
    if xn[-1] > xQ:
        yn = np.append(yn, y_np1(xn[-1]))
    else:
        yn = np.append(yn, y_mp1(xn[-1]))
else:
    xn = np.append(xn, x_e(yn[-1]))
    NT = len(xn)

# 图解法计算理论塔板数的图示数据
xNT = np.array([xD])
yNT = np.array([xD])
for n, i in enumerate(xn):
    xNT = np.append(xNT, i)
    yNT = np.append(yNT, yn[n])
    xNT = np.append(xNT, i)
    if i >= xQ:
        yNT = np.append(yNT, y_np1(i))
    else:
        yNT = np.append(yNT, y_mp1(i))

# 作图设置
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# 设置 matplotlib.pyplot 字体显示正常
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 作图
# 对角线
y_array = x_array
ax.plot(x_array, y_array, ls="-", label="对角线")

xQ_plot = [0.5, 0.5]
yQ_plot = [0.5, 0.7]

ax.plot(xQ_plot, yQ_plot, label="$q$ 线")
ax.plot(x_array, y_e_array, label="平衡线")
ax.plot(x_array, y_np1_array, label="精馏操作线")
ax.plot(x_array, y_mp1_array, label="提馏操作线")
ax.plot(xn, yn, label="塔板操作平衡点", ls=":", marker="+", markersize=10)
ax.plot(xNT, yNT, label="图解法—理论塔板", ls=":")

ax.plot(xD, xD, marker=".", markersize=10)  # 画点
ax.plot(xW, xW, marker=".", markersize=10)
ax.plot(xQ_plot, yQ_plot, marker=".", markersize=10)

# 点注释
ax.annotate("$W$ 点", xy=(xW, xW), xytext=(xW + 0.05, xW), arrowprops=dict(arrowstyle="->"))
ax.annotate("$D$ 点", xy=(xD, xD), xytext=(xD, xD - 0.05), arrowprops=dict(arrowstyle="->"))
ax.annotate("$Q$ 点", xy=(xQ, yQ), xytext=(xQ, yQ - 0.05), arrowprops=dict(arrowstyle="->"))
ax.legend()

# 获取坐标轴
ax = plt.gca()
# 设置坐标轴顶部线条粗细
ax.spines["top"].set_linewidth(2)
# 设置坐标轴底部线条粗细
ax.spines["bottom"].set_linewidth(2)
# 设置坐标轴左侧线条粗细
ax.spines["left"].set_linewidth(2)
# 设置坐标轴右侧线条粗细
ax.spines["right"].set_linewidth(2)
ax.grid()
# 图中显示所需理论板数
ax.text(x=0.6, y=0.4, s="所需理论板数：%d" % (len(xn) - 1))
# 图标题
ax.set_title("图解法求理论塔板数")
ax.set_ylabel("$y$")
ax.set_xlabel("$x$")
plt.savefig(r'./拟合图结果/图解法求理论塔板数.png', dpi=300)
plt.show()

# 输出结果
print("x 数据数组:\n", pd.DataFrame({"$x$": x_array}, columns=["$x$"]).to_latex(index=False))
print("\ny 数据数组:\n", pd.DataFrame({"$y$": y_array}, columns=["$y$"]).to_latex(index=False))
print("塔板上的平衡数据:")
print(pd.DataFrame({"$y_n$": ["{:.4f}".format(val) for val in yn]}).to_latex(index=False))
print("\n塔板上的平衡数据:")
print(pd.DataFrame({"$x_n$": ["{:.4f}".format(val) for val in xn]}).to_latex(index=False))
print("\n$Q$ 点:\n", pd.DataFrame({"$x_Q$": [xQ], "$y_Q$": [yQ]}).to_latex(index=False))
print("\n$NT$ =", NT - 1)
