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

# 原始数据
re_no_enhancement = np.array([20700, 38100, 33078.9, 36800, 40525, 45200])
nu_pr_no_enhancement = np.array([44.46, 61.9, 70.165, 74.57, 81.65, 102.99])

re_enhancement = np.array([14500, 20900, 23200, 27280.4, 29498.78, 32700])
nu_pr_enhancement = np.array([33.31, 52.3, 61.0683, 65.96, 63, 60.87])

# 对数转换
log_re_no_enhancement = np.log10(re_no_enhancement)
log_nu_pr_no_enhancement = np.log10(nu_pr_no_enhancement)

log_re_enhancement = np.log10(re_enhancement)
log_nu_pr_enhancement = np.log10(nu_pr_enhancement)

# 执行线性拟合
fit_no_enhancement = np.polyfit(log_re_no_enhancement, log_nu_pr_no_enhancement, 1)
fit_fn_no_enhancement = np.poly1d(fit_no_enhancement)

fit_enhancement = np.polyfit(log_re_enhancement, log_nu_pr_enhancement, 1)
fit_fn_enhancement = np.poly1d(fit_enhancement)

# 设置字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 绘制双对数图
plt.figure(figsize=(8, 6))

# 无强化丝数据及拟合直线
plt.scatter(log_re_no_enhancement, log_nu_pr_no_enhancement, color='blue', label='无强化丝数据')
plt.plot(log_re_no_enhancement, fit_fn_no_enhancement(log_re_no_enhancement), color='red', label='无强化丝拟合线')

# 有强化丝数据及拟合直线
plt.scatter(log_re_enhancement, log_nu_pr_enhancement, color='green', label='有强化丝数据')
plt.plot(log_re_enhancement, fit_fn_enhancement(log_re_enhancement), color='orange', label='有强化丝拟合线')

plt.xlabel(r'$\log_{10}(Re)$')
plt.ylabel(r'$\log_{10}(\frac{Nu}{Pr^{0.4}})$')
plt.title('有强化丝vs.无强化丝')

# 添加拟合方程
plt.text(4.25, 1.80, f'无强化丝拟合方程: y = {fit_no_enhancement[0]:.2f}x + {fit_no_enhancement[1]:.2f}', fontsize=10, color='red')
plt.text(4.10, 1.70, f'有强化丝拟合方程: y = {fit_enhancement[0]:.2f}x + {fit_enhancement[1]:.2f}', fontsize=10, color='orange')

plt.legend()
plt.grid(True)
plt.savefig("./拟合图结果/3.png",dpi=300)
plt.show()

