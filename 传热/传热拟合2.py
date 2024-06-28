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
re = np.array([14500, 20900, 23200, 27280.4, 29498.78, 32700])
nu_pr = np.array([33.31, 52.3, 61.0683, 65.96, 63, 60.87])

# 对数转换
log_re = np.log10(re)
log_nu_pr = np.log10(nu_pr)

# 执行线性拟合
fit = np.polyfit(log_re, log_nu_pr, 1)
fit_fn = np.poly1d(fit)

# 设置字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 绘制双对数图
plt.figure(figsize=(8, 6))
plt.scatter(log_re, log_nu_pr, color='r', label='拟合数据点')
plt.plot(log_re, fit_fn(log_re), color='k', label='拟合曲线')
plt.xlabel(r'$\log_{10}(Re)$')
plt.ylabel(r'$\log_{10}(\frac{Nu}{Pr^{0.4}})$')
plt.title('有强化丝双对数拟合')
plt.text(4.3,1.8, f'有强化丝拟合:y = {fit[0]:.2f}x + {fit[1]:.2f}', fontsize=12, color='k')

plt.legend()
plt.grid(True,which='both')
plt.minorticks_on()

plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)

plt.savefig("./拟合图结果/2.png",dpi=300)
plt.show()

