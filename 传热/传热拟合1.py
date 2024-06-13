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
re = np.array([20700, 38100, 33078.9, 36800, 40525, 45200])
nu_pr = np.array([44.46, 61.9, 70.165, 74.57, 81.65, 102.99])

# 对数转换
log_nu_pr = np.log10(nu_pr)
log_re = np.log10(re)

# 执行线性拟合
fit = np.polyfit(log_re, log_nu_pr, 1)
fit_fn = np.poly1d(fit)

# 设置字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 绘制双对数图
plt.figure(figsize=(8, 6))
plt.scatter(log_re, log_nu_pr, color='blue', label='Data')
plt.plot(log_re, fit_fn(log_re), color='red', label='Fit')
plt.xlabel(r'$\log_{10}(Re)$')
plt.ylabel(r'$\log_{10}(\frac{Nu}{Pr^{0.4}})$')
plt.title('无强化丝双对数拟合')
plt.text(4.33,1.8, f'无强化丝拟合:y = {fit[0]:.2f}x + {fit[1]:.2f}', fontsize=12, color='red')
plt.legend()
plt.grid(True)
plt.savefig("./拟合图结果/1.png",dpi=300)
plt.show()

