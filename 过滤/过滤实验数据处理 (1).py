# This project is created by @aFei-CQUT
# ------------------------------------------------------------------------------------------------------------------------------------
# About @ aFei-CQUT
# - Interests&Hobbies: Programing,  ChatGPT,  Reading serious books,  Studying academical papers.
# - CurrentlyLearning: Mathmodeling，Python and Mathmatica (preparing for National College Mathematical Contest in Modeling).
# - Email:2039787966@qq.com
# - Pronouns: Chemical Engineering, Computer Science, Enterprising, Diligent, Hard-working, Sophomore,Chongqing Institute of Technology,
# - Looking forward to collaborating on experimental data processing of chemical engineering principle
# ------------------------------------------------------------------------------------------------------------------------------------

# 拿第一组数据建立基本流程

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 导入数据
imported_data = pd.read_excel(r"./过滤实验数据-非.xlsx", sheet_name=None)
sheet_name = list(imported_data.keys())[0]
data = imported_data[sheet_name]

# 选取数据
selected_data = data.iloc[1:12, 1:4]
data_array = selected_data.values

# 对第一列除以100化为标准单位
data_array[:, 0] = data_array[:, 0] / 100

# 计算 Δq
a = 0.0475
deltaV = 9.446 * 10**-4
deltaQ = deltaV / a

# 计算 Δθ 和 Δθ/Δq
delta_theta_list = np.diff(data_array[:, 1])
delta_q_list = np.full(len(delta_theta_list), deltaQ)
delta_theta_over_delta_q_list = delta_theta_list / delta_q_list

# 构造 q_list
q_list = np.linspace(0.05, 0.05 + len(delta_theta_list) * deltaQ, len(delta_theta_list) + 1)

# 取中点值拟合
q_to_fit_list = (q_list[:-1] + q_list[1:]) / 2
delta_theta_over_delta_q_to_fit_list = delta_theta_over_delta_q_list

# 数据配对
fit_data = np.column_stack((q_to_fit_list, delta_theta_over_delta_q_to_fit_list))

# 线性拟合
model = LinearRegression()
model.fit(fit_data[:, 0].reshape(-1, 1), fit_data[:, 1])
k_value = model.coef_[0]
q_e_intercept = model.intercept_

# 去掉最后一个数据点并重新进行拟合
fit_data_without_last_point = fit_data[:-1]
model_without_last_point = LinearRegression()
model_without_last_point.fit(fit_data_without_last_point[:, 0].reshape(-1, 1), fit_data_without_last_point[:, 1])

# 添加自身结尾点确保最右端辅助线边界存在
delta_theta_over_delta_q_list = np.append(delta_theta_over_delta_q_list, delta_theta_over_delta_q_list[-1])

# 输出结果
print("拟合直线斜率：", k_value)
print("拟合直线截距：", q_e_intercept)

# 设置字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 设置图片分辨度
plt.rcParams['figure.dpi'] = 300

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(fit_data[:, 0], fit_data[:, 1], color='red', label='拟合数据')
plt.plot(fit_data[:, 0], model.predict(fit_data[:, 0].reshape(-1, 1)), color='blue', label='拟合线')
plt.plot(fit_data[:, 0], model_without_last_point.predict(fit_data[:, 0].reshape(-1, 1)), color='green', label='去掉最后一个点的拟合线')

# 添加线条
for i in range(len(q_list) - 1):
    plt.axvline(x=q_list[i], color='black', linestyle='dashed')
    plt.hlines(y=delta_theta_over_delta_q_list[i], xmin=q_list[i], xmax=q_list[i + 1], color='black')
    plt.axvline(x=q_list[i + 1], color='black', linestyle='dashed')

plt.xlabel('q 值')
plt.ylabel('Δθ/Δq')
plt.legend(loc='upper left')
plt.savefig(r'./拟合图结果/0.png', dpi=300)
plt.show()
