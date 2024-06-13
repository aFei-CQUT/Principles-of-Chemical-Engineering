# This project is created by @aFei-CQUT
# ------------------------------------------------------------------------------------------------------------------------------------
# About @ aFei-CQUT
# - Interests&Hobbies: Programing,  ChatGPT,  Reading serious books,  Studying academical papers.
# - CurrentlyLearning: Mathmodeling，Python and Mathmatica (preparing for National College Mathematical Contest in Modeling).
# - Email:2039787966@qq.com
# - Pronouns: Chemical Engineering, Computer Science, Enterprising, Diligent, Hard-working, Sophomore,Chongqing Institute of Technology,
# - Looking forward to collaborating on experimental data processing of chemical engineering principle
# ------------------------------------------------------------------------------------------------------------------------------------

# 函数化
# 不去掉最后一个点的拟合线 vs.去掉最后一个点的拟合线

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def filter_fit_line(selected_data):
    a = 0.0475
    deltaV = 9.446 * 10**-4
    deltaQ = deltaV / a
    delta_theta_list = np.diff(selected_data[:, 1])
    delta_q_list = np.full(len(delta_theta_list), deltaQ)
    delta_theta_over_delta_q_list = delta_theta_list / delta_q_list
    q_list = np.linspace(0.05, 0.05 + len(delta_theta_list) * deltaQ, len(delta_theta_list) + 1)
    q_mean_to_fit_line_list = (q_list[:-1] + q_list[1:]) / 2
    delta_theta_over_delta_q_mean_to_fit_line_list = delta_theta_over_delta_q_list
    fit_coordinates = np.column_stack((q_mean_to_fit_line_list, delta_theta_over_delta_q_mean_to_fit_line_list))
    model = LinearRegression()
    model.fit(fit_coordinates[:, 0].reshape(-1, 1), fit_coordinates[:, 1])
    k_value = model.coef_[0]
    q_e_intercept = model.intercept_
    fit_coordinates_without_last_point = fit_coordinates[:-1]
    model_without_last_point = LinearRegression()
    model_without_last_point.fit(fit_coordinates_without_last_point[:, 0].reshape(-1, 1), fit_coordinates_without_last_point[:, 1])
    k_value_without_last_point = model_without_last_point.coef_[0]
    q_e_intercept_without_last_point = model_without_last_point.intercept_
    delta_theta_over_delta_q_list = np.append(delta_theta_over_delta_q_list, delta_theta_over_delta_q_list[-1])
    return k_value, q_e_intercept, k_value_without_last_point, q_e_intercept_without_last_point, fit_coordinates, model, fit_coordinates_without_last_point, model_without_last_point, delta_theta_over_delta_q_list

def filter_plot(selected_data, fit_coordinates, fit_line, fit_line_without_last_point, q_mean_to_fit_line_list, delta_theta_over_delta_q_list, save_path):
    a = 0.0475
    deltaV = 9.446 * 10**-4
    deltaQ = deltaV / a
    q_list = np.linspace(0.05, 0.05 + len(selected_data) * deltaQ, len(selected_data) + 1)
    plt.figure(figsize=(10, 6))
    plt.scatter(fit_coordinates[:, 0], fit_coordinates[:, 1], color='red', label='拟合数据')
    plt.plot(fit_coordinates[:, 0], fit_line.predict(fit_coordinates[:, 0].reshape(-1, 1)), color='blue', label='不去掉最后一个点的拟合线')
    plt.plot(fit_coordinates[:, 0], fit_line_without_last_point.predict(fit_coordinates[:, 0].reshape(-1, 1)), color='green', label='去掉最后一个点的拟合线')

    for i in range(len(q_list) - 2):
        plt.axvline(x=q_list[i], color='black', linestyle='dashed')
        plt.hlines(y=delta_theta_over_delta_q_list[i], xmin=q_list[i], xmax=q_list[i + 1], color='black')
        plt.axvline(x=q_list[i + 1], color='black', linestyle='dashed')
        plt.hlines(y=fit_line_without_last_point.predict(np.array([[q_list[i]]]))[0], xmin=q_list[i], xmax=q_list[i + 1], color='gray', linestyle='dotted')
        plt.plot(q_list[i], fit_line_without_last_point.predict(np.array([[q_list[i]]]))[0], marker='o', markersize=5, color='gray')

    plt.xlabel('q 值')
    plt.ylabel('Δθ/Δq')
    plt.legend(loc='upper left')
    plt.title('不去掉最后一个点的拟合线 vs.去掉最后一个点的拟合线')  # 添加标题
    # 保存图像
    plt.savefig(save_path,dpi=300)
    # 显示图像
    plt.show()
    # 关闭图像以释放内存
    plt.close()

# 设置字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 设置图片分辨度
plt.rcParams['figure.dpi'] = 300
# 导入数据
imported_data = pd.read_excel(r"./过滤实验数据-非.xlsx", sheet_name=None)
sheet_name = list(imported_data.keys())[0]
data = imported_data[sheet_name]
for i in range(3):
    # 选取数据
    selected_data = data.iloc[1:12, 1+3*i:4+3*i]
    data_array = selected_data.values
    # 对第一列除以100化为标准单位
    data_array[:, 0] = data_array[:, 0] / 100
    k_value, q_e_intercept, k_value_without_last_point, q_e_intercept_without_last_point, fit_coordinates, model, fit_coordinates_without_last_point, model_without_last_point, delta_theta_over_delta_q_list = filter_fit_line(data_array)
    print("拟合直线斜率：", k_value)
    print("拟合直线截距：", q_e_intercept)
    print("去掉最后一个点的拟合直线斜率：", k_value_without_last_point)
    print("去掉最后一个点的拟合直线截距：", q_e_intercept_without_last_point)
    save_path = f"./拟合图结果/{i+1}.png"  # 动态保存路径
    filter_plot(data_array, fit_coordinates, model, model_without_last_point, fit_coordinates[:, 0], delta_theta_over_delta_q_list, save_path)
