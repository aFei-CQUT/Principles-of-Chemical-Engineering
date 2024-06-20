# This project is created by @aFei-CQUT
# ------------------------------------------------------------------------------------------------------------------------------------
# About @ aFei-CQUT
# - Interests&Hobbies: Programing,  ChatGPT,  Reading serious books,  Studying academical papers.
# - CurrentlyLearning: Mathmodeling，Python and Mathmatica (preparing for National College Mathematical Contest in Modeling).
# - Email:2039787966@qq.com
# - Pronouns: Chemical Engineering, Computer Science, Enterprising, Diligent, Hard-working, Sophomore,Chongqing Institute of Technology,
# - Looking forward to collaborating on experimental data processing of chemical engineering principle
# ------------------------------------------------------------------------------------------------------------------------------------
'''
保留所有数据点的拟合线
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 定义函数：拟合直线
def filter_fit_line(selected_data):
    # 滤膜参数
    a = 0.0475
    deltaV = 9.446 * 10**-4
    deltaQ = deltaV / a
    # 计算 Δθ/Δq
    delta_theta_list = np.diff(selected_data[:, 1])
    delta_q_list = np.full(len(delta_theta_list), deltaQ)
    delta_theta_over_delta_q_list = delta_theta_list / delta_q_list
    # 计算 q 值
    q_list = np.linspace(0, 0 + len(delta_theta_list) * deltaQ, len(delta_theta_list) + 1)
    q_mean_to_fit_line_list = (q_list[:-1] + q_list[1:]) / 2
    delta_theta_over_delta_q_mean_to_fit_line_list = delta_theta_over_delta_q_list
    fit_coordinates = np.column_stack((q_mean_to_fit_line_list, delta_theta_over_delta_q_mean_to_fit_line_list))
    # 线性回归拟合
    model = LinearRegression()
    model.fit(fit_coordinates[:, 0].reshape(-1, 1), fit_coordinates[:, 1])
    k_value = model.coef_[0]
    q_e_intercept = model.intercept_
    # 为了绘制完整曲线，将最后一个点重复添加
    delta_theta_over_delta_q_list = np.append(delta_theta_over_delta_q_list, delta_theta_over_delta_q_list[-1])
    return k_value, q_e_intercept, fit_coordinates, model, delta_theta_over_delta_q_list

# 定义函数：绘制图表
def filter_plot(selected_data_list, fit_coordinates_list, fit_line_list, delta_theta_over_delta_q_list_list, save_path):
    a = 0.0475
    deltaV = 9.446 * 10**-4
    deltaQ = deltaV / a
    q_list = np.linspace(0, 0 + len(selected_data_list[0]) * deltaQ, len(selected_data_list[0]) + 1)
    plt.figure(figsize=(10, 6))
    
    for i in range(len(selected_data_list)):
        # 绘制拟合数据集散点和拟合直线
        plt.scatter(fit_coordinates_list[i][:, 0], fit_coordinates_list[i][:, 1], label='拟合数据点集' + str(i+1))
        plt.plot(fit_coordinates_list[i][:, 0], fit_line_list[i].predict(fit_coordinates_list[i][:, 0].reshape(-1, 1)), label='保留所有数据点的拟合线' + str(i+1))
    
    for i in range(len(q_list) - 2):
        for j in range(len(selected_data_list)):
            # 绘制Δθ/Δq的水平线
            plt.hlines(y=delta_theta_over_delta_q_list_list[j][i], xmin=q_list[i], xmax=q_list[i + 1], color='black', alpha=0.5)
        # 绘制垂直虚线
        plt.axvline(x=q_list[i], color='black', linestyle='dashed')
        plt.axvline(x=q_list[i + 1], color='black', linestyle='dashed')

    # 强制刷新绘图
    plt.draw()
    xmin, xmax = plt.xlim()
    plt.xlim(xmin,0.200)
    # 添加图例
    plt.xlabel('q 值')
    plt.ylabel('Δθ/Δq')
    plt.legend(loc='upper left')
    # 添加标题
    plt.title('保留所有数据点的拟合线')
    # 保存图像
    plt.savefig(save_path,dpi=300)
    # 显示图像
    plt.show()
    # 关闭图像以释放内存
    plt.close()

# 设置字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 设置图片分辨率
plt.rcParams['figure.dpi'] = 125
# 导入数据
imported_data = pd.read_excel(r"./过滤实验数据-非.xlsx", sheet_name=None)
sheet_name = list(imported_data.keys())[0]
data = imported_data[sheet_name]
# 创建空列表
selected_data_list = []
fit_coordinates_list = []
delta_theta_over_delta_q_list_list = []
fit_line_list = []

# 对每组数据进行拟合和绘图
for i in range(3):
    # 选取数据
    selected_data = data.iloc[1:12, 1+3*i:4+3*i]
    data_array = selected_data.values
    # 对第一列除以100化为标准单位
    data_array[:, 0] = data_array[:, 0] / 100
    # 拟合直线
    k_value, q_e_intercept, fit_coordinates, model, delta_theta_over_delta_q_list = filter_fit_line(data_array)
    fit_coordinates_list.append(fit_coordinates)
    fit_line_list.append(model)
    delta_theta_over_delta_q_list_list.append(delta_theta_over_delta_q_list)
    selected_data_list.append(data_array)
    # 输出拟合结果
    print("拟合直线斜率：", k_value)
    print("拟合直线截距：", q_e_intercept)

# 绘制图表并保存
save_path = "./拟合图结果/4.png"
filter_plot(selected_data_list, fit_coordinates_list, fit_line_list, delta_theta_over_delta_q_list_list, save_path)
