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
拿第一组数据建立基本流程
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.image as mpimg

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 定义函数：添加辅助线
def add_auxiliary_lines(q_list, delta_theta_over_delta_q_list):
    for i in range(len(q_list) - 1):
        plt.axvline(x=q_list[i], color='black', linestyle='dashed')
        plt.hlines(y=delta_theta_over_delta_q_list[i], xmin=q_list[i], xmax=q_list[i + 1], color='black')
        plt.axvline(x=q_list[i + 1], color='black', linestyle='dashed')

# 定义函数：初始拟合
def initial_fit(data, q_list, delta_theta_over_delta_q_list):
    # 构造数据对
    fit_data = np.column_stack((q_to_fit_list, delta_theta_over_delta_q_to_fit_list))
    
    # 执行线性回归
    model = LinearRegression()
    model.fit(fit_data[:, 0].reshape(-1, 1), fit_data[:, 1])
    
    return model, fit_data

# 定义函数：检测异常值
def detect_outliers(fit_data, threshold=2):
    # 计算Z-score
    z_scores = np.abs((fit_data[:, 1] - np.mean(fit_data[:, 1])) / np.std(fit_data[:, 1]))
    
    # 检测异常值
    outliers = np.where(z_scores > threshold)[0]
    
    return outliers

# 定义函数：排除异常值后重新拟合
def refit_after_outliers_removed(fit_data, outliers):
    filtered_data = np.delete(fit_data, outliers, axis=0)
    X = filtered_data[:, 0].reshape(-1, 1)
    y = filtered_data[:, 1]
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model, filtered_data, outliers

def export_to_excel(delta_theta_over_delta_q_to_fit_lists, q_to_fit_lists,fit_slopes, fit_intercepts,data_file, result_file):
    # 构建拟合数据和结果字典
    fit_data = {
        f'Δθ/Δq {i+1}': delta_theta_over_delta_q_to_fit_lists[i]
        for i in range(3)
    }
    fit_data.update({
        f'q {i+1}': q_to_fit_lists[i]
        for i in range(3)
    })

    fit_result = {
        '组号': [1, 2, 3],
        '拟合斜率': fit_slopes,
        '拟合截距': fit_intercepts
    }
    
    # 将字典转换为DataFrame
    df_data = pd.DataFrame(fit_data)
    df_result = pd.DataFrame(fit_result)
    
    # 使用pandas的ExcelWriter将数据写入Excel文件
    with pd.ExcelWriter(data_file, engine='xlsxwriter') as writer_data:
        df_data.to_excel(writer_data, index=False, sheet_name='拟合数据')
    
    with pd.ExcelWriter(result_file, engine='xlsxwriter') as writer_result:
        df_result.to_excel(writer_result, index=False, sheet_name='拟合结果')


# 定义空存储列表
delta_theta_over_delta_q_to_fit_lists = []
q_to_fit_lists = []
delta_theta_over_delta_q_to_refit_lists = []
q_to_refit_lists = []
initial_fit_slopes = []
initial_fit_intercepts = []
refit_slopes = []
refit_intercepts = []

# 主程序
if __name__ == '__main__':
    # 导入数据
    imported_data = pd.read_excel(r'./原始数据.xlsx', sheet_name=None)
    sheet_name = list(imported_data.keys())[0]
    data = imported_data[sheet_name]
    
    # 定义所有组数据的坐标轴范围
    plot_ranges_initial = [
        {'x_min': 0, 'x_max': 0.200, 'y_min': 0, 'y_max': 140000},
        {'x_min': 0, 'x_max': 0.200, 'y_min': 0, 'y_max': 25000},
        {'x_min': 0, 'x_max': 0.200, 'y_min': 0, 'y_max': 8000}
    ]
    
    plot_ranges_refit = [
        {'x_min': 0, 'x_max': 0.200, 'y_min': 0, 'y_max': 15000},
        {'x_min': 0, 'x_max': 0.200, 'y_min': 0, 'y_max': 5000},
        {'x_min': 0, 'x_max': 0.200, 'y_min': 0, 'y_max': 4000}
    ]
    
    # 对每组数据进行拟合和绘图
    for i in range(3):
        # 选取数据
        selected_data = data.iloc[1:12, 1 + 3*i:4 + 3*i]
        data_array = selected_data.values
        data_array[:, 0] = data_array[:, 0] / 100  # 将第一列转换为标准单位
    
        # 计算 deltaQ
        a = 0.0475
        deltaV = 9.446 * 10**-4
        deltaQ = deltaV / a
    
        # 计算 delta_theta_over_delta_q_list
        delta_theta_list = np.diff(data_array[:, 1])
        delta_q_list = np.full(len(delta_theta_list), deltaQ)
        delta_theta_over_delta_q_list = delta_theta_list / delta_q_list
    
        # 构造 q_list
        q_list = np.linspace(0, 0 + len(delta_theta_list) * deltaQ, len(delta_theta_list) + 1)
        
        # 准备数据进行拟合
        q_to_fit_list = (q_list[:-1] + q_list[1:]) / 2
        delta_theta_over_delta_q_to_fit_list = delta_theta_over_delta_q_list
        
        # 初始拟合
        model, fit_data = initial_fit(data_array, q_list, delta_theta_over_delta_q_list)
        k_value = model.coef_[0]
        q_e_intercept = model.intercept_
        
        # 记录初始拟合数据
        q_to_fit_lists.append(q_to_fit_list)
        delta_theta_over_delta_q_to_fit_lists.append(delta_theta_over_delta_q_to_fit_list)
        
        # 记录初拟合斜率和截距
        initial_fit_slopes.append(k_value)
        initial_fit_intercepts.append(q_e_intercept)
    
        # 输出初始拟合结果
        print(f'第{i+1}组数据初始拟合结果:')
        print('初始拟合斜率:', k_value)
        print('初始拟合截距:', q_e_intercept)
    
        # 绘制初始拟合图
        plt.figure(figsize=(10, 6))
        plt.scatter(fit_data[:, 0], fit_data[:, 1], color='red', label='拟合数据')
        plt.plot(fit_data[:, 0], model.predict(fit_data[:, 0].reshape(-1, 1)), color='blue', label='拟合线')
    
        # 添加辅助线和标记异常值
        add_auxiliary_lines(q_list, delta_theta_over_delta_q_list)
        outliers = detect_outliers(fit_data)
        plt.scatter(fit_data[outliers, 0], fit_data[outliers, 1], color='green', label='异常值')
    
        # 获取当前循环的坐标轴范围参数
        current_range_initial = plot_ranges_initial[i]
        plt.xlim(current_range_initial['x_min'], current_range_initial['x_max'])
        plt.ylim(current_range_initial['y_min'], current_range_initial['y_max'])
    
        plt.xlabel('q 值')
        plt.ylabel('Δθ/Δq')
        plt.legend(loc='upper left')
        plt.title(f'第{i+1}组数据初始拟合')
        plt.savefig(f'./过滤拟合图结果/{2*i+1}.png', dpi=300)
        plt.show()
    
        # 排除异常值后重新拟合
        if len(outliers) > 0:
            model, filtered_data, _ = refit_after_outliers_removed(fit_data, outliers)
            
            # 记录再拟合数据
            delta_theta_over_delta_q_to_refit_lists.append(filtered_data[:,1])
            q_to_refit_lists.append(filtered_data[:,0])
            
            # 记录再拟合斜率和截距
            refit_slope = model.coef_[0]
            refit_intercept = model.intercept_
            refit_slopes.append(refit_slope)
            refit_intercepts.append(refit_intercept)

            # 输出重新拟合结果
            print(f'第{i+1}组数据排除异常值后重新拟合结果:')
            print('排除异常值后斜率:', model.coef_[0])
            print('排除异常值后截距:', model.intercept_)
    
            # 绘制重新拟合图
            plt.figure(figsize=(10, 6))
            plt.scatter(filtered_data[:, 0], filtered_data[:, 1], color='red', label='拟合数据')
            plt.plot(filtered_data[:, 0], model.predict(filtered_data[:, 0].reshape(-1, 1)), color='blue', label='拟合线')
    
            # 添加辅助线
            add_auxiliary_lines(q_list, delta_theta_over_delta_q_list)
    
            # 获取当前循环的坐标轴范围参数
            current_range_refit = plot_ranges_refit[i]
            plt.xlim(current_range_refit['x_min'], current_range_refit['x_max'])
            plt.ylim(current_range_refit['y_min'], current_range_refit['y_max'])
    
            plt.xlabel('q 值')
            plt.ylabel('Δθ/Δq')
            plt.legend(loc='upper left')
            plt.title(f'第{i+1}组数据排除异常值后重新拟合')
            plt.savefig(f'./过滤拟合图结果/{2*i+2}.png', dpi=300)
            plt.show()
            
            
plt.figure(figsize=(10, 6))

for i in range(3):
    plt.scatter(q_to_fit_lists[i], delta_theta_over_delta_q_to_fit_lists[i], label=f'第{i+1}组数据')
    plt.plot(q_to_fit_lists[i], initial_fit_slopes[i] * q_to_fit_lists[i] + initial_fit_intercepts[i], label=f'拟合线{i+1}')
    # 添加辅助线
    add_auxiliary_lines(q_list, delta_theta_over_delta_q_to_fit_lists[i])

plt.xlim(0, current_range_refit['x_max'])
plt.xlabel('q 值')
plt.ylabel('Δθ/Δq')
plt.title('三组数据保留所有数据点初拟合对比')
plt.legend(loc='upper left')
plt.savefig('./过滤拟合图结果/7.png', dpi=300)
plt.show()
            
plt.figure(figsize=(10, 6))

q_list = q_list[:-1]
for i in range(3):
    plt.scatter(q_to_refit_lists[i], delta_theta_over_delta_q_to_refit_lists[i], label=f'第{i+1}组数据')
    plt.plot(q_to_refit_lists[i], refit_slopes[i] * q_to_refit_lists[i] + refit_intercepts[i], label=f'拟合线{i+1}')
    # 添加辅助线
    add_auxiliary_lines(q_list, delta_theta_over_delta_q_to_refit_lists[i])

plt.xlim(0, 0.200)
plt.xlabel('q 值')
plt.ylabel('Δθ/Δq')
plt.title('三组数据排除异常值后再拟合对比')
plt.legend(loc='upper left')
plt.savefig('./过滤拟合图结果/8.png', dpi=300)
plt.show()

# 读取图像文件
images = []
for i in range(1, 9):
    img = mpimg.imread(f'./过滤拟合图结果/{i}.png')
    images.append(img)

# 创建一个4x2布局的图
fig, axes = plt.subplots(4, 2, figsize=(10, 12))

# 遍历每个子图并显示相应的图像
for ax, img in zip(axes.flatten(), images):
    ax.imshow(img)
    ax.axis('off')  # 隐藏坐标轴

# 调整布局，减少图像之间的间隙
plt.subplots_adjust(wspace=0.01, hspace=0.01)

# 保存图像并移除多余的边框
plt.savefig(r'./过滤拟合图结果/拟合图整合图.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()


# 导出初拟合数据和结果
export_to_excel(delta_theta_over_delta_q_to_fit_lists, q_to_fit_lists, initial_fit_slopes, initial_fit_intercepts,
                r'./过滤输出文件/过滤初拟合数据.xlsx', r'./过滤输出文件/过滤初拟合结果.xlsx')

# 导出再拟合数据和结果
export_to_excel(delta_theta_over_delta_q_to_refit_lists, q_to_refit_lists, refit_slopes, refit_intercepts,
                r'./过滤输出文件/过滤再拟合数据.xlsx', r'./过滤输出文件/过滤再拟合结果.xlsx')