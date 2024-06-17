# This project is created by aFei-CQUT
# ------------------------------------------------------------------------------------------------------------------------------------
#   About aFei-CQUT
# - Interests&Hobbies: Programing,  ChatGPT,  Reading serious books,  Studying academical papers.
# - CurrentlyLearning: Mathmodeling，Python and Mathmatica (preparing for National College Mathematical Contest in Modeling).
# - Email:2039787966@qq.com
# - Pronouns: Chemical Engineering, Computer Science, Enterprising, Diligent, Hard-working, Sophomore,Chongqing Institute of Technology,
# - Looking forward to collaborating on experimental data processing of chemical engineering principle
# ------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 定义图片路径和文件名列表
folder_path = "D:/spyNow/化工原理实验/过滤/拟合图结果/"
image_files = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png"]

# 创建一个 3x2 的子图布局
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

# 遍历图片文件并将其显示在相应的子图中
for i, ax in enumerate(axes.flat):
    if i < len(image_files):
        img = mpimg.imread(folder_path + image_files[i])
        ax.imshow(img)
        ax.axis('off')  # 关闭坐标轴
    else:
        ax.axis('off')  # 如果图片不够填满子图，关闭多余的子图

# 调整布局，避免重叠
plt.tight_layout()

# 保存图片
plt.savefig("./拟合图结果/7.png", dpi=300)

# 显示合成的子图
plt.show()
