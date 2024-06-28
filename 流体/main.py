# This project is created by @aFei-CQUT
# ------------------------------------------------------------------------------------------------------------------------------------
# About @ aFei-CQUT
# - Interests&Hobbies: Programing,  ChatGPT,  Reading serious books,  Studying academical papers.
# - CurrentlyLearning: Mathmodeling，Python and Mathmatica (preparing for National College Mathematical Contest in Modeling).
# - Email:2039787966@qq.com
# - Pronouns: Chemical Engineering, Computer Science, Enterprising, Diligent, Hard-working, Sophomore,Chongqing Institute of Technology,
# - Looking forward to collaborating on experimental data processing of chemical engineering principle
# ------------------------------------------------------------------------------------------------------------------------------------

import subprocess

def run_script(script_path):
    try:
        result = subprocess.run(['python', script_path], check=True, capture_output=True, text=True)
        print(f"Script {script_path} executed successfully.")
        print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error executing script {script_path}.")
        print("Error message:\n", e.stderr)

if __name__ == "__main__":
    scripts = [
        r"./离心泵特性曲线拟合.py",
        r"./流体阻力曲线拟合.py",
    ]
    

    '''
    拟合图结果压缩
    '''
    import zipfile
    import os

    # 定义要压缩的目录
    directory_to_zip = r'./拟合图结果'

    # 定义压缩文件的名称
    zip_file_name = r'./拟合图结果.zip'

    # 创建ZipFile对象
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:

        # 遍历目录
        for root, dirs, files in os.walk(directory_to_zip):
            for file in files:

                # 创建相对文件路径并将其写入zip文件
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory_to_zip)
                zipf.write(file_path, arcname)

    print(f'压缩完成，文件保存为: {zip_file_name}')

    for script in scripts:
        run_script(script)
