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
        r"./过滤实验数据处理 (1).py",
        r"./过滤实验数据处理 (2).py",
        r"./过滤实验数据处理 (3).py",
        r"./过滤实验数据处理 (4).py",
        r"./过滤实验数据处理 (5).py",
    ]

    for script in scripts:
        run_script(script)
