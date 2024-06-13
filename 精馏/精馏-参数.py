# This project is created by @aFei-CQUT
# ------------------------------------------------------------------------------------------------------------------------------------
# About @ aFei-CQUT
# - Interests&Hobbies: Programing,  ChatGPT,  Reading serious books,  Studying academical papers.
# - CurrentlyLearning: Mathmodeling，Python and Mathmatica (preparing for National College Mathematical Contest in Modeling).
# - Email:2039787966@qq.com
# - Pronouns: Chemical Engineering, Computer Science, Enterprising, Diligent, Hard-working, Sophomore,Chongqing Institute of Technology,
# - Looking forward to collaborating on experimental data processing of chemical engineering principle
# ------------------------------------------------------------------------------------------------------------------------------------

α = 2.5
R = 2.0

sF = 21
sD = 90
sW = 40

ρA = 0.789
ρB = 1000

cA = 70
cB = 90

xA = 0.4
xB = 0.6

rA = 850
rB = 2260

MA = 46
MB = 18

cpm = xA * cA * MA + xB * cB * MB
rm = xA * rA * MA + xB * rB * MB

# 查询参数后代入
tS = None
tF = 31

q = (cpm * (tS - tF) + rm) / rm

F = 100
D = None
W = None

xF = (sF * ρA / MA) / ((sF * ρA / MA) + (1 - sF) * ρB / MB)
xD = (sD * ρA / MA) / ((sD * ρA / MA) + (1 - sD) * ρB / MB)
xW = (sW * ρA / MA) / ((sW * ρA / MA) + (1 - sW) * ρB / MB)