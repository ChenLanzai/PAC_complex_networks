import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# 读取结果表格
results_df = pd.read_excel('all_b_1_EDA_EEG_PACMI_outcome.xlsx')

# 初始化静息状态和实验状态的MI值列表
rest_mi_values = []
exp_mi_values = []

# 获取每个 ID 的静息状态和实验状态的 MI 值
for id_value in ['b_1_1', 'b_2_1', 'b_3_1', 'b_4_1', 'b_5_1', 'b_6_1', 'b_8_1', 'b_9_1', 'b_10_1']:
    rest_value = results_df.loc[results_df['id'] == id_value, 'rest'].values[0]  # 静息状态MI值
    exp_values = results_df.loc[results_df['id'] == id_value, 'exp'].dropna().values  # 剔除缺失值的实验状态MI值
    rest_mi_values.append(rest_value)
    exp_mi_values.extend(exp_values)  # 将实验状态MI值添加到列表中

    # 打印读取到的值
    print(f"ID: {id_value}, Rest MI: {rest_value}, Exp MI: {exp_values}")

# 转换为NumPy数组
rest_mi_values = np.array(rest_mi_values)
exp_mi_values = np.array(exp_mi_values)

# 可视化原始数据分布
plt.figure(figsize=(10, 6))
plt.boxplot([rest_mi_values, exp_mi_values], labels=['Rest', 'Exp'])
plt.ylabel('Modulation Index (MI)')
plt.title('Comparison of Original Rest and Experimental MI Values')
plt.grid()
plt.show()

# 数据预处理：去除异常值
def remove_outliers(data):
    # 计算四分位数
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    # 定义异常值的范围
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 返回去除异常值后的数据
    return data[(data >= lower_bound) & (data <= upper_bound)]

# 去除静息状态和实验状态的异常值
rest_mi_values_cleaned = remove_outliers(rest_mi_values)
exp_mi_values_cleaned = remove_outliers(exp_mi_values)

# 打印清理后的数据分布
print("清理后的静息状态MI值:", rest_mi_values_cleaned)
print("清理后的实验状态MI值:", exp_mi_values_cleaned)

# 可视化清理后的数据分布
plt.figure(figsize=(10, 6))
plt.boxplot([rest_mi_values_cleaned, exp_mi_values_cleaned], labels=['Rest', 'Exp'])
plt.ylabel('Modulation Index (MI)')
plt.title('Comparison of Cleaned Rest and Experimental MI Values')
plt.grid()
plt.show()

# 检查是否有有效的实验状态值
if len(exp_mi_values_cleaned) == 0:
    print("没有有效的实验状态MI值，无法进行显著性检验。")
else:
    # Mann-Whitney U 检验
    stat, p_value = mannwhitneyu(rest_mi_values_cleaned, exp_mi_values_cleaned, alternative='two-sided')

    print(f"U统计量: {stat}")
    print(f"p值: {p_value}")

    # 根据p值判断显著性
    alpha = 0.05
    if p_value < alpha:
        print("结果显著：静息状态和实验状态的MI值有显著差异")
    else:
        print("结果不显著：静息状态和实验状态的MI值无显著差异")
