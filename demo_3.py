import matplotlib.pyplot as plt
import pandas as pd

# 读取结果表格
results_df = pd.read_excel('all_b_1_EDA_EEG_PACMI_outcome.xlsx')

# 获取不同 ID 的数据
id_b_1_1 = results_df[results_df['id'] == 'b_1_1']
id_b_1_3 = results_df[results_df['id'] == 'b_1_3']

# 设置图形
plt.figure(figsize=(12, 6))

# 绘制 b_1_1 的静息状态和实验状态
plt.bar(['Rest1'], id_b_1_1['rest'], color='red', label='b_1_1 Rest State MI')
plt.plot(range(len(id_b_1_1['exp'])), id_b_1_1['exp'], marker='o', color='black', linestyle='--', label='b_1_1 Experimental State MI')

# 绘制 b_1_3 的静息状态和实验状态
plt.bar(['Rest2'], id_b_1_3['rest'], color='green', label='b_1_3 Rest State MI')
plt.plot(range(len(id_b_1_3['exp'])), id_b_1_3['exp'], marker='o', color='black', linestyle='-', label='b_1_3 Experimental State MI')

# 设置图形属性
plt.xlabel('exp')
plt.ylabel('Modulation Index (MI)')
plt.title('Comparison of PAC between Different IDs')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# 显示图形
plt.show()
