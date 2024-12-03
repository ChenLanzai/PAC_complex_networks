import numpy as np
import pandas as pd
from scipy.signal import hilbert, hilbert2
from scipy.stats import zscore
import matplotlib.pyplot as plt

# 加载数据
rest_data = pd.read_excel('./data/rest_b_8_1_process.xlsx')
exp_data_sheet1 = pd.read_excel('./data/exp_b_8_1_process.xlsx', sheet_name='Sheet1')
# exp_data_sheet2 = pd.read_excel('./data/exp_b_8_1_process.xlsx', sheet_name='Sheet2')

# 合并实验状态的两种生理信号数据
# exp_data = pd.concat([exp_data_sheet1[['EEG', 'EDA']], exp_data_sheet2[['EEG', 'EDA']]], ignore_index=True)
exp_data = exp_data_sheet1


# 提取EEG和EDA信号，并进行z-score标准化
rest_eeg = zscore(rest_data['EEG'].values)
rest_eda = zscore(rest_data['EDA'].values)
exp_eeg = zscore(exp_data['EEG'].values)
exp_eda = zscore(exp_data['EDA'].values)

# 采样频率和分箱时间
fs = 1200
bin_duration = 1 * 60  # 每个分箱包含的时间长度为1分钟
bin_samples = bin_duration * fs  # 每个分箱包含的样本数


# 定义函数计算调制指数（MI）
def modulation_index(phase, amplitude, n_bins=36):
    # 计算相位分布的直方图
    hist, bin_edges = np.histogram(phase, bins='auto')
    optimal_bins = len(hist[hist > 0])  # 选择有数据的分箱数量
    n_bins = min(n_bins, optimal_bins)  # 使用较小的值作为分箱数量

    # 打印实际用到的分箱数
    print(f"使用的分箱数: {n_bins}")

    # 用选定的分箱数量重新计算 MI
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    phase_bins = np.digitize(phase, bin_edges) - 1
    phase_bin_mean_amplitudes = [amplitude[phase_bins == i].mean() if np.any(phase_bins == i) else 0 for i in
                                 range(n_bins)]
    phase_bin_mean_amplitudes = np.array(phase_bin_mean_amplitudes)
    phase_bin_mean_amplitudes /= np.sum(phase_bin_mean_amplitudes)  # 归一化
    uniform_distribution = np.ones(n_bins) / n_bins

    # 如果 n_bins 为 0，返回 0
    if n_bins == 0:
        return 0

    mi = np.sum(phase_bin_mean_amplitudes * np.log(phase_bin_mean_amplitudes / uniform_distribution))
    return mi


# 计算静息状态的PAC
rest_eda_phase = np.angle(hilbert(rest_eda))  # 提取EDA瞬时相位
rest_eeg_envelope = np.abs(hilbert2(rest_eeg)).squeeze()  # 获取EEG上包络的瞬时幅度并转换为一维数组
rest_mi = modulation_index(rest_eda_phase, rest_eeg_envelope)

# 计算实验状态的PAC（按1分钟分箱）
exp_mi_values = []
for i in range(0, len(exp_eda), bin_samples):
    exp_bin_eda = exp_eda[i:i + bin_samples]
    exp_bin_eeg = exp_eeg[i:i + bin_samples]
    if len(exp_bin_eda) < bin_samples or len(exp_bin_eeg) < bin_samples:
        break  # 跳过不足一个完整分箱的数据

    exp_bin_eda_phase = np.angle(hilbert(exp_bin_eda))
    exp_bin_eeg_envelope = np.abs(hilbert2(exp_bin_eeg)).squeeze()

    # 检查并处理 NaN 值
    if np.any(np.isnan(exp_bin_eda_phase)) or np.any(np.isnan(exp_bin_eeg_envelope)):
        print(f"跳过段 {i // bin_samples + 1}，因为包含 NaN 值")
        continue  # 跳过包含 NaN 的段
    
    exp_mi = modulation_index(exp_bin_eda_phase, exp_bin_eeg_envelope, n_bins=36)

    # 如果 MI 为 0，表示无效
    if exp_mi == 0:
        exp_mi_values.append(0)
    else:
        exp_mi_values.append(exp_mi)

    # 打印每一小段的MI值
    print(f"实验状态数据段 {i // bin_samples + 1} 的 MI 值: {exp_mi}")

# 创建结果表格并保存到 Excel
id_input = input("请输入自定义ID（例如：参与者编号）：")  # 获取用户输入的ID
new_data = {
    'id': [id_input] * len(exp_mi_values),  # 将ID重复以匹配实验状态的MI值数量
    'rest': [rest_mi] * len(exp_mi_values),  # 将静息状态MI值扩展为与实验状态MI值数量相同
    'exp': exp_mi_values  # 存储所有实验状态的MI值
}

# 创建 DataFrame
new_results_df = pd.DataFrame(new_data)

# 尝试读取现有的 Excel 文件
try:
    existing_df = pd.read_excel('all_b_1_EDA_EEG_PACMI_outcome.xlsx')
    # 将新数据附加到现有数据
    combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)
except FileNotFoundError:
    # 如果文件不存在，则使用新数据
    combined_df = new_results_df

# 保存到 Excel 文件
combined_df.to_excel('EDA_EEG_PACMI_outcome.xlsx', index=False)

# 绘制静息和实验状态的MI对比
plt.figure(figsize=(10, 6))
plt.bar(['rest'], [rest_mi], color='y', label='Rest State MI')
plt.plot(range(len(exp_mi_values)), exp_mi_values, marker='o', color='r', label='Experimental State MI (1-min bins)')
plt.xlabel('exp')
plt.ylabel('Modulation Index (MI)')
plt.title('Comparison of PAC between Rest and Experimental States')
plt.legend()
plt.show()
