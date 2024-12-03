import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import zscore
import matplotlib.pyplot as plt
import networkx as nx

# 加载数据
rest_data = pd.read_excel('./data/rest_b_1_1_process.xlsx')  # 静息状态数据
exp_data = pd.read_excel('./data/exp_b_1_1_process.xlsx')  # 实验状态数据

# 提取信号并进行z-score标准化
signals = ['EEG', 'ECG', 'EDA', 'PPG', 'EMG']
normalized_signals = {signal: zscore(rest_data[signal].values) for signal in signals}  # 静息状态信号
normalized_signals_exp = {signal: zscore(exp_data[signal].values) for signal in signals}  # 实验状态信号

# 采样频率和分箱时间
fs = 1200
bin_duration = 1 * 60  # 每个分箱包含的时间长度为1分钟
bin_samples = bin_duration * fs  # 每个分箱包含的样本数

# 定义函数计算调制指数（MI）
def modulation_index(phase, amplitude, n_bins=36):
    hist, bin_edges = np.histogram(phase, bins='auto')
    optimal_bins = len(hist[hist > 0])
    n_bins = min(n_bins, optimal_bins)
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    phase_bins = np.digitize(phase, bin_edges) - 1
    phase_bin_mean_amplitudes = [amplitude[phase_bins == i].mean() if np.any(phase_bins == i) else 0 for i in range(n_bins)]
    phase_bin_mean_amplitudes = np.array(phase_bin_mean_amplitudes)
    phase_bin_mean_amplitudes /= np.sum(phase_bin_mean_amplitudes)  # 归一化
    uniform_distribution = np.ones(n_bins) / n_bins
    if n_bins == 0:
        return 0
    mi = np.sum(phase_bin_mean_amplitudes * np.log(phase_bin_mean_amplitudes / uniform_distribution))
    return mi, n_bins  # 返回MI值和分箱数

# 设定显著性水平
alpha = 0.05  # 原始显著性水平
m = len(signals) * (len(signals) - 1)  # 比较的总数
adjusted_significance_level = alpha / m  # Bonferroni 校正后的显著性水平

# 计算静息状态的PAC矩阵
rest_pac_matrix = np.zeros((len(signals), len(signals)))  # 初始化静息状态PAC矩阵

for i in range(len(signals)):
    signal_a = normalized_signals[signals[i]]
    phase_a = np.angle(hilbert(signal_a))
    for j in range(len(signals)):
        if i != j:  # 不计算同一信号之间的PAC
            signal_b = normalized_signals[signals[j]]
            amplitude_b = np.abs(hilbert(signal_b))  # 使用同一信号计算幅度
            pac_value, n_bins = modulation_index(phase_a, amplitude_b)
            rest_pac_matrix[i, j] = pac_value  # 保留原始MI值

# 打印静息状态PAC矩阵
print("Rest State PAC Matrix:")
print(rest_pac_matrix)  # 保留原始MI值

# 计算实验状态的PAC（按1分钟分箱）
exp_pac_matrices = []  # 存储实验状态的多个PAC矩阵
coupling_counts = {f"{signals[i]}→{signals[j]}": 0 for i in range(len(signals)) for j in range(len(signals)) if i != j}  # 初始化耦合计数

for k in range(0, len(normalized_signals_exp[signals[0]]), bin_samples):
    exp_bin_matrix = np.zeros((len(signals), len(signals)))  # 初始化实验状态PAC矩阵
    for i in range(len(signals)):
        exp_bin_signal_a = normalized_signals_exp[signals[i]][k:k + bin_samples]
        if len(exp_bin_signal_a) < bin_samples:
            print(f"Slice {k // bin_samples + 1} is skipped due to insufficient data.")
            break  # 跳过不足一个完整分箱的数据
        phase_a = np.angle(hilbert(exp_bin_signal_a))
        for j in range(len(signals)):
            if i != j:  # 不计算同一信号之间的PAC
                exp_bin_signal_b = normalized_signals_exp[signals[j]][k:k + bin_samples]
                amplitude_b = np.abs(hilbert(exp_bin_signal_b))  # 使用当前切片计算幅度
                pac_value, n_bins = modulation_index(phase_a, amplitude_b)
                exp_bin_matrix[i, j] = pac_value  # 保留原始MI值
                
                # 统计耦合值
                if pac_value > adjusted_significance_level:  # 只统计显著的耦合值
                    coupling_counts[f"{signals[i]}→{signals[j]}"] += 1  # 记录信号i调制信号j的耦合

        print(f"Slice {k // bin_samples + 1}, Signal Pair ({signals[i]}, {signals[j]}): Number of Bins = {n_bins}")  # 打印分箱数
    exp_pac_matrices.append(exp_bin_matrix)

# 打印实验状态的PAC矩阵
for idx, matrix in enumerate(exp_pac_matrices):
    if np.any(matrix):  # 只打印有有效值的矩阵
        print(f"Experimental State PAC Matrix for Slice {idx + 1}:")
        print(matrix)  # 保留原始MI值

# 创建静息状态PAC矩阵的可视化
def plot_pac_matrix(matrix, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='Blues', aspect='auto')  # 使用蓝色调的矩阵图
    plt.colorbar(label='Coupling Value (MI)')
    plt.xticks(np.arange(len(signals)), signals, rotation=45)
    plt.yticks(np.arange(len(signals)), signals)
    plt.title(title)
    plt.xlabel('Signals')
    plt.ylabel('Signals')
    plt.grid(False)

    # 在矩阵中添加MI值
    for i in range(len(signals)):
        for j in range(len(signals)):
            if i != j:  # 只在非对角线位置添加MI值
                plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='black')

    plt.show()

# 绘制静息状态PAC矩阵图
plot_pac_matrix(rest_pac_matrix, 'Rest State PAC Matrix')

# 绘制实验状态PAC矩阵图
for idx, matrix in enumerate(exp_pac_matrices):
    if np.any(matrix):  # 只绘制有有效值的矩阵
        plot_pac_matrix(matrix, f'Experimental State PAC Matrix (Slice {idx + 1})')

# 绘制耦合统计条形图
def plot_coupling_statistics(coupling_counts):
    # 过滤掉耦合计数为0的信号对
    filtered_counts = {k: v for k, v in coupling_counts.items() if v > 0}
    
    plt.figure(figsize=(10, 6))
    plt.bar(filtered_counts.keys(), filtered_counts.values(), color='skyblue')
    plt.title('Coupling Statistics of Signal Pairs')
    plt.xlabel('Signal Pairs (Phase Modulating → Amplitude Modulated)')
    plt.ylabel('Number of Significant Couplings')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# 绘制耦合统计条形图
plot_coupling_statistics(coupling_counts)