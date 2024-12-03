import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import zscore


# 计算PAC和PPC
def compute_coupling(phase, amplitude):
    pac = np.abs(np.mean(amplitude * np.exp(1j * phase)))
    ppc = np.abs(np.mean(np.exp(1j * (phase - np.angle(hilbert(amplitude))))))
    return pac, ppc


# 生成空的耦合矩阵
def initialize_coupling_matrix(num_signals):
    return np.zeros((num_signals, num_signals))


# 处理每个时间窗口
def process_window(window_signals, num_signals, pac_matrix, ppc_matrix):
    for j in range(num_signals):
        for k in range(num_signals):
            if j != k:
                phase_j = np.angle(hilbert(window_signals[j]))
                amplitude_k = np.abs(hilbert(window_signals[k]))

                # 计算PAC和PPC
                pac, ppc = compute_coupling(phase_j, amplitude_k)

                # 计算p值
                null_distribution_pac, null_distribution_ppc = compute_p_values(window_signals[j], window_signals[k])

                p_value_pac = np.mean(null_distribution_pac >= pac)
                p_value_ppc = np.mean(null_distribution_ppc >= ppc)

                # 应用Bonferroni校正
                alpha = 0.05
                num_comparisons = num_signals * (num_signals - 1) // 2
                corrected_alpha = alpha / num_comparisons

                # 更新PAC和PPC矩阵
                if p_value_pac < corrected_alpha:
                    pac_matrix[j, k] = pac
                if p_value_ppc < corrected_alpha:
                    ppc_matrix[j, k] = ppc


# 计算p值
def compute_p_values(signal_a, signal_b, num_bootstrap=500):
    null_distribution_pac = []
    null_distribution_ppc = []

    for _ in range(num_bootstrap):
        bootstrap_phase = np.angle(hilbert(np.random.choice(signal_a, len(signal_a), replace=True)))
        null_distribution_pac.append(compute_coupling(bootstrap_phase, signal_b)[0])
        bootstrap_phase2 = np.angle(hilbert(np.random.choice(signal_b, len(signal_b), replace=True)))
        null_distribution_ppc.append(compute_coupling(bootstrap_phase, bootstrap_phase2)[1])

    return null_distribution_pac, null_distribution_ppc


# 绘制耦合网络图
def plot_coupling_matrix(matrix, title, signal_labels):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='Blues', interpolation='none')
    plt.colorbar(label='Coupling Strength')
    plt.title(title)
    plt.xticks(np.arange(len(signal_labels)), signal_labels, rotation=45)
    plt.yticks(np.arange(len(signal_labels)), signal_labels)
    plt.show()


# 读取Excel文件
rest_file_path = './data/rest_b_1_1_process.xlsx'
exp_file_path = './data/exp_b_1_1_process.xlsx'

df_rest = pd.read_excel(rest_file_path)
df_exp = pd.read_excel(exp_file_path)

# 提取数据列并标准化
signals_rest = [zscore(df_rest[col].values) for col in ['PPG', 'EMG', 'EEG', 'EDA', 'ECG']]
signals_exp = [zscore(df_exp[col].values) for col in ['PPG', 'EMG', 'EEG', 'EDA', 'ECG']]
num_signals = len(signals_rest)
signal_labels = ['PPG', 'EMG', 'EEG', 'EDA', 'ECG']

# 定义时间窗口大小
T_rest = len(signals_rest[0])  # 静息状态完整持续时间为1分钟
T_exp = 72000  # 实验状态1分钟的采样点数

# 处理静息状态
pac_matrix_rest = initialize_coupling_matrix(num_signals)
ppc_matrix_rest = initialize_coupling_matrix(num_signals)
process_window(signals_rest, num_signals, pac_matrix_rest, ppc_matrix_rest)

# 绘制静息状态耦合网络图
plot_coupling_matrix(pac_matrix_rest, 'PAC Coupling (Rest)', signal_labels)
plot_coupling_matrix(ppc_matrix_rest, 'PPC Coupling (Rest)', signal_labels)

# 实验状态窗口数量
num_windows_exp = len(signals_exp[0]) // T_exp

# 处理实验状态
for window_idx in range(num_windows_exp):
    start_idx = window_idx * T_exp
    end_idx = start_idx + T_exp

    window_signals_exp = [signal[start_idx:end_idx] for signal in signals_exp]
    pac_matrix_exp = initialize_coupling_matrix(num_signals)
    ppc_matrix_exp = initialize_coupling_matrix(num_signals)
    process_window(window_signals_exp, num_signals, pac_matrix_exp, ppc_matrix_exp)

    # 绘制实验状态耦合网络图
    plot_coupling_matrix(pac_matrix_exp, f'PAC Coupling (Experiment, Window {window_idx + 1})', signal_labels)
    plot_coupling_matrix(ppc_matrix_exp, f'PPC Coupling (Experiment, Window {window_idx + 1})', signal_labels)
