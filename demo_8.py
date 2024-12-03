import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import zscore
import networkx as nx  # 用于模块度和聚集系数计算
import openpyxl  # 确保支持 Excel 文件的追加写入

# 计算PPC
def compute_ppc(phase1, phase2):
    return np.abs(np.mean(np.exp(1j * (phase1 - phase2))))

# 初始化PPC矩阵
def initialize_ppc_matrix(num_signals):
    return np.zeros((num_signals, num_signals))

# 统计网络特征
def compute_network_features(matrix):
    """计算全局特征和局部特征"""
    features = {}

    # 将矩阵转为 NetworkX 图
    G = nx.from_numpy_array(matrix, create_using=nx.Graph)

    # 全局特征
    features['density'] = nx.density(G)
    features['average_strength'] = np.mean(matrix[matrix > 0]) if np.any(matrix > 0) else 0

    # 度中心性（局部特征）
    degree_centrality = np.sum(matrix > 0, axis=1)
    features.update({f'degree_centrality_{i}': dc for i, dc in enumerate(degree_centrality)})

    # 模块度
    partition = nx.community.greedy_modularity_communities(G)
    modularity = nx.algorithms.community.modularity(G, partition)
    features['modularity'] = modularity

    # 全局聚集系数
    features['global_clustering'] = nx.transitivity(G)

    return features

# 对每个窗口处理PPC
def process_window(window_signals, hilbert_cache, num_signals, ppc_matrix, signal_labels, title):
    for j in range(num_signals):
        phase_j = np.angle(hilbert_cache[j])
        for k in range(num_signals):  # 考虑非对称性
            if j != k:  # 自己与自己不计算
                phase_k = np.angle(hilbert_cache[k])

                # 计算PPC
                ppc = compute_ppc(phase_j, phase_k)

                # 计算p值
                null_distribution_ppc = compute_p_values(window_signals[j], window_signals[k])
                p_value_ppc = np.mean(null_distribution_ppc >= ppc)

                # 应用Bonferroni校正
                alpha = 0.05
                num_comparisons = num_signals * (num_signals - 1)
                corrected_alpha = alpha / num_comparisons

                # 更新PPC矩阵
                if p_value_ppc < corrected_alpha:
                    ppc_matrix[j, k] = ppc
                else:
                    ppc_matrix[j, k] = 0  # 无边时为0

    # 绘制实时PPC矩阵
    plot_ppc_matrix(ppc_matrix, title, signal_labels)

# 计算p值
def compute_p_values(signal_a, signal_b, num_bootstrap=500):
    null_distribution_ppc = []

    for _ in range(num_bootstrap):
        bootstrap_phase_a = np.angle(hilbert(np.random.choice(signal_a, len(signal_a), replace=True)))
        bootstrap_phase_b = np.angle(hilbert(np.random.choice(signal_b, len(signal_b), replace=True)))
        null_distribution_ppc.append(compute_ppc(bootstrap_phase_a, bootstrap_phase_b))

    return null_distribution_ppc

# 绘制PPC矩阵
def plot_ppc_matrix(matrix, title, signal_labels):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='Blues', interpolation='none', vmin=0, vmax=1)

    # 在矩阵图上显示值
    for i in range(len(signal_labels)):
        for j in range(len(signal_labels)):
            text = f"{matrix[i, j]:.2f}" if matrix[i, j] > 0 else "0"
            plt.text(j, i, text, ha='center', va='center', color='black', fontsize=8)

    plt.colorbar(label='PPC Strength')
    plt.title(title)
    plt.xticks(np.arange(len(signal_labels)), signal_labels, rotation=45)
    plt.yticks(np.arange(len(signal_labels)), signal_labels)
    plt.show()

# 读取Excel文件
rest_file_path = './data/rest_b_2_1_process.xlsx'
exp_file_path = './data/exp_b_2_1_process.xlsx'

# 读取静息状态数据
df_rest = pd.read_excel(rest_file_path)
# 读取实验状态数据，处理Sheet1和Sheet2
try:
    sheet_names = pd.ExcelFile(exp_file_path).sheet_names
    sheet1_data = pd.read_excel(exp_file_path, sheet_name="Sheet1") if "Sheet1" in sheet_names else None
    sheet2_data = pd.read_excel(exp_file_path, sheet_name="Sheet2") if "Sheet2" in sheet_names else None
    df_exp = pd.concat([df for df in [sheet1_data, sheet2_data] if df is not None], ignore_index=True)
except ValueError:
    raise ValueError("实验状态文件格式有误，请检查是否包含 Sheet1 和 Sheet2 工作表！")

# 提取数据列并标准化
signals_rest = [zscore(df_rest[col].values) for col in ['PPG', 'EMG', 'EEG', 'EDA', 'ECG']]
signals_exp = [zscore(df_exp[col].values) for col in ['PPG', 'EMG', 'EEG', 'EDA', 'ECG']]
num_signals = len(signals_rest)
signal_labels = ['PPG', 'EMG', 'EEG', 'EDA', 'ECG']

# 定义时间窗口大小
T_rest = len(signals_rest[0])  # 静息状态完整持续时间为1分钟
T_exp = 72000  # 实验状态1分钟的采样点数

# 保存特征结果的列表
features_results = []

# 处理静息状态
ppc_matrix_rest = initialize_ppc_matrix(num_signals)
hilbert_cache_rest = [hilbert(signal[:T_rest]) for signal in signals_rest]
process_window(signals_rest, hilbert_cache_rest, num_signals, ppc_matrix_rest, signal_labels, 'PPC Coupling (Rest)')
np.save('b_2_1_ppc_matrix_rest.npy', ppc_matrix_rest)
features_rest = compute_network_features(ppc_matrix_rest)
features_rest['id'] = 'rest'
features_results.append(features_rest)

# 处理实验状态
num_windows_exp = len(signals_exp[0]) // T_exp
ppc_matrices_exp = []

for window_idx in range(num_windows_exp):
    start_idx = window_idx * T_exp
    end_idx = start_idx + T_exp
    window_signals_exp = [signal[start_idx:end_idx] for signal in signals_exp]
    hilbert_cache_exp = [hilbert(signal) for signal in window_signals_exp]
    ppc_matrix_exp = initialize_ppc_matrix(num_signals)
    process_window(window_signals_exp, hilbert_cache_exp, num_signals, ppc_matrix_exp, signal_labels, f'PPC Coupling (Experiment, Window {window_idx + 1})')
    ppc_matrices_exp.append(ppc_matrix_exp)
    features_exp = compute_network_features(ppc_matrix_exp)
    features_exp['id'] = f'exp_window_{window_idx + 1}'
    features_results.append(features_exp)

np.save('b_2_1_ppc_matrices_exp.npy', ppc_matrices_exp)

# 保存特征到 Excel
features_df = pd.DataFrame(features_results)
features_df = features_df.rename(columns={"id": "id"})
features_df["id"] = features_df["id"].replace({
    "rest": "b_2_1_rest",
    **{f"exp_window_{i + 1}": f"b_2_1_exp_slice{i + 1}" for i in range(len(features_df) - 1)}
})

output_path = 'b_first_network_features_ppc.xlsx'
try:
    existing_df = pd.read_excel(output_path)
    combined_df = pd.concat([existing_df, features_df], ignore_index=True)
    combined_df = combined_df[["id"] + [col for col in combined_df.columns if col != "id"]]
    combined_df.to_excel(output_path, index=False)
except FileNotFoundError:
    features_df = features_df[["id"] + [col for col in features_df.columns if col != "id"]]
    features_df.to_excel(output_path, index=False)
