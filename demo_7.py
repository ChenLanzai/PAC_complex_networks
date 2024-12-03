import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import zscore
import networkx as nx  # 用于模块度和聚集系数计算
import openpyxl  # 确保支持 Excel 文件的追加写入

# 计算PAC
def compute_pac(phase, amplitude):
    return np.abs(np.mean(amplitude * np.exp(1j * phase)))

# 初始化PAC矩阵
def initialize_pac_matrix(num_signals):
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

# 对每个窗口处理PAC
def process_window(window_signals, hilbert_cache, num_signals, pac_matrix, signal_labels, title):
    for j in range(num_signals):
        phase_j = np.angle(hilbert_cache[j])
        for k in range(num_signals):  # 考虑非对称性
            if j != k:  # 自己与自己不计算
                amplitude_k = np.abs(hilbert_cache[k])

                # 计算PAC
                pac = compute_pac(phase_j, amplitude_k)

                # 计算p值
                null_distribution_pac = compute_p_values(window_signals[j], window_signals[k])
                p_value_pac = np.mean(null_distribution_pac >= pac)

                # 应用Bonferroni校正
                alpha = 0.05
                num_comparisons = num_signals * (num_signals - 1)
                corrected_alpha = alpha / num_comparisons

                # 更新PAC矩阵
                if p_value_pac < corrected_alpha:
                    pac_matrix[j, k] = pac
                else:
                    pac_matrix[j, k] = 0  # 无边时为0

    # 绘制实时PAC矩阵
    plot_pac_matrix(pac_matrix, title, signal_labels)

# 计算p值
def compute_p_values(signal_a, signal_b, num_bootstrap=500):
    null_distribution_pac = []

    for _ in range(num_bootstrap):
        bootstrap_phase = np.angle(hilbert(np.random.choice(signal_a, len(signal_a), replace=True)))
        null_distribution_pac.append(compute_pac(bootstrap_phase, signal_b))

    return null_distribution_pac

# 绘制PAC矩阵
def plot_pac_matrix(matrix, title, signal_labels):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='Blues', interpolation='none', vmin=0, vmax=1)

    # 在矩阵图上显示值
    for i in range(len(signal_labels)):
        for j in range(len(signal_labels)):
            text = f"{matrix[i, j]:.2f}" if matrix[i, j] > 0 else "0"
            plt.text(j, i, text, ha='center', va='center', color='black', fontsize=8)

    plt.colorbar(label='PAC Strength')
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
    # 获取工作簿的所有工作表名称
    sheet_names = pd.ExcelFile(exp_file_path).sheet_names
    # 判断Sheet1和Sheet2的存在性
    sheet1_data = pd.read_excel(exp_file_path, sheet_name="Sheet1") if "Sheet1" in sheet_names else None
    sheet2_data = pd.read_excel(exp_file_path, sheet_name="Sheet2") if "Sheet2" in sheet_names else None

    # 合并数据，忽略空数据集
    df_exp = pd.concat([df for df in [sheet1_data, sheet2_data] if df is not None], ignore_index=True)
except ValueError:
    # 如果读取出错，提示错误信息
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
pac_matrix_rest = initialize_pac_matrix(num_signals)

# 对静息信号计算Hilbert变换并缓存
hilbert_cache_rest = [hilbert(signal[:T_rest]) for signal in signals_rest]
process_window(signals_rest, hilbert_cache_rest, num_signals, pac_matrix_rest, signal_labels, 'PAC Coupling (Rest)')

# 保存静息状态结果
np.save('b_2_1_pac_matrix_rest.npy', pac_matrix_rest)

# 统计静息状态特征
features_rest = compute_network_features(pac_matrix_rest)
features_rest['id'] = 'rest'
features_results.append(features_rest)

# 处理实验状态
num_windows_exp = len(signals_exp[0]) // T_exp
pac_matrices_exp = []

for window_idx in range(num_windows_exp):
    start_idx = window_idx * T_exp
    end_idx = start_idx + T_exp

    window_signals_exp = [signal[start_idx:end_idx] for signal in signals_exp]

    # 对实验信号计算Hilbert变换并缓存
    hilbert_cache_exp = [hilbert(signal) for signal in window_signals_exp]
    pac_matrix_exp = initialize_pac_matrix(num_signals)
    process_window(window_signals_exp, hilbert_cache_exp, num_signals, pac_matrix_exp, signal_labels, f'PAC Coupling (Experiment, Window {window_idx + 1})')

    # 保存每个窗口的结果
    pac_matrices_exp.append(pac_matrix_exp)

    # 统计实验状态特征
    features_exp = compute_network_features(pac_matrix_exp)
    features_exp['id'] = f'exp_window_{window_idx + 1}'
    features_results.append(features_exp)

# 保存实验状态结果
np.save('b_2_1_pac_matrices_exp.npy', pac_matrices_exp)

# 保存特征到 Excel
features_df = pd.DataFrame(features_results)
# 修改数据框格式
features_df = features_df.rename(columns={"id": "id"})  # 确保列名为 "id"
features_df["id"] = features_df["id"].replace({
    "rest": "b_2_1_rest",
    **{f"exp_window_{i + 1}": f"b_2_1_exp_slice{i + 1}" for i in range(len(features_df) - 1)}
})

# 追加写入 Excel 文件
output_path = 'b_first_network_features.xlsx'
try:
    # 如果文件已存在，读取现有内容
    existing_df = pd.read_excel(output_path)
    # 确保列 "id" 始终在第一列
    combined_df = pd.concat([existing_df, features_df], ignore_index=True)
    combined_df = combined_df[["id"] + [col for col in combined_df.columns if col != "id"]]
    combined_df.to_excel(output_path, index=False)
except FileNotFoundError:
    # 如果文件不存在，直接保存，并确保列 "id" 在第一列
    features_df = features_df[["id"] + [col for col in features_df.columns if col != "id"]]
    features_df.to_excel(output_path, index=False)

