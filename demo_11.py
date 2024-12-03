import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# 读取数据
file_path = './data/exp_b_9_3_process.xlsx'
signals = ['EEG', 'ECG', 'EDA', 'PPG', 'EMG']
df = pd.read_excel(file_path)

# 标准化信号
def standardize_signals(signals_data):
    return (signals_data - signals_data.mean()) / signals_data.std()

# 采样频率和分箱时间
fs = 1200
bin_duration = 1 * 60  # 每个分箱包含的时间长度为1分钟
bin_samples = bin_duration * fs  # 每个分箱包含的样本数

# 定义函数计算调制指数（MI）
def modulation_index(phase, amplitude, n_bins=360):
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
        return 0, 0  # 返回0和0以避免后续计算错误
    mi = np.sum(phase_bin_mean_amplitudes * np.log(phase_bin_mean_amplitudes / uniform_distribution))
    return mi, n_bins  # 返回MI值和分箱数

# 定义函数计算相位幅度耦合值（向量平均值方法）
def vector_average_coupling(phase, amplitude):
    # 计算耦合值
    coupling_value = np.abs(np.mean(amplitude * np.exp(1j * phase)))  # 更新公式
    return coupling_value

# 初始化MI矩阵
num_signals = len(signals)
num_segments = len(range(0, len(df[signals[0]]), bin_samples))  # 计算总片段数
mi_matrices = np.zeros((num_segments, num_signals, num_signals)) * np.nan  # 三维数组存储每个片段的MI矩阵
n_bins_list = []  # 存储每个片段的分箱数

# 在主循环中，初始化一个耦合矩阵
coupling_matrices = np.zeros((num_segments, num_signals, num_signals)) * np.nan  # 三维数组存储每个片段的耦合矩阵

# 处理每个信号
for segment_index in range(num_segments):
    for i, signal in enumerate(signals):
        signal_data = df[signal].values
        standardized_signal = standardize_signals(signal_data)

        # 进行希尔伯特变换
        analytic_signal = hilbert(standardized_signal)
        amplitude = np.abs(analytic_signal)
        phase = np.angle(analytic_signal)

        # 切片
        start = segment_index * bin_samples
        end = start + bin_samples
        if end > len(standardized_signal):
            break

        phase_slice = phase[start:end]
        amplitude_slice = amplitude[start:end]

        # 计算调制指数
        mi, n_bins = modulation_index(phase_slice, amplitude_slice)

        # 存储MI值到矩阵
        mi_matrices[segment_index, i, (i + 1) % num_signals] = mi  # 假设相位信号与下一个信号的振幅耦合
        n_bins_list.append(n_bins)  # 存储当前片段的分箱数

        # 计算耦合值并存储到耦合矩阵
        for j in range(num_signals):
            if i != j:  # 只计算不同信号对的耦合值
                coupling_value = vector_average_coupling(phase_slice, amplitude_slice)
                coupling_matrices[segment_index, i, j] = coupling_value  # 存储耦合值

# 打印每个信号片段的耦合矩阵及其信息
for segment_index in range(num_segments):
    print(f"Segment {segment_index + 1} Coupling Matrix:")
    print(coupling_matrices[segment_index])
    print("Signal Labels:", signals)
    print(f"Final Bins for this segment: {n_bins_list[segment_index]}")  # 显示当前片段的分箱数

