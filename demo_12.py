import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import zscore
import networkx as nx
import openpyxl  # 用于 Excel 文件操作


class PACNetworkAnalysis:
    def __init__(self, rest_file_path, exp_file_path, output_file, experiment_id, use_gpu=True):
        self.rest_file_path = rest_file_path
        self.exp_file_path = exp_file_path
        self.output_file = output_file
        self.experiment_id = experiment_id  # 如 b_2_1
        self.signal_labels = ['PPG', 'EMG', 'EEG', 'EDA', 'ECG']
        self.num_signals = len(self.signal_labels)
        self.features_results = []
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    @staticmethod
    def compute_pac(phase, amplitude):
        return torch.abs(torch.mean(amplitude * torch.exp(1j * phase)))

    @staticmethod
    def initialize_pac_matrix(num_signals, device):
        return torch.zeros((num_signals, num_signals), device=device)

    @staticmethod
    def compute_p_values(signal_a, signal_b, num_bootstrap=500):
        null_distribution_pac = []
        # 确保 signal_a 和 signal_b 是 torch.Tensor 类型
        signal_a = torch.tensor(signal_a, device=signal_a.device) if not isinstance(signal_a, torch.Tensor) else signal_a
        signal_b = torch.tensor(signal_b, device=signal_b.device) if not isinstance(signal_b, torch.Tensor) else signal_b
        
        for _ in range(num_bootstrap):
            bootstrap_indices = torch.randint(0, len(signal_a), (len(signal_a),), device=signal_a.device)
            bootstrap_signal_a = signal_a[bootstrap_indices]
            # 使用 hilbert_cuda 代替 scipy.signal.hilbert
            hilbert_result = PACNetworkAnalysis.hilbert_cuda(bootstrap_signal_a)  # 调用 GPU 版本的 Hilbert 变换
            # 计算 phase
            bootstrap_phase = torch.angle(hilbert_result)  # 确保使用 torch 的方法
            # 计算 PAC 值
            null_distribution_pac.append(PACNetworkAnalysis.compute_pac(bootstrap_phase, signal_b))
        return torch.tensor(null_distribution_pac, device=signal_a.device)


    @staticmethod
    def hilbert_cuda(signal: torch.Tensor) -> torch.Tensor:
        N = signal.size(-1)
        fft_signal = torch.fft.fft(signal)  # FFT 操作
        h = torch.zeros(N, device=signal.device, dtype=torch.complex64)

        # 创建希尔伯特滤波器
        if N % 2 == 0:  # 偶数长度
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:  # 奇数长度
            h[0] = 1
            h[1:(N + 1) // 2] = 2

        # 应用希尔伯特滤波器
        fft_signal = fft_signal * h

        # 逆 FFT
        analytic_signal = torch.fft.ifft(fft_signal)
        return analytic_signal  # 保证返回 torch.Tensor 类型

    @staticmethod
    def plot_pac_matrix(matrix, title, signal_labels):
        matrix = matrix.cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap='Blues', interpolation='none', vmin=0, vmax=1)
        for i in range(len(signal_labels)):
            for j in range(len(signal_labels)):
                text = f"{matrix[i, j]:.2f}" if matrix[i, j] > 0 else "0"
                plt.text(j, i, text, ha='center', va='center', color='black', fontsize=8)
        plt.colorbar(label='PAC Strength')
        plt.title(title)
        plt.xticks(np.arange(len(signal_labels)), signal_labels, rotation=45)
        plt.yticks(np.arange(len(signal_labels)), signal_labels)
        plt.show()

    @staticmethod
    def standardize_signals(signals, device):
        """标准化信号"""
        return [zscore(torch.tensor(signal, device=device).cpu().numpy()) for signal in signals]

    def compute_network_features(self, matrix):
        features = {}
        G = nx.from_numpy_array(matrix.cpu().numpy(), create_using=nx.Graph())

        features['density'] = nx.density(G)
        features['average_strength'] = np.mean(matrix[matrix > 0].cpu().numpy()) if torch.any(matrix > 0) else 0

        degree_centrality = torch.sum(matrix > 0, dim=1)
        features.update(
            {self.signal_labels[i] + '_degree_centrality': dc.item() for i, dc in enumerate(degree_centrality)})

        partition = nx.community.greedy_modularity_communities(G)
        modularity = nx.algorithms.community.modularity(G, partition)
        features['modularity'] = modularity

        features['global_clustering'] = nx.transitivity(G)
        return features

    def process_window(self, window_signals, hilbert_cache, pac_matrix, title):
        for j in range(self.num_signals):
            phase_j = torch.angle(hilbert_cache[j])
            for k in range(self.num_signals):
                if j != k:
                    amplitude_k = torch.abs(hilbert_cache[k])
                    pac = self.compute_pac(phase_j, amplitude_k)

                    # 确保窗口信号传入计算函数时为 torch.Tensor 类型
                    null_distribution_pac = self.compute_p_values(torch.tensor(window_signals[j], device=self.device),
                                                                  torch.tensor(window_signals[k], device=self.device))
                    p_value_pac = torch.mean((null_distribution_pac.cuda() >= pac).float())
                    alpha = 0.05
                    num_comparisons = self.num_signals * (self.num_signals - 1)
                    corrected_alpha = alpha / num_comparisons

                    pac_matrix[j, k] = pac if p_value_pac < corrected_alpha else 0
        # self.plot_pac_matrix(pac_matrix, title, self.signal_labels)

    def analyze(self):
        # 读取静息状态数据
        df_rest = pd.read_excel(self.rest_file_path)
        # 读取实验状态数据，处理Sheet1和Sheet2
        try:
            # 获取工作簿的所有工作表名称
            sheet_names = pd.ExcelFile(self.exp_file_path).sheet_names
            # 判断Sheet1和Sheet2的存在性
            sheet1_data = pd.read_excel(self.exp_file_path, sheet_name="Sheet1") if "Sheet1" in sheet_names else None
            sheet2_data = pd.read_excel(self.exp_file_path, sheet_name="Sheet2") if "Sheet2" in sheet_names else None

            # 合并数据，忽略空数据集
            df_exp = pd.concat([df for df in [sheet1_data, sheet2_data] if df is not None], ignore_index=True)
        except ValueError:
            # 如果读取出错，提示错误信息
            raise ValueError("实验状态文件格式有误，请检查是否包含 Sheet1 和 Sheet2 工作表！")

        signals_rest = self.standardize_signals([df_rest[col].values for col in self.signal_labels], self.device)
        signals_exp = self.standardize_signals([df_exp[col].values for col in self.signal_labels], self.device)

        T_rest = len(signals_rest[0])
        T_exp = 72000

        # 处理静息状态
        pac_matrix_rest = self.initialize_pac_matrix(self.num_signals, self.device)
        hilbert_cache_rest = [self.hilbert_cuda(torch.tensor(signal[:T_rest], device=self.device)) for signal in signals_rest]
        self.process_window(signals_rest, hilbert_cache_rest, pac_matrix_rest, 'PAC Coupling (Rest)')
        torch.save(pac_matrix_rest, f'./save_networks_pt/{self.experiment_id}_pac_matrix_rest.pt')

        features_rest = self.compute_network_features(pac_matrix_rest)
        features_rest['id'] = 'rest'
        self.features_results.append(features_rest)

        # 处理实验状态
        num_windows_exp = len(signals_exp[0]) // T_exp
        pac_matrices_exp = [self.initialize_pac_matrix(self.num_signals, self.device) for _ in
                            range(num_windows_exp)]  # 在循环外部初始化PAC矩阵列表

        for window_idx in range(num_windows_exp):
            start_idx = window_idx * T_exp
            end_idx = start_idx + T_exp
            window_signals_exp = [signal[start_idx:end_idx] for signal in signals_exp]

            hilbert_cache_exp = [self.hilbert_cuda(torch.tensor(signal, device=self.device)) for signal in window_signals_exp]
            self.process_window(window_signals_exp, hilbert_cache_exp, pac_matrices_exp[window_idx],
                                f'PAC Coupling (Experiment, Window {window_idx + 1})')  # 使用预先初始化的PAC矩阵

            features_exp = self.compute_network_features(pac_matrices_exp[window_idx])
            features_exp['id'] = f'exp_window_{window_idx + 1}'
            self.features_results.append(features_exp)

        torch.save(pac_matrices_exp, f'./save_networks_pt/{self.experiment_id}_pac_matrices_exp.pt')

        # 保存特征结果到 Excel
        features_df = pd.DataFrame(self.features_results)
        features_df.insert(0, 'id', features_df.pop('id'))
        features_df['id'] = features_df['id'].replace({
            "rest": f"{self.experiment_id}_rest",
            **{f"exp_window_{i + 1}": f"{self.experiment_id}_exp_slice{i + 1}" for i in range(len(features_df) - 1)}
        })
        try:
            existing_df = pd.read_excel(self.output_file)
            combined_df = pd.concat([existing_df, features_df], ignore_index=True)
            combined_df.to_excel(self.output_file, index=False)
        except FileNotFoundError:
            features_df.to_excel(self.output_file, index=False)


# 调用类
rest_file = './data/rest_b_2_1_process.xlsx'
exp_file = './data/exp_b_2_1_process.xlsx'
output_file = 'b_first_network_features_chxs.xlsx'
experiment_id = 'b_2_1'

analysis = PACNetworkAnalysis(rest_file, exp_file, output_file, experiment_id, use_gpu=True)
analysis.analyze()
