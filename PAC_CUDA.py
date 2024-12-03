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

        # 初始化 PAC 矩阵
        self.pac_matrix_rest = self.initialize_pac_matrix(self.num_signals, self.device)
        self.pac_matrices_exp = []  # 用于存储实验状态的 PAC 矩阵

    @staticmethod
    def compute_pac(phase, amplitude):
        return torch.abs(torch.mean(amplitude * torch.exp(1j * phase)))

    @staticmethod
    def initialize_pac_matrix(num_signals, device):
        return torch.zeros((num_signals, num_signals), device=device)

    @staticmethod
    def compute_p_values(signal_a, signal_b, num_bootstrap=100):
        null_distribution_pac = []
        for _ in range(num_bootstrap):
            bootstrap_indices = torch.randint(0, len(signal_a), (len(signal_a),), device=signal_a[0].device)
            bootstrap_signal_a = signal_a[bootstrap_indices]
            bootstrap_phase = torch.angle(
                torch.tensor(hilbert(bootstrap_signal_a.cpu().numpy()), device=signal_a.device))
            null_distribution_pac.append(PACNetworkAnalysis.compute_pac(bootstrap_phase, signal_b))
        return torch.tensor(null_distribution_pac, device=signal_a.device)

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
        signals_tensor = [torch.tensor(signal, device=device) for signal in signals]
        return [(signal - signal.mean()) / signal.std() for signal in signals_tensor]

    def compute_network_features(self, matrix):
        features = {}
        G = nx.from_numpy_array(matrix.cpu().numpy(), create_using=nx.Graph())

        if G.number_of_edges() == 0:
            features['density'] = 0
            features['average_strength'] = 0
            features['modularity'] = 0
            features['global_clustering'] = 0
            return features

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

    def process_rest_window(self, window_signals, hilbert_cache, pac_matrix, title):
        for j in range(self.num_signals):
            phase_j = torch.angle(hilbert_cache[j])
            for k in range(self.num_signals):
                if j != k:
                    amplitude_k = torch.abs(hilbert_cache[k]) #float64
                    pac = self.compute_pac(phase_j, amplitude_k)

                    null_distribution_pac = self.compute_p_values(window_signals[j], window_signals[k]) #送72000也就是整个片段的j信号和k信号去完成虚假分布
                    p_value_pac = torch.mean((null_distribution_pac >= pac).to(dtype=torch.float32))

                    alpha = 0.05
                    num_comparisons = self.num_signals * (self.num_signals - 1)
                    corrected_alpha = alpha / num_comparisons

                    pac_matrix[j, k] = pac if p_value_pac < corrected_alpha else 0
        # self.plot_pac_matrix(pac_matrix, title, self.signal_labels)

    def process_exp_window(self, window_signals, hilbert_cache, pac_matrix, title, window_index):
        pac_values = []  # 用于存储每个片段的 PAC 值、信号对索引和片段号
        for j in range(self.num_signals):
            phase_j = torch.angle(hilbert_cache[j]).to(self.device)  # 将数据移到设备上
            for k in range(self.num_signals):
                if j != k:
                    amplitude_k = torch.abs(hilbert_cache[k]).to(self.device)  # 将数据移到设备上
                    pac = self.compute_pac(phase_j, amplitude_k)

                    pac_values.append({
                        'window_index': window_index,  # 存储片段号
                        'phase_signal': j,
                        'amplitude_signal': k,
                        'pac_value': pac,
                        'phase_signal_values': window_signals[j].to(self.device),  # 存储相位信号值，而不是希尔伯特变换之后的值。注意
                        'amplitude_signal_values': window_signals[k].to(self.device)  # 存储幅值信号值，而不是希尔伯特变换之后的值。注意
                    })  # 存储信号对索引、PAC值及原始信号
        return pac_values  # 返回 PAC 值列表

    # def determine_edge_existence(self, all_pac_values, signals_exp):
    #     """判断边的存在性，通过比较实际PAC值与虚假分布"""
    #     alpha = 0.05  # 显著性水平
    #     num_comparisons = self.num_signals * (self.num_signals - 1)  # 比较的信号对数
    #     corrected_alpha = alpha / num_comparisons  # Bonferroni校正
    #
    #     # 计算信号数据长度和随机抽取的样本数
    #     num_samples = len(signals_exp[0])  # 每个信号序列的长度
    #     num_selected = int(0.25 * num_samples)  # 选取的样本数量
    #     random_indices = torch.randint(0, num_samples, (num_selected,), device=signals_exp[0].device)  # 随机生成索引
    #
    #     # 缓存虚假分布的PAC值
    #     null_distribution_cache = {}
    #
    #     # 遍历每个信号对的所有窗口
    #     for pac_values in all_pac_values:
    #         for pac_value_info in pac_values:
    #             window_index = pac_value_info['window_index']
    #             phase_signal = pac_value_info['phase_signal']
    #             amplitude_signal = pac_value_info['amplitude_signal']
    #             pac_value = pac_value_info['pac_value']  # float64
    #
    #             # 构造信号对 (phase_signal, amplitude_signal) 的键
    #             signal_pair_key = (phase_signal, amplitude_signal)
    #
    #             # 如果该信号对的虚假分布已经计算过，直接从缓存中获取
    #             if signal_pair_key not in null_distribution_cache:
    #                 # 从信号中选择 75% 的数据
    #                 phase_signal_data = torch.stack(
    #                     [signal[random_indices] for signal in signals_exp], dim=0
    #                 )
    #                 amplitude_signal_data = torch.stack(
    #                     [signal[random_indices] for signal in signals_exp], dim=0
    #                 )
    #                 # 计算虚假分布的 PAC 值
    #                 null_distribution_pac = self.compute_p_values(phase_signal_data, amplitude_signal_data)
    #                 # 缓存该信号对的虚假分布
    #                 null_distribution_cache[signal_pair_key] = null_distribution_pac
    #             else:
    #                 # 从缓存中获取虚假分布
    #                 null_distribution_pac = null_distribution_cache[signal_pair_key]
    #             # 计算 p 值
    #             p_value = torch.mean((null_distribution_pac >= pac_value).to(dtype=torch.float32))
    #
    #             # 判断是否存在显著耦合关系
    #             if p_value < corrected_alpha:
    #                 # 如果存在显著耦合关系，设置该位置为 PAC 值
    #                 self.pac_matrices_exp[window_index][phase_signal, amplitude_signal] = pac_value
    #             else:
    #                 # 否则，置为 0
    #                 self.pac_matrices_exp[window_index][phase_signal, amplitude_signal] = 0
    # 注释掉的是校正方法不一样而已
    def benjamini_hochberg_fdr(self, p_values, alpha=0.05):
        """Benjamini-Hochberg FDR 校正"""
        # 对p值进行排序
        sorted_p_values, sorted_indices = torch.sort(p_values)
        m = len(sorted_p_values)
        thresholds = (torch.arange(1, m + 1, dtype=torch.float32) / m) * alpha  # FDR 校正的阈值
        # 确定哪些 p 值显著
        reject = sorted_p_values <= thresholds
        # 将结果恢复到原来的顺序
        reject_order = torch.zeros_like(reject, dtype=torch.bool)
        reject_order[sorted_indices] = reject
        return reject_order

    def determine_edge_existence(self, all_pac_values, signals_exp):
        """判断边的存在性，通过比较实际PAC值与虚假分布"""
        alpha = 0.05  # 显著性水平
        num_comparisons = self.num_signals * (self.num_signals - 1)  # 比较的信号对数
        # 不再进行 Bonferroni 校正，使用 FDR 校正

        # 计算信号数据长度和随机抽取的样本数
        num_samples = len(signals_exp[0])  # 每个信号序列的长度
        num_selected = int(0.25 * num_samples)  # 选取的样本数量
        random_indices = torch.randint(0, num_samples, (num_selected,), device=signals_exp[0].device)  # 随机生成索引

        # 缓存虚假分布的PAC值
        null_distribution_cache = {}

        # 存储所有的 p 值
        p_values_all = []

        # 遍历每个信号对的所有窗口
        for pac_values in all_pac_values:
            for pac_value_info in pac_values:
                window_index = pac_value_info['window_index']
                phase_signal = pac_value_info['phase_signal']
                amplitude_signal = pac_value_info['amplitude_signal']
                pac_value = pac_value_info['pac_value']  # float64

                # 构造信号对 (phase_signal, amplitude_signal) 的键
                signal_pair_key = (phase_signal, amplitude_signal)

                # 如果该信号对的虚假分布已经计算过，直接从缓存中获取
                if signal_pair_key not in null_distribution_cache:
                    # 从信号中选择 25% 的数据
                    phase_signal_data = torch.stack(
                        [signal[random_indices] for signal in signals_exp], dim=0
                    )
                    amplitude_signal_data = torch.stack(
                        [signal[random_indices] for signal in signals_exp], dim=0
                    )
                    # 计算虚假分布的 PAC 值
                    null_distribution_pac = self.compute_p_values(phase_signal_data, amplitude_signal_data)
                    # 缓存该信号对的虚假分布
                    null_distribution_cache[signal_pair_key] = null_distribution_pac
                else:
                    # 从缓存中获取虚假分布
                    null_distribution_pac = null_distribution_cache[signal_pair_key]

                # 计算 p 值
                p_value = torch.mean((null_distribution_pac >= pac_value).to(dtype=torch.float32))
                p_values_all.append(p_value)

        # 将所有 p 值转换为 tensor
        p_values_tensor = torch.tensor(p_values_all)

        # 进行 Benjamini-Hochberg FDR 校正
        reject_order = self.benjamini_hochberg_fdr(p_values_tensor, alpha=alpha)

        # 遍历每个信号对，判断是否显著
        p_values_idx = 0
        for pac_values in all_pac_values:
            for pac_value_info in pac_values:
                window_index = pac_value_info['window_index']
                phase_signal = pac_value_info['phase_signal']
                amplitude_signal = pac_value_info['amplitude_signal']
                pac_value = pac_value_info['pac_value']  # float64

                # 使用 Benjamini-Hochberg 校正后的 p 值决定是否显著
                if reject_order[p_values_idx]:
                    # 如果显著，设置 PAC 值
                    self.pac_matrices_exp[window_index][phase_signal, amplitude_signal] = pac_value
                else:
                    # 否则，置为 0
                    self.pac_matrices_exp[window_index][phase_signal, amplitude_signal] = 0

                p_values_idx += 1

    # 在 GPU 上实现希尔伯特变换的函数
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
        return analytic_signal

    def load_data(self):
        """读取数据"""
        # 读取静息状态数据
        df_rest = pd.read_excel(self.rest_file_path)
        # 读取实验状态数据，处理Sheet1和Sheet2
        try:
            sheet_names = pd.ExcelFile(self.exp_file_path).sheet_names
            sheet1_data = pd.read_excel(self.exp_file_path, sheet_name="Sheet1") if "Sheet1" in sheet_names else None
            sheet2_data = pd.read_excel(self.exp_file_path, sheet_name="Sheet2") if "Sheet2" in sheet_names else None

            df_exp = pd.concat([df for df in [sheet1_data, sheet2_data] if df is not None], ignore_index=True)
        except ValueError:
            raise ValueError("实验状态文件格式有误，请检查是否包含 Sheet1 和 Sheet2 工作表！")
        return df_rest, df_exp

    def process_rest_data(self, df_rest):
        """处理静息状态数据"""
        signals_rest = self.standardize_signals([df_rest[col].values for col in self.signal_labels], self.device)#float64
        T_rest = len(signals_rest[0])

        hilbert_cache_rest = [
            self.hilbert_cuda(signal[:T_rest]) for signal in signals_rest
        ]  # 直接在 GPU 上计算希尔伯特变换
        self.process_rest_window(
            signals_rest, hilbert_cache_rest, self.pac_matrix_rest, 'PAC Coupling (Rest)'
        )
        torch.save(self.pac_matrix_rest, f'./save_networks_all/{self.experiment_id}_pac_matrix_rest.pt')

        features_rest = self.compute_network_features(self.pac_matrix_rest)
        features_rest['id'] = 'rest'
        self.features_results.append(features_rest)

    def process_exp_data(self, df_exp):
        """处理实验状态数据"""
        signals_exp = self.standardize_signals([df_exp[col].values for col in self.signal_labels], self.device)

        T_exp = 72000
        num_windows_exp = len(signals_exp[0]) // T_exp
        self.pac_matrices_exp = [self.initialize_pac_matrix(self.num_signals, self.device) for _ in
                                 range(num_windows_exp)]

        all_pac_values = []  # 用于存储所有窗口的 PAC 值

        for window_idx in range(num_windows_exp):
            start_idx = window_idx * T_exp
            end_idx = start_idx + T_exp
            window_signals_exp = [signal[start_idx:end_idx] for signal in signals_exp]

            hilbert_cache_exp = [self.hilbert_cuda(signal) for signal in window_signals_exp]

            pac_values = self.process_exp_window(window_signals_exp, hilbert_cache_exp,
                                                 self.pac_matrices_exp[window_idx],
                                                 f'PAC Coupling (Experiment, Window {window_idx + 1})', window_idx)
            all_pac_values.append(pac_values)  # 将每个窗口的 PAC 值添加到列表中
            
        self.determine_edge_existence(all_pac_values, signals_exp)  # 调用新函数判断边的存在性  这里的列表中的值是float64
        # 保存实验状态下的 PAC 矩阵
        torch.save(self.pac_matrices_exp, f'./save_networks_all/{self.experiment_id}_pac_matrices_exp.pt')
        # 然后计算网络特征
        for window_idx in range(num_windows_exp):
            features_exp = self.compute_network_features(self.pac_matrices_exp[window_idx])
            features_exp['id'] = f'exp_window_{window_idx + 1}'
            self.features_results.append(features_exp)  # 将特征结果添加到列表
        # 在此处可以对 all_pac_values 进行进一步处理或存储

    def save_results(self):
        """保存结果"""
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

    def analyze(self):
        df_rest, df_exp = self.load_data()  # 加载数据
        self.process_rest_data(df_rest)  # 处理静息状态数据
        self.process_exp_data(df_exp)  # 处理实验状态数据
        self.save_results()  # 保存结果


# 调用类
rest_file = './data/rest_b_2_3_process.xlsx'
exp_file = './data/exp_b_2_3_process.xlsx'
output_file = 'b_third_network_features.xlsx'
experiment_id = 'b_2_3'

analysis = PACNetworkAnalysis(rest_file, exp_file, output_file, experiment_id, use_gpu=True)
analysis.analyze()
