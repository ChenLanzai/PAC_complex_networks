import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats

# 读取数据并计算特征
def process_experiment_files(file_list, window_size):
    all_modularities = []
    all_avg_path_lengths = []
    all_efficiencies = []

    for file in file_list:
        data = np.load(file)
        modularities, avg_path_lengths, efficiencies = calculate_modularity_and_efficiency(data, window_size)
        
        all_modularities.append(modularities)
        all_avg_path_lengths.append(avg_path_lengths)
        all_efficiencies.append(efficiencies)

    return all_modularities, all_avg_path_lengths, all_efficiencies


# 假设 num_time_slices 是从数据中获取的时间切片个数
# 这里我们假设数据的形状为 (num_time_slices, num_nodes, num_nodes)
# 你需要根据实际数据来设置这个值
def get_num_time_slices(file):
    data = np.load(file)
    return data.shape[2]  # 返回网络的个数


def calculate_modularity_and_efficiency(data, window_size):
    modularities = []
    avg_path_lengths = []
    efficiencies = []

    for start in range(num_time_slices - window_size + 1):
        # 创建滑动窗口
        window = data[start:start + window_size]

        # 计算窗口内的平均连接矩阵
        avg_matrix = np.mean(window, axis=0)

        # 创建图
        G = nx.from_numpy_array(avg_matrix)

        # 计算模块度
        partition = nx.community.greedy_modularity_communities(G)
        modularity = nx.community.modularity(G, partition)
        modularities.append(modularity)

        # 计算平均路径长度
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
        else:
            avg_path_length = float('inf')  # 如果图不连通，返回无穷大
        avg_path_lengths.append(avg_path_length)

        # 计算全局效率
        efficiency = nx.global_efficiency(G)
        efficiencies.append(efficiency)

    return modularities, avg_path_lengths, efficiencies


# 实验一文件列表
experiment_1_files = [f'b_{i}_1_pac_matrices_exp.npy' for i in range(1, 11)]
# 假设我们从实验一的第一个文件中获取时间切片个数
num_time_slices = get_num_time_slices(experiment_1_files[0])  # 从第一个实验文件动态获取

# 设置滑动窗口大小
window_size = num_time_slices  # 自动设置为读取到的时间切片个数

# 实验三文件列表
experiment_3_files = [f'b_{i}_3_pac_matrices_exp.npy' for i in range(1, 11)]

# 处理实验一和实验三的数据
modularities_exp1, avg_path_lengths_exp1, efficiencies_exp1 = process_experiment_files(experiment_1_files, window_size)
modularities_exp3, avg_path_lengths_exp3, efficiencies_exp3 = process_experiment_files(experiment_3_files, window_size)

# 统计分析
def statistical_analysis(exp1_data, exp3_data):
    results = {}
    for feature_name, exp1_values, exp3_values in zip(['Modularity', 'Avg Path Length', 'Efficiency'],
                                                       [modularities_exp1, avg_path_lengths_exp1, efficiencies_exp1],
                                                       [modularities_exp3, avg_path_lengths_exp3, efficiencies_exp3]):
        # 进行t检验
        t_stat, p_value = stats.ttest_ind(exp1_values, exp3_values, equal_var=False)
        results[feature_name] = {'t_stat': t_stat, 'p_value': p_value}
    return results

# 进行统计分析
analysis_results = statistical_analysis(modularities_exp1, modularities_exp3)

# 输出结果
print("实验一与实验三的统计分析结果:")
for feature, result in analysis_results.items():
    print(f"{feature}: t_stat = {result['t_stat']}, p_value = {result['p_value']}")

