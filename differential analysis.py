import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def remove_outliers(data, multiplier=1.5):
    """
    使用四分位数法 (IQR) 剔除离群值。

    参数：
    data (array-like): 输入数据
    multiplier (float): IQR 的倍数，默认是 1.5

    返回：
    np.ndarray: 剔除离群值后的数据
    """
    q1 = np.percentile(data, 25)  # 第 1 四分位数
    q3 = np.percentile(data, 75)  # 第 3 四分位数
    iqr = q3 - q1  # 四分位距
    lower_bound = q1 - multiplier * iqr  # 下界
    upper_bound = q3 + multiplier * iqr  # 上界
    return data[(data >= lower_bound) & (data <= upper_bound)]


# 数据读取
df_first = pd.read_excel('b_first_network_features.xlsx')
df_third = pd.read_excel('b_third_network_features.xlsx')

# 筛选实验状态数据
first_exp_features = df_first[df_first['id'].str.contains('b_.*exp_slice')]
third_exp_features = df_third[df_third['id'].str.contains('b_.*exp_slice')]

# 需要比较的特征
features_to_compare = ['density', 'average_strength', 'PPG_degree_centrality',
                       'EMG_degree_centrality', 'EEG_degree_centrality',
                       'EDA_degree_centrality', 'ECG_degree_centrality',
                       'modularity', 'global_clustering']

results = {}

for feature in features_to_compare:
    first_values = first_exp_features[feature].values
    third_values = third_exp_features[feature].values

    # 移除离群点
    first_values = remove_outliers(first_values)
    third_values = remove_outliers(third_values)

    # 1. Welch's t 检验 - 用于比较两组数据的均值差异
    t_stat, p_value = stats.ttest_ind(first_values, third_values, equal_var=False)
    results[feature] = {'t_statistic': t_stat, 'p_value': p_value}

    # 2. Mann-Whitney U 检验 - 非参数检验，不要求数据正态分布
    u_stat, p_value_nonparam = mannwhitneyu(first_values, third_values, alternative='two-sided')
    results[feature]['p_value_nonparam'] = p_value_nonparam

    # 3. Cohen's d 效应量 - 评估差异的实际显著性大小
    pooled_std = np.sqrt(((len(first_values) - 1) * first_values.std() ** 2 +
                          (len(third_values) - 1) * third_values.std() ** 2) /
                         (len(first_values) + len(third_values) - 2))
    cohens_d = (first_values.mean() - third_values.mean()) / pooled_std
    results[feature]['cohens_d'] = cohens_d

    # 动态调整直方图和箱型图
    # （绘图代码略）
    
    # 新增：绘制箱型图对比实验一和实验三的特征
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[first_values, third_values], palette="Set2")
    plt.xticks([0, 1], ['exp1', 'exp3'])
    plt.title(f'{feature} VS')
    plt.ylabel(feature)
    plt.show()  # 显示箱型图而不是保存

# 多重检验校正
p_values = [results[feature]['p_value'] for feature in features_to_compare]
_, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

# 输出校正结果
for i, feature in enumerate(features_to_compare):
    results[feature]['p_corrected'] = p_corrected[i]
    
    # 添加显著性判断
    results[feature]['significant'] = {
        't_test': results[feature]['p_value'] < 0.05,
        'mann_whitney': results[feature]['p_value_nonparam'] < 0.05,
        'cohens_d': abs(results[feature]['cohens_d']) > 0.2  # 0.2 为小效应，0.5 为中效应，0.8 为大效应
    }

# 输出结果 DataFrame
results_df = pd.DataFrame(results).T

# 新增显著性列
results_df['significant'] = results_df.apply(lambda row: '显著' if row['significant']['t_test'] or row['significant']['mann_whitney'] else '不显著', axis=1)

# 优化输出格式
results_df['effect_size'] = results_df['cohens_d'].apply(lambda x: '小效应' if abs(x) <= 0.2 else ('中效应' if abs(x) <= 0.5 else '大效应'))
results_df = results_df[['significant', 't_statistic', 'p_value', 'p_value_nonparam', 'cohens_d', 'effect_size']]

# 新增：将结果导出到 output.xlsx 文件
# results_df.to_excel('output2.xlsx', index=True)  # index=True 以保留索引
print(results_df)

