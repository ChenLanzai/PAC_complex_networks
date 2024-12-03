import numpy as np
import torch

# 读取 .npy 文件
file_path = './save_networks/b_4_1_pac_matrices_exp.npy'
pac_matrices_exp = np.load(file_path, allow_pickle=True)
#
# 打印矩阵信息
print("读取的 PAC 矩阵数量:", len(pac_matrices_exp))
for idx, matrix in enumerate(pac_matrices_exp):
    print(f"\nPAC 矩阵 {idx + 1}:")
    print("形状:", matrix.shape)
    print(matrix)


# # 读取 .pt 文件  ./save_networks_all/
# file_path_pt = 'b_2_1_pac_matrices_exp.pt'
# pac_matrices_exp_pt = torch.load(file_path_pt)
# # 打印矩阵信息
# print("读取的 PAC 矩阵数量 (PyTorch):", len(pac_matrices_exp_pt))
# for idx, matrix in enumerate(pac_matrices_exp_pt):
#     print(f"\nPAC 矩阵 {idx + 1} (PyTorch):")
#     print("形状:", matrix.shape)
#     print(matrix.cpu().numpy())  # 转换为 NumPy 数组以便打印
