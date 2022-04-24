import numpy as np
from scipy.optimize import linear_sum_assignment

cost = np.array([[2, 0, 5],
                 [3, 2, 2]]
                )
row_ind, col_ind = linear_sum_assignment(cost)
print(row_ind)  # 开销矩阵对应的行索引
print(col_ind)  # 对应行索引的最优指派的列索引
print(cost[row_ind, col_ind])  # 提取每个行索引的最优指派列索引所在的元素，形成数组
print(cost[row_ind, col_ind].sum())  # 数组求和


