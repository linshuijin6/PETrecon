import pandas as pd
import numpy as np
from itertools import product


def process_excel_data(path, path1):
    # 读取原始Excel文件
    df = pd.read_excel(path, header=None)

    # 初始化存储结果数据
    results = []

    # 分组处理数据
    i = 0
    while i < len(df):
        # 提取6行数据
        group = df.iloc[i:i + 6, :2].values

        # 检查是否是空行，若空行则跳过
        if np.isnan(group).all():
            i += 1
            continue

        # 遍历所有组合，找到方差最小的组合
        min_variance = float('inf')
        best_combination = None

        for combination in product([0, 1], repeat=6):
            # 根据组合选择数据
            selected_values = [group[row][col] for row, col in enumerate(combination)]

            # 计算方差
            variance = np.var(selected_values)

            # 选择方差最小的组合
            if variance < min_variance:
                min_variance = variance
                best_combination = selected_values

        tm1 = np.var(group[:, 0])
        tm2 = np.var(group[:, 1])
        tm3 = np.var(best_combination)

        # 保存最小方差组合
        results.append({
            "combination": best_combination,
            "variances": [tm1, tm2, tm3]
        })

        # 跳到下一个组
        i += 7  # 每组6行+1行空行

    # 构建新数据框，包含原始数据及处理后的结果
    new_df = df.copy()

    # 将每组方差最小组合添加到新列
    combination_column = []
    variance_columns = [None] * len(df)  # 用于存储方差

    # 遍历每组，将结果合并
    for idx, result in enumerate(results):
        start_idx = idx * 7

        # 添加方差最小组合的6行数据
        combination_column.extend(result["combination"] + [None])

        # 设置每列方差到新表格的第四列的前三行
        if start_idx + 3 < len(df):
            for j, var in enumerate(result["variances"]):
                variance_columns[start_idx + j] = var

    # 将方差最小组合添加为新列
    new_df["最小方差组合"] = combination_column[:-1]

    # 将每列的方差添加为第四列
    new_df["方差"] = variance_columns

    # 保存结果到新的Excel文件
    new_df.to_excel(path1, index=False, header=False)


# 使用示例
path = r"D:\2Z.xlsx"
path1 = r"D:\2Z1.xlsx"
process_excel_data(path, path1)
