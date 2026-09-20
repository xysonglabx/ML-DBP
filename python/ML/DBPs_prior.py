import pandas as pd
import numpy as np

# ==================== 读取数据 ====================
file_path = "4nopriorx.xlsx"  # 请确保文件路径正确
df = pd.read_excel(file_path, sheet_name="4noprior")

# 选取需要的列（添加 Abbreviation 列）
df_raw = df[["pdb_name", "DBPs", "Abbreviation", "SMILES", "pLC50", "BCF", "binding_energy", "Persistence"]].copy()

# ==================== 数据清洗 ====================
print("缺失值统计：")
print(df_raw.isnull().sum())

df_clean = df_raw.dropna().reset_index(drop=True)

# ==================== 构建正向指标矩阵 ====================
df_clean["neg_binding_energy"] = -df_clean["binding_energy"]  # 负结合能越大越好


def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())


indicators = ["pLC50", "BCF", "neg_binding_energy", "Persistence"]
df_norm = df_clean.copy()
for col in indicators:
    df_norm[col + "_norm"] = min_max_normalize(df_clean[col])

# 标准化矩阵 X (n_samples x 4)
X = df_norm[[col + "_norm" for col in indicators]].values


# ==================== Modified CRITIC 权重计算 ====================
def modified_critic_weight(X, importance_coeffs):
    """
    X: 标准化矩阵 (样本 x 指标), 所有指标已正向化为 [0,1]
    importance_coeffs: list, 每个指标的重要性系数 (λ)
    返回: 调整后的 CRITIC 权重向量
    """
    n, m = X.shape

    # Step 1: 计算每个指标的标准差 (对比强度)
    stds = np.std(X, axis=0, ddof=1)  # 样本标准差

    # Step 2: 计算相关系数矩阵 (Pearson)
    corr = np.corrcoef(X.T)  # shape (m, m)

    # Step 3: 计算冲突强度 R_j = sum_{k=1}^{m} (1 - r_jk)
    conflict = np.zeros(m)
    for j in range(m):
        conflict[j] = np.sum(1 - corr[j, :])

    # Step 4: 信息含量 C_j = std_j * R_j
    C = stds * conflict

    # Step 5: 引入重要性系数，计算调整后的权重
    lambda_C = importance_coeffs * C
    w = lambda_C / np.sum(lambda_C)
    return w


# 设置重要性系数: pLC50=2.5, 其他=1
importance_coeffs = np.array([2.5, 1.0, 1.0, 1.0])
weights = modified_critic_weight(X, importance_coeffs)

print("\nModified CRITIC 权重 (pLC50 重要性系数=2.5):")
for col, w in zip(indicators, weights):
    print(f"{col:20s}: {w:.4f}")

# ==================== 计算综合得分 ====================
score = np.dot(X, weights)
df_norm["BSEV_like"] = score
df_norm["BSEV_like_scaled"] = min_max_normalize(df_norm["BSEV_like"])

# ==================== 结果输出 ====================
result_cols = ["pdb_name", "DBPs", "Abbreviation", "SMILES", "pLC50", "BCF", "binding_energy", "Persistence",
               "neg_binding_energy", "pLC50_norm", "BCF_norm", "neg_binding_energy_norm", "Persistence_norm",
               "BSEV_like", "BSEV_like_scaled"]
result_df = df_norm[result_cols]
result_df = result_df.sort_values("BSEV_like", ascending=False).reset_index(drop=True)

output_path = "BSEV_result_ModifiedCRITIC_pLC50x2.xlsx"
result_df.to_excel(output_path, index=False)
print(f"\n计算完成！结果已保存至 {output_path}")

print("\n前10个高风险化学品 (Modified CRITIC):")
print(result_df[["pdb_name", "Abbreviation", "BSEV_like_scaled"]].head(10))