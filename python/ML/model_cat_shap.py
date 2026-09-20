import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor as _RFR
from sklearn.preprocessing import StandardScaler as _SS
import lightgbm as lgb
from catboost import CatBoostRegressor
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

print("=" * 80)
print("SHAP分析 - 从已保存的模型加载 (CAT数据集)")
print("=" * 80)

# ==========================================
# 配置参数
# ==========================================
OUTPUT_DIR = r"F:\python-workspace\Our\YanEr\DBPs_Script\beiyesi\ml_model_CATmodelx\output"            # 输出目录
MODELS_DIR = r"F:\python-workspace\Our\YanEr\DBPs_Script\beiyesi\ml_model_CATmodelx\saved_models"   # 模型目录
TOP_N_FEATURES = 15                         # Top特征数量

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n✓ 输出目录: {OUTPUT_DIR}")
print(f"✓ 模型目录: {MODELS_DIR}")

# 图形参数
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

# 配色方案
COLOR_SCHEMES = {
    'viridis': plt.cm.viridis,
    'plasma': plt.cm.plasma,
    'coolwarm': plt.cm.coolwarm,
    'RdYlBu': plt.cm.RdYlBu,
    'RdBu_r': plt.cm.RdBu_r
}

# ==========================================
# 1. 特征提取函数 (与训练时一致)
# ==========================================
def extract_enhanced_features(smiles_list):
    fps = []
    phys_features = []
    valid_indices = []
    valid_smiles = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            fp_2048 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_1024_r3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)

            desc_list = [
                Descriptors.MolWt(mol),
                Descriptors.HeavyAtomMolWt(mol),
                Descriptors.ExactMolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.MolMR(mol),
                Descriptors.TPSA(mol),
                Descriptors.LabuteASA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.RingCount(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.MaxAbsPartialCharge(mol) if Descriptors.MaxAbsPartialCharge(mol) else 0,
                Descriptors.MinAbsPartialCharge(mol) if Descriptors.MinAbsPartialCharge(mol) else 0,
                Descriptors.MaxPartialCharge(mol) if Descriptors.MaxPartialCharge(mol) else 0,
                Descriptors.MinPartialCharge(mol) if Descriptors.MinPartialCharge(mol) else 0,
                smiles.count('F'),
                smiles.count('C(F)(F)'),
                smiles.count('C(F)(F)F'),
                Descriptors.NOCount(mol),
                Descriptors.NHOHCount(mol),
                Descriptors.NumValenceElectrons(mol),
                Descriptors.NumRadicalElectrons(mol),
                Descriptors.BalabanJ(mol),
                Descriptors.BertzCT(mol),
                Descriptors.Chi0(mol),
                Descriptors.Chi0n(mol),
                Descriptors.Chi0v(mol),
                Descriptors.Chi1(mol),
                Descriptors.Chi1n(mol),
                Descriptors.Chi1v(mol),
                Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),
                Descriptors.Kappa3(mol),
                Descriptors.HallKierAlpha(mol),
                Descriptors.Ipc(mol),
            ]
            desc_list = [0 if (x is None or np.isnan(x) or np.isinf(x)) else x for x in desc_list]

            fps.append(np.concatenate([np.array(fp_2048), np.array(fp_1024_r3)]))
            phys_features.append(desc_list)
            valid_indices.append(i)
            valid_smiles.append(smiles)

        except Exception:
            continue

    X_fp = np.array(fps)
    X_phys = np.array(phys_features)
    X_combined = np.hstack([X_fp, X_phys])
    return X_combined, valid_indices, valid_smiles, X_phys.shape[1]

# ==========================================
# 2. 共线性筛选工具 (与训练时一致)
# ==========================================
def remove_low_variance_features(X_phys, feature_names, threshold=0.01):
    from sklearn.preprocessing import StandardScaler
    scaler_tmp = StandardScaler()
    X_std = scaler_tmp.fit_transform(X_phys)
    variances = np.var(X_std, axis=0)
    mask = variances > threshold
    removed = [feature_names[i] for i in range(len(feature_names)) if not mask[i]]
    if removed:
        print(f"  [低方差过滤] 移除 {len(removed)} 个低方差特征: {removed}")
    else:
        print(f"  [低方差过滤] 无特征被移除 (阈值={threshold})")
    return mask, removed

def remove_correlated_features(X_phys, feature_names, corr_threshold=0.95):
    corr_matrix = np.corrcoef(X_phys.T)
    n = corr_matrix.shape[0]
    to_remove = set()
    corr_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) >= corr_threshold:
                corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
                if j not in to_remove:
                    to_remove.add(j)

    keep_mask = np.array([i not in to_remove for i in range(n)])
    removed_names = [feature_names[i] for i in to_remove]

    print(f"\n  [相关性过滤] 阈值 |r| >= {corr_threshold}")
    print(f"  发现 {len(corr_pairs)} 对高度相关特征对，移除 {len(removed_names)} 个冗余描述符:")
    for name in removed_names:
        print(f"    - {name}")
    return keep_mask, removed_names, corr_pairs

# ==========================================
# 3. 加载数据
# ==========================================
print("\n" + "=" * 80)
print("步骤 1: 加载数据")
print("=" * 80)

DATA_FILE = "CAT.csv"
possible_paths = [
    DATA_FILE,
    os.path.join(os.getcwd(), DATA_FILE),
    r"F:\python-workspace\Our\YanEr\DBPs_Script\beiyesi\CAT.csv",
    "/mnt/user-data/uploads/sys1.csv"
]

df = None
for path in possible_paths:
    try:
        df = pd.read_csv(path)
        df.columns = ['smiles', 'CAT']
        print(f"✓ 成功加载数据: {path}")
        break
    except:
        continue

if df is None:
    print("❌ 无法找到 CAT.csv 文件!")
    exit(1)

print(f"✓ 数据行数: {len(df)}")
print(f"✓ 目标值范围: [{df['CAT'].min():.2f}, {df['CAT'].max():.2f}]")

# ==========================================
# 4. 提取特征 (与训练时一致)
# ==========================================
print("\n正在提取增强特征 (与训练时一致)...")
X, valid_idx, valid_smiles, n_desc = extract_enhanced_features(df['smiles'])
y = df['CAT'].iloc[valid_idx].values

print(f"✓ 有效样本数: {len(X)}")
print(f"✓ 特征维度: {X.shape[1]} (指纹: 3072, 描述符: {n_desc})")

# ==========================================
# 5. 描述符共线性筛选 (与训练时一致)
# ==========================================
print("\n" + "=" * 80)
print("描述符共线性筛选 (复现训练时的过滤流程)")
print("=" * 80)

n_fp = X.shape[1] - n_desc
X_fp_part = X[:, :n_fp]
X_desc_part = X[:, n_fp:]

descriptor_names_all = [
    'MolWt', 'HeavyAtomMolWt', 'ExactMolWt',
    'MolLogP', 'MolMR', 'TPSA', 'LabuteASA',
    'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
    'NumHeteroatoms', 'NumAromaticRings', 'NumAliphaticRings',
    'RingCount', 'FractionCSP3',
    'MaxAbsPartialCharge', 'MinAbsPartialCharge',
    'MaxPartialCharge', 'MinPartialCharge',
    'F_Count', 'CF2_Count', 'CF3_Count',
    'NOCount', 'NHOHCount',
    'NumValenceElectrons', 'NumRadicalElectrons',
    'BalabanJ', 'BertzCT',
    'Chi0', 'Chi0n', 'Chi0v',
    'Chi1', 'Chi1n', 'Chi1v',
    'Kappa1', 'Kappa2', 'Kappa3',
    'HallKierAlpha', 'Ipc',
]
assert len(descriptor_names_all) == n_desc, "描述符名称数量不匹配！"

print(f"\n原始描述符数量: {n_desc}")

# Step 1: 低方差过滤
lv_mask, lv_removed = remove_low_variance_features(X_desc_part, descriptor_names_all, threshold=0.01)
X_desc_filtered = X_desc_part[:, lv_mask]
names_after_lv = [descriptor_names_all[i] for i in range(n_desc) if lv_mask[i]]
print(f"  低方差过滤后描述符数量: {len(names_after_lv)}")

# Step 2: 相关性过滤（|r| >= 0.95）
corr_mask, corr_removed, corr_pairs = remove_correlated_features(
    X_desc_filtered, names_after_lv, corr_threshold=0.95
)
X_desc_clean = X_desc_filtered[:, corr_mask]
final_descriptor_names = [names_after_lv[i] for i in range(len(names_after_lv)) if corr_mask[i]]
print(f"  相关性过滤后描述符数量: {len(final_descriptor_names)}")
print(f"  保留的描述符: {final_descriptor_names}")

print(f"\n  特征筛选完成:")
print(f"    原始总维度: {X.shape[1]} (指纹 {n_fp} + 描述符 {n_desc})")
print(f"    过滤后描述符维度: {len(final_descriptor_names)}")

# ==========================================
# 6. 特征工程 (与训练时一致)
# ==========================================
print("\n" + "=" * 80)
print("特征工程 (复现训练时流程)")
print("=" * 80)

# --- A. 指纹位方差过滤 (加载训练时保存的过滤器) ---
print("\n[A] 指纹位方差过滤...")
vt_path = os.path.join(MODELS_DIR, "fp_variance_threshold.pkl")
try:
    vt = joblib.load(vt_path)
    X_fp_selected = vt.transform(X_fp_part)
    print(f"✓ 加载指纹过滤器: {vt_path}")
except Exception as e:
    print(f"⚠ 无法加载指纹过滤器({e}), 使用默认阈值0.01重新计算")
    vt = VarianceThreshold(threshold=0.01)
    X_fp_selected = vt.fit_transform(X_fp_part)
n_fp_kept = X_fp_selected.shape[1]
print(f"  保留指纹维度: {n_fp_kept} 位")

# --- B. 描述符衍生特征构造 ---
print("\n[B] 描述符衍生特征构造...")
desc_df = pd.DataFrame(X_desc_clean, columns=final_descriptor_names)

def safe_get(df, col):
    return df[col].values if col in df.columns else np.zeros(len(df))

eng_features = {}
eng_feature_names = []

mw      = safe_get(desc_df, 'MolWt') + 1e-6
tpsa    = safe_get(desc_df, 'TPSA')
logp    = safe_get(desc_df, 'MolLogP')
mr      = safe_get(desc_df, 'MolMR')
hbd     = safe_get(desc_df, 'NumHDonors')
hba     = safe_get(desc_df, 'NumHAcceptors')
ar      = safe_get(desc_df, 'NumAromaticRings')
rc      = safe_get(desc_df, 'RingCount')
csp3    = safe_get(desc_df, 'FractionCSP3')
rotb    = safe_get(desc_df, 'NumRotatableBonds')
f_cnt   = safe_get(desc_df, 'F_Count')
chi0    = safe_get(desc_df, 'Chi0')
kap1    = safe_get(desc_df, 'Kappa1')
kap3    = safe_get(desc_df, 'Kappa3')
bertz   = safe_get(desc_df, 'BertzCT')
max_chg = safe_get(desc_df, 'MaxPartialCharge')
min_chg = safe_get(desc_df, 'MinPartialCharge')

def _add(name, values):
    eng_features[name] = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
    eng_feature_names.append(name)

if 'TPSA' in final_descriptor_names and 'MolWt' in final_descriptor_names:
    _add('TPSA_per_MW', tpsa / mw)
if 'MolLogP' in final_descriptor_names and 'MolWt' in final_descriptor_names:
    _add('LogP_per_MW', logp / mw)
if 'MolMR' in final_descriptor_names and 'MolWt' in final_descriptor_names:
    _add('MR_per_MW', mr / mw)
if 'NumHDonors' in final_descriptor_names and 'NumHAcceptors' in final_descriptor_names:
    _add('HBond_Capacity', hbd * hba)
    _add('HBond_Total', hbd + hba)
if 'NumAromaticRings' in final_descriptor_names and 'RingCount' in final_descriptor_names:
    _add('AromaticRing_Ratio', ar / (rc + 1))
if 'FractionCSP3' in final_descriptor_names and 'NumRotatableBonds' in final_descriptor_names:
    _add('SP3_Flexibility', csp3 * rotb)
if 'F_Count' in final_descriptor_names and 'MolWt' in final_descriptor_names:
    _add('Fluorine_Density', f_cnt / mw)
if 'Chi0' in final_descriptor_names and 'Kappa1' in final_descriptor_names:
    _add('Chi0_x_Kappa1', chi0 * kap1)
if 'BertzCT' in final_descriptor_names and 'MolWt' in final_descriptor_names:
    _add('Complexity_per_MW', bertz / mw)
if 'TPSA' in final_descriptor_names and 'MolLogP' in final_descriptor_names:
    _add('TPSA_x_LogP', tpsa * logp)
if 'MaxPartialCharge' in final_descriptor_names and 'MinPartialCharge' in final_descriptor_names:
    _add('Charge_Span', np.abs(max_chg - min_chg))
if 'Kappa1' in final_descriptor_names and 'Kappa3' in final_descriptor_names:
    _add('Kappa_Anisotropy', kap1 - kap3)

if eng_features:
    X_eng = np.column_stack([eng_features[k] for k in eng_feature_names])
    X_eng = np.nan_to_num(X_eng, nan=0, posinf=0, neginf=0)
    print(f"  构造衍生特征: {len(eng_feature_names)} 个")
    for n in eng_feature_names:
        print(f"    + {n}")
else:
    X_eng = np.zeros((len(X_fp_part), 0))
    print("  无可构造的衍生特征（描述符不足）")

# --- C. 拼合全部特征 ---
X_engineered = np.hstack([X_fp_selected, X_desc_clean, X_eng])
all_feature_names = (
    [f"FP_{i}" for i in range(n_fp_kept)]
    + final_descriptor_names
    + eng_feature_names
)
print(f"\n  特征工程后总维度: {X_engineered.shape[1]}")

# ==========================================
# 7. 特征选择 (三方法投票，与训练时一致)
# ==========================================
print("\n" + "=" * 80)
print("特征选择 (复现训练时三方法投票流程)")
print("=" * 80)

n_fp_final = n_fp_kept
X_fp_final = X_engineered[:, :n_fp_final]
X_desc_eng = X_engineered[:, n_fp_final:]
desc_eng_names = all_feature_names[n_fp_final:]

print(f"\n  输入: 描述符+衍生特征维度 = {X_desc_eng.shape[1]}")

_ss = _SS()
X_desc_eng_std = _ss.fit_transform(X_desc_eng)
X_desc_eng_std = np.nan_to_num(X_desc_eng_std, nan=0, posinf=0, neginf=0)

# 方法1: 互信息
print("[方法1] 互信息特征评分...")
mi_scores = mutual_info_regression(X_desc_eng, y, random_state=42)
mi_ranking = np.argsort(mi_scores)[::-1]
top_k_mi = max(10, int(X_desc_eng.shape[1] * 0.6))
mi_selected = set(mi_ranking[:top_k_mi].tolist())

# 方法2: 随机森林重要性
print("[方法2] 随机森林特征重要性评分...")
rf_selector = _RFR(
    n_estimators=300, max_depth=10,
    min_samples_leaf=2, max_features='sqrt',
    n_jobs=-1, random_state=42
)
rf_selector.fit(X_desc_eng, y)
rf_importances = rf_selector.feature_importances_
rf_ranking = np.argsort(rf_importances)[::-1]
top_k_rf = max(10, int(X_desc_eng.shape[1] * 0.6))
rf_selected = set(rf_ranking[:top_k_rf].tolist())

# 方法3: LASSO
print("[方法3] LASSO特征选择...")
lasso_cv = LassoCV(
    cv=5, max_iter=5000, random_state=42,
    alphas=np.logspace(-4, 0, 50), n_jobs=-1
)
lasso_cv.fit(X_desc_eng_std, y)
lasso_coefs = np.abs(lasso_cv.coef_)
lasso_selected = set(np.where(lasso_coefs > 0)[0].tolist())

# 投票融合
vote_counts = {}
for idx in range(X_desc_eng.shape[1]):
    votes = (
        (1 if idx in mi_selected    else 0) +
        (1 if idx in rf_selected    else 0) +
        (1 if idx in lasso_selected else 0)
    )
    vote_counts[idx] = votes

voting_selected = sorted([i for i, v in vote_counts.items() if v >= 2])
if len(voting_selected) < 5:
    print(f"  [警告] 投票选出特征过少({len(voting_selected)})，回退到RF Top-15")
    voting_selected = sorted(rf_ranking[:15].tolist())

X_desc_eng_selected = X_desc_eng[:, voting_selected]
selected_desc_names = [desc_eng_names[i] for i in voting_selected]

print(f"\n  投票结果: {len(voting_selected)} 个描述符/衍生特征被保留")
print(f"  保留的特征: {selected_desc_names}")

# 拼合最终特征矩阵 (与训练时X_final完全一致)
X_final = np.hstack([X_fp_final, X_desc_eng_selected])
print(f"\n  ★ 最终特征矩阵维度: {X_final.shape[1]}")
print(f"    过滤指纹: {n_fp_final}, 精选描述符/衍生: {len(selected_desc_names)}")

# ==========================================
# 8. 加载模型和缩放器
# ==========================================
print("\n" + "=" * 80)
print("步骤 2: 加载已训练的模型")
print("=" * 80)

# 加载scaler
scaler_path = os.path.join(MODELS_DIR, "robust_scaler.pkl")
try:
    scaler = joblib.load(scaler_path)
    print(f"✓ 加载scaler: {scaler_path}")
except Exception as e:
    print(f"❌ 无法加载scaler: {e}")
    exit(1)

# 加载模型
lgb_model_path = os.path.join(MODELS_DIR, "lightgbm_model.pkl")
cat_model_path = os.path.join(MODELS_DIR, "catboost_model.pkl")
xgb_model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
rf_model_path = os.path.join(MODELS_DIR, "randomforest_model.pkl")
gb_model_path = os.path.join(MODELS_DIR, "gradientboosting_model.pkl")

try:
    lightgbm_model = joblib.load(lgb_model_path)
    catboost_model = joblib.load(cat_model_path)
    xgboost_model = joblib.load(xgb_model_path)
    randomforest_model = joblib.load(rf_model_path)
    gradient_model = joblib.load(gb_model_path)
    print("✓ 所有模型加载完成")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit(1)

# ==========================================
# 9. 数据预处理和划分 (与训练时一致)
# ==========================================
print("\n数据预处理...")
X_scaled = scaler.transform(X_final)
X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

print("使用训练时的数据划分 (test_size=0.15, random_state=42)...")
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_scaled, y, range(len(y)), test_size=0.15, random_state=42
)
X_train_sub, X_val, y_train_sub, y_val, train_sub_indices, val_indices = train_test_split(
    X_train, y_train, train_indices, test_size=0.15, random_state=42
)

# 获取对应的SMILES
train_smiles = [valid_smiles[i] for i in train_indices]
val_smiles = [valid_smiles[i] for i in val_indices]
test_smiles = [valid_smiles[i] for i in test_indices]

print(f"✓ 训练集: {len(X_train_sub)} 条")
print(f"✓ 验证集: {len(X_val)} 条")
print(f"✓ 测试集: {len(X_test)} 条")

# ==========================================
# 10. 模型预测和评估
# ==========================================
print("\n" + "=" * 80)
print("步骤 3: 模型评估")
print("=" * 80)

def evaluate_model(model, X_tr, y_tr, X_te, y_te, name):
    pred_train = model.predict(X_tr)
    pred_test = model.predict(X_te)
    r2_train = r2_score(y_tr, pred_train)
    r2_test = r2_score(y_te, pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_te, pred_test))
    mae_test = mean_absolute_error(y_te, pred_test)
    print(f"\n{name}:")
    print(f"  训练集 R²: {r2_train:.4f}")
    print(f"  测试集 R²: {r2_test:.4f}")
    print(f"  测试集 RMSE: {rmse_test:.4f}")
    print(f"  测试集 MAE: {mae_test:.4f}")
    return r2_train, r2_test, rmse_test, mae_test, pred_test

lightgbm_r2_train, lightgbm_r2_test, lightgbm_rmse_test, lightgbm_mae_test, lightgbm_pred_test = evaluate_model(
    lightgbm_model, X_train, y_train, X_test, y_test, "LightGBM")
catboost_r2_train, catboost_r2_test, catboost_rmse_test, catboost_mae_test, catboost_pred_test = evaluate_model(
    catboost_model, X_train, y_train, X_test, y_test, "CatBoost")
xgboost_r2_train, xgboost_r2_test, xgboost_rmse_test, xgboost_mae_test, xgboost_pred_test = evaluate_model(
    xgboost_model, X_train, y_train, X_test, y_test, "XGBoost")
rf_r2_train, rf_r2_test, rf_rmse_test, rf_mae_test, rf_pred_test = evaluate_model(
    randomforest_model, X_train, y_train, X_test, y_test, "RandomForest")
gb_r2_train, gb_r2_test, gb_rmse_test, gb_mae_test, gb_pred_test = evaluate_model(
    gradient_model, X_train, y_train, X_test, y_test, "GradientBoosting")

# ==========================================
# 11. SHAP分析 (针对精选描述符/衍生特征)
# ==========================================
print("\n" + "=" * 80)
print("步骤 4: SHAP分析 (精选描述符/衍生特征)")
print("=" * 80)

# 提取描述符/衍生特征部分
X_test_desc = X_test[:, n_fp_final:]
X_test_desc_df = pd.DataFrame(X_test_desc, columns=selected_desc_names)

import shap
import numpy as np
import pandas as pd


def get_top_features_shap(model, X_data_full, X_data_df, feature_names, top_n=20, model_name="Model"):
    print(f"\n计算 {model_name} 的SHAP值...")

    # 对于 XGBoost 模型，使用 PermutationExplainer 避免解析错误
    if model_name == "XGBoost":
        print("  使用 PermutationExplainer (因为 TreeExplainer 解析失败)...")

        # 定义预测函数
        def predict_fn(x):
            return model.predict(x)

        # 创建 PermutationExplainer
        explainer = shap.PermutationExplainer(predict_fn, X_data_full, random_state=42)
        shap_values = explainer.shap_values(X_data_full)
    else:
        # 其他模型仍使用 TreeExplainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_data_full)
        except Exception as e:
            print(f"  TreeExplainer 失败 ({e})，改用 PermutationExplainer...")

            def predict_fn(x):
                return model.predict(x)

            explainer = shap.PermutationExplainer(predict_fn, X_data_full, random_state=42)
            shap_values = explainer.shap_values(X_data_full)

    # 提取描述符/衍生特征部分的SHAP值 (偏移量为 n_fp_final)
    shap_values_desc = shap_values[:, n_fp_final:]
    mean_abs_shap = np.abs(shap_values_desc).mean(axis=0)

    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)

    top_features = feature_importance.head(top_n)['Feature'].tolist()
    return shap_values_desc, feature_importance, top_features

lightgbm_shap, lightgbm_imp, lightgbm_top = get_top_features_shap(
    lightgbm_model, X_test, X_test_desc_df, selected_desc_names, TOP_N_FEATURES, "LightGBM")
catboost_shap, catboost_imp, catboost_top = get_top_features_shap(
    catboost_model, X_test, X_test_desc_df, selected_desc_names, TOP_N_FEATURES, "CatBoost")
xgboost_shap, xgboost_imp, xgboost_top = get_top_features_shap(
    xgboost_model, X_test, X_test_desc_df, selected_desc_names, TOP_N_FEATURES, "XGBoost")
rf_shap, rf_imp, rf_top = get_top_features_shap(
    randomforest_model, X_test, X_test_desc_df, selected_desc_names, TOP_N_FEATURES, "RandomForest")
gb_shap, gb_imp, gb_top = get_top_features_shap(
    gradient_model, X_test, X_test_desc_df, selected_desc_names, TOP_N_FEATURES, "GradientBoosting")

# 保存特征重要性
all_imp = pd.concat([
    lightgbm_imp.assign(Model='LightGBM'),
    catboost_imp.assign(Model='CatBoost'),
    xgboost_imp.assign(Model='XGBoost'),
    rf_imp.assign(Model='RandomForest'),
    gb_imp.assign(Model='GradientBoosting')
])
all_imp.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_all.csv"), index=False)

# 保存Top特征对比
top_comparison = pd.DataFrame({
    'Rank': range(1, TOP_N_FEATURES+1),
    'LightGBM_Feature': lightgbm_top,
    'LightGBM_SHAP': [lightgbm_imp[lightgbm_imp['Feature']==f]['Mean_Abs_SHAP'].values[0] for f in lightgbm_top],
    'CatBoost_Feature': catboost_top,
    'CatBoost_SHAP': [catboost_imp[catboost_imp['Feature']==f]['Mean_Abs_SHAP'].values[0] for f in catboost_top],
    'XGBoost_Feature': xgboost_top,
    'XGBoost_SHAP': [xgboost_imp[xgboost_imp['Feature']==f]['Mean_Abs_SHAP'].values[0] for f in xgboost_top],
    'RandomForest_Feature': rf_top,
    'RandomForest_SHAP': [rf_imp[rf_imp['Feature']==f]['Mean_Abs_SHAP'].values[0] for f in rf_top],
    'GradientBoosting_Feature': gb_top,
    'GradientBoosting_SHAP': [gb_imp[gb_imp['Feature']==f]['Mean_Abs_SHAP'].values[0] for f in gb_top]
})
top_comparison.to_csv(os.path.join(OUTPUT_DIR, f"top{TOP_N_FEATURES}_features.csv"), index=False)

# ==========================================
# 12. SHAP可视化函数 (与原shap_plot.py相同)
# ==========================================
def create_optimized_cmap(base_cmap, start=0.2, end=0.9):
    colors = base_cmap(np.linspace(start, end, 256))
    return LinearSegmentedColormap.from_list('optimized', colors)

def plot_shap_barplot_with_rose(shap_values, X_data, feature_names, model_name, color_scheme='viridis'):
    """
    绘制条形图+玫瑰图组合 (正方形条形图)
    特征名称在最右边,与条形精确对齐,无边框,不遮挡图形
    """
    # 选择指定特征的数据
    feature_indices = [list(X_data.columns).index(f) for f in feature_names]
    shap_subset = shap_values[:, feature_indices]

    # 计算排序信息
    mean_abs_shap = np.abs(shap_subset).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_names)
    shap_series.sort_values(ascending=False, inplace=True)
    sorted_features = shap_series.index.tolist()
    sorted_shap_values = shap_series.values

    # 玫瑰图数据
    base_length, fixed_increment, colored_ring_width = 2.0, 0.25, 1.0
    num_vars = len(feature_names)
    one_oclock_offset = np.pi / 21
    percentages = (sorted_shap_values / sorted_shap_values.sum()) * 100
    widths = (sorted_shap_values / sorted_shap_values.sum()) * 2 * np.pi
    thetas = np.cumsum([0] + widths[:-1].tolist()) - one_oclock_offset
    total_lengths = [base_length + i * fixed_increment for i in range(num_vars)]
    inner_heights = [max(0, tl - colored_ring_width) for tl in total_lengths]
    inner_colors = ['#F5F5F5', '#FFFFFF'] * (num_vars // 2 + 1)

    # 颜色映射
    cmap_base = COLOR_SCHEMES[color_scheme]
    cmap = create_optimized_cmap(cmap_base)
    color_norm = mcolors.Normalize(vmin=np.quantile(sorted_shap_values, 0.25),
                                   vmax=np.quantile(sorted_shap_values, 0.75))
    colors = cmap(color_norm(sorted_shap_values))

    # 正方形图形布局 - 为右侧特征名称预留更多空间
    fig_width = 13
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=600, facecolor='white')

    # 布局参数 - 主图占据左侧大部分空间
    left_margin = 0.10
    right_margin = 0.34  # 为特征名称预留右侧空间
    bottom_margin = 0.12
    top_margin = 0.12
    main_plot_width = 1 - left_margin - right_margin
    plot_bottom = bottom_margin
    plot_height = 1 - bottom_margin - top_margin

    # 左侧颜色条
    cbar_left = 0.02
    colorbar_width = 0.015
    ax_cbar = fig.add_axes([cbar_left, plot_bottom, colorbar_width, plot_height])
    sm = ScalarMappable(cmap=cmap, norm=color_norm)
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation='vertical')
    cbar.set_ticks([])
    cbar.ax.yaxis.set_ticks_position('left')
    ax_cbar.text(-3.5, 0.98, 'High', transform=ax_cbar.transAxes, ha='center',
                 va='bottom', fontsize=24, fontweight='bold', family='Arial')
    ax_cbar.text(-3.5, 0.02, 'Low', transform=ax_cbar.transAxes, ha='center',
                 va='top', fontsize=24, fontweight='bold', family='Arial')
    cbar.outline.set_visible(False)
    ax_cbar.text(-3.5, 0.5, 'SHAP Value', transform=ax_cbar.transAxes,
                 fontsize=28, rotation=90, va='center', fontweight='bold', ha='center', family='Arial')
    ax_cbar.set_facecolor('white')

    # 条形图 (主图区域)
    main_ax_left = cbar_left + colorbar_width + 0.05
    ax0 = fig.add_axes([main_ax_left, plot_bottom, main_plot_width, plot_height])
    ax0.xaxis.tick_bottom()
    ax0.xaxis.set_label_position("bottom")
    ax0.invert_xaxis()
    ax0.set_xlim(max(sorted_shap_values) * 1.06, 0)
    # 绘制条形图，设置条形高度
    bar_height = 0.65
    bar_positions = range(len(sorted_features))
    bars = ax0.barh(y=bar_positions, width=sorted_shap_values, color=colors,
                    height=bar_height, edgecolor='white', linewidth=1.2)

    # 数值标签：强制放在黑色竖线 x=0 左侧，避免越过右边界
    max_val = np.max(sorted_shap_values)
    label_gap = max_val * 0.012  # 标签与条形末端的间距
    min_left_from_zero = max_val * 0.035  # 距离 x=0 黑线的最小安全距离

    for i, (bar, value) in enumerate(zip(bars, sorted_shap_values)):
        # 原则：标签尽量靠近条形末端，但不能太靠近 x=0
        label_x = max(value - label_gap, min_left_from_zero)

        ax0.text(
            label_x, i, f'{value:.4f}',
            ha='right', va='center',
            fontsize=12, fontweight='bold', family='Arial',
            clip_on=True,
            bbox=dict(
                boxstyle='round,pad=0.18',
                facecolor='white',
                edgecolor='#CCCCCC',
                alpha=0.9
            )
        )

    ax0.invert_yaxis()
    ax0.set_xlabel('SHAP Value', size=28, labelpad=8, fontweight='bold', family='Arial')
    ax0.set_yticks([])
    ax0.spines[['left', 'top']].set_visible(False)
    ax0.spines['right'].set_position(('data', 0))
    ax0.spines['right'].set_visible(True)
    ax0.spines['bottom'].set_visible(True)
    ax0.tick_params(axis='x', which='major', direction='in', labelsize=18,
                    length=8, pad=10, width=2)
    for label in ax0.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')
    ax0.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax0.tick_params(axis='x', which='minor', direction='in', length=5, width=1.5)
    for spine in ax0.spines.values():
        spine.set_linewidth(3)
        spine.set_color('#333333')

    # 玫瑰图 (嵌入条形图左下角)
    inset_size = min(main_plot_width, plot_height) * 0.65
    inset_left = main_ax_left - 0.08
    inset_bottom = plot_bottom - 0.02
    inset_ax_rect = [inset_left, inset_bottom, inset_size, inset_size]
    ax1 = fig.add_axes(inset_ax_rect, projection='polar')
    ax1.patch.set_alpha(0)
    ax1.bar(x=thetas, height=inner_heights, width=widths, color=inner_colors,
            align='edge', edgecolor='white', linewidth=2.0)
    ax1.bar(x=thetas, height=[colored_ring_width] * num_vars, width=widths,
            bottom=inner_heights, color=colors, align='edge', edgecolor='white',
            linewidth=2.0)
    for i in range(num_vars):
        label_angle_rad = thetas[i] + widths[i] / 2
        label_radius = total_lengths[i] + 0.6
        ax1.text(label_angle_rad, label_radius, f'{percentages[i]:.2f}%',
                 ha='center', va='center', fontsize=7, fontweight='bold', family='Arial',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                           edgecolor='#CCCCCC', alpha=0.9))
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.spines['polar'].set_visible(False)
    ax1.grid(False)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_ylim(0, max(total_lengths) + 2.5)

    # 特征名称 - 使用ax0的坐标系精确对齐条形位置
    label_fontsize = 24
    label_x = 1 - right_margin + 0.04

    for i, feature in enumerate(sorted_features):
        display_coords = ax0.transData.transform((0, i))
        fig_coords = fig.transFigure.inverted().transform(display_coords)
        y_position = fig_coords[1]

        fig.text(label_x, y_position, feature,
                 ha='left', va='center',
                 color='black', fontsize=label_fontsize, fontweight='bold', family='Arial')

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, f'shap_barplot_rose_{model_name}_{color_scheme}.jpg')
    plt.savefig(output_file, dpi=600, bbox_inches=None, facecolor='white', edgecolor='none')
    plt.close()
    return output_file

def plot_shap_beeswarm(shap_values, X_data, feature_names, model_name, color_scheme='viridis'):
    """
    绘制蜂窝图 (单独输出)
    """
    # 选择指定特征的数据
    feature_indices = [list(X_data.columns).index(f) for f in feature_names]
    shap_subset = shap_values[:, feature_indices]
    X_subset = X_data[feature_names]

    # 计算排序
    mean_abs_shap = np.abs(shap_subset).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_names)
    shap_series.sort_values(ascending=False, inplace=True)
    sorted_features = shap_series.index.tolist()

    # 重排SHAP值以匹配排序
    sorted_indices = [feature_names.index(f) for f in sorted_features]
    shap_subset_sorted = shap_subset[:, sorted_indices]
    X_subset_sorted = X_subset[sorted_features]

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10), dpi=600, facecolor='white')

    # 使用shap.summary_plot绘制
    shap.summary_plot(shap_subset_sorted, X_subset_sorted,
                      plot_type="dot", show=False,
                      max_display=len(sorted_features), cmap=color_scheme)

    # 设置标签和刻度
    ax = plt.gca()
    ax.set_xlabel("SHAP Value", fontsize=28, family='Arial', fontweight='bold', labelpad=12)
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=24, direction='in', width=2, length=8, pad=10)
    ax.tick_params(axis='y', labelsize=18, direction='out', width=2, length=6, pad=8)

    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')
        label.set_fontsize(22)

    ax.spines['bottom'].set_linewidth(3)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_linewidth(3)
    ax.spines['left'].set_color('#333333')
    for spine_name in ['top', 'right']:
        if spine_name in ax.spines:
            ax.spines[spine_name].set_visible(False)

    # 右侧颜色条设置
    if len(fig.axes) > 3:
        cbar_ax = fig.axes[-1]
        cbar_ax.set_ylabel('Feature Value', size=28, rotation=270,
                           labelpad=15, fontweight='bold', family='Arial')
        cbar_ax.tick_params(labelsize=24, width=2)
        tick_labels = cbar_ax.get_yticklabels()
        if len(tick_labels) >= 2:
            tick_labels[0].set_text("Low")
            tick_labels[-1].set_text("High")
            for tick_label in tick_labels:
                tick_label.set_fontweight('bold')
                tick_label.set_fontfamily('Arial')
            cbar_ax.set_yticklabels(tick_labels, fontsize=24, fontweight='bold', family='Arial')

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, f'shap_beeswarm_{model_name}_{color_scheme}.jpg')
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return output_file

def plot_shap_violin(shap_values, X_data, feature_names, model_name, color_scheme='viridis'):
    """
    绘制SHAP小提琴图 - 使用与蜂窝图一致的配色
    """
    # 选择指定特征的数据
    feature_indices = [list(X_data.columns).index(f) for f in feature_names]
    shap_subset = shap_values[:, feature_indices]
    X_subset = X_data[feature_names]

    # 计算排序
    mean_abs_shap = np.abs(shap_subset).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_names)
    shap_series.sort_values(ascending=False, inplace=True)
    sorted_features = shap_series.index.tolist()

    # 重排SHAP值
    sorted_indices = [feature_names.index(f) for f in sorted_features]
    shap_subset_sorted = shap_subset[:, sorted_indices]
    X_subset_sorted = X_subset[sorted_features]

    # 创建图形
    fig, ax = plt.subplots(figsize=(20, 10), dpi=600, facecolor='white')

    # 使用指定配色方案绘制
    shap.summary_plot(shap_subset_sorted, X_subset_sorted,
                      plot_type="layered_violin", cmap=color_scheme,
                      show=False, max_display=len(feature_names))

    # 设置标签
    plt.xlabel('SHAP Value', fontsize=28, fontweight='bold', family='Arial', labelpad=12)

    # 获取当前坐标轴
    ax = plt.gca()

    # 将主Y轴（特征名称）移到右边
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # 设置刻度
    ax.tick_params(axis='x', labelsize=24, width=2, length=6)
    ax.tick_params(axis='y', labelsize=24, width=2, length=6)

    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')
        label.set_fontsize(22)

    yticklabels = [label.get_text().lstrip('- ') for label in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels)

    # 获取当前x轴范围
    current_xlim = ax.get_xlim()
    ax.spines['left'].set_bounds(ax.get_ylim()[0], ax.get_ylim()[1])
    ax.spines['bottom'].set_bounds(current_xlim[0], current_xlim[1])

    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # 调整颜色条
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[-1]
        cbar_ax.set_ylabel('')

        cbar_pos = cbar_ax.get_position()
        new_pos = [0.08, cbar_pos.y0, cbar_pos.width, cbar_pos.height]
        cbar_ax.set_position(new_pos)

        main_ax_pos = ax.get_position()
        ax.set_position([0.15, main_ax_pos.y0, main_ax_pos.width - 0.05, main_ax_pos.height])

        cbar_ax.yaxis.set_ticks_position('left')
        cbar_ax.yaxis.set_label_position('left')

        cbar_ax.text(-3.5, 0.5, 'Feature Value', transform=cbar_ax.transAxes,
                     fontsize=28, rotation=90, va='center', fontweight='bold', ha='center', family='Arial')
        cbar_ax.set_facecolor('white')

        cbar_ax.tick_params(labelsize=24, width=2)
        for tick_label in cbar_ax.get_yticklabels():
            tick_label.set_fontweight('bold')
            tick_label.set_fontfamily('Arial')

    output_file = os.path.join(OUTPUT_DIR, f'shap_violin_{model_name}_{color_scheme}.jpg')
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return output_file

def plot_shap_beeswarm_rose_combined(
        shap_values,
        X_data_df,
        importance_df,
        model_name,
        top_n=15,
        color_scheme='RdBu_r',
        output_dir=None
):
    """
    左侧：SHAP蜂窝图
    右侧：玫瑰图（SHAP贡献百分比）
    """
    if output_dir is None:
        output_dir = SHAP_OUTPUT_DIR

    print(f"\n正在生成 {model_name} 的蜂巢图 + 玫瑰图组合...")

    SHAP_X = 0.05
    SHAP_Y = 0.15
    SHAP_W = 0.55
    SHAP_H = 0.75

    POLAR_X = 0.72
    POLAR_Y = 0.1
    POLAR_SIZE = 0.75
    POLAR_BOTTOM_VAL = 15
    POLAR_GAP = 2.0

    # --- 修改点 1：定义扇形缩放系数（1.5 倍，可自行调整）---
    SECTOR_SCALE = 1.5   # 扇形拉长倍数

    top_features = importance_df.sort_values('Mean_Abs_SHAP', ascending=False).head(top_n)
    feature_names = top_features['Feature'].tolist()
    mean_abs_shap = top_features['Mean_Abs_SHAP'].values

    feature_indices = [list(X_data_df.columns).index(f) for f in feature_names]
    shap_subset = shap_values[:, feature_indices]
    X_subset = X_data_df[feature_names]

    percentages = mean_abs_shap / mean_abs_shap.sum() * 100
    # 应用缩放，得到新的扇形高度
    percentages_scaled = percentages * SECTOR_SCALE

    n_features = len(feature_names)
    total_angle = 2 * np.pi
    gap_ratio = 0.1
    width = (total_angle / n_features) * (1 - gap_ratio)
    theta = np.linspace(0, total_angle, n_features, endpoint=False)
    theta = theta + width / 2 + (total_angle / n_features) * gap_ratio / 2

    circle_theta = np.linspace(0, 2 * np.pi, 100)
    circle_r = np.full_like(circle_theta, POLAR_BOTTOM_VAL)

    fig = plt.figure(figsize=(40, 10), dpi=600, facecolor='white')
    cmap = COLOR_SCHEMES[color_scheme] if color_scheme in COLOR_SCHEMES else plt.cm.RdBu_r
    vmin = 0
    vmax = np.ceil(max(mean_abs_shap)) if len(mean_abs_shap) else 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-12)
    colors = cmap(norm(mean_abs_shap))

    # 左侧 SHAP 蜂窝图
    ax1_pos = [SHAP_X, SHAP_Y, SHAP_W, SHAP_H]
    ax1 = fig.add_axes(ax1_pos)
    plt.sca(ax1)

    shap.summary_plot(
        shap_subset,
        X_subset,
        plot_type="dot",
        show=False,
        max_display=len(feature_names),
        sort=False,
        cmap=cmap,
        color_bar=True
    )

    ax = plt.gca()
    for coll in ax.collections:
        try:
            coll.set_sizes([40])
        except Exception:
            pass

    fig = plt.gcf()
    axes = fig.axes
    cbar_ax = axes[-1]

    ax1.set_xlabel("SHAP Value", fontsize=28, fontweight='bold', family='Arial', labelpad=12)
    ax1.tick_params(axis='x', which='major', direction='in', labelsize=18, length=8, pad=10, width=2)
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax1.tick_params(axis='x', which='minor', direction='in', length=5, width=1.5)
    ax1.tick_params(axis='y', which='major', direction='in', labelsize=18, width=2)

    for label in ax1.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')

    for spine in ax1.spines.values():
        spine.set_linewidth(3)
        spine.set_color('#333333')

    cbar_ax.set_ylabel("Feature value", fontsize=16, fontweight='bold',
                       family='Arial', rotation=90)
    cbar_ax.yaxis.set_label_coords(2.0, 0.5)
    cbar_ax.tick_params(labelsize=14, width=2, length=6, direction='in')
    for label in cbar_ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')

    # 右侧玫瑰图（极坐标）
    ax2_pos = [POLAR_X, POLAR_Y, POLAR_SIZE, POLAR_SIZE]
    ax2 = fig.add_axes(ax2_pos, projection='polar')

    norm = mcolors.Normalize(vmin=min(mean_abs_shap), vmax=max(mean_abs_shap) if len(mean_abs_shap) else 1)
    colors = cmap(norm(mean_abs_shap))

    # 绘制扇形（使用缩放后的高度）
    ax2.bar(
        theta,
        percentages_scaled,
        width=width,
        bottom=POLAR_BOTTOM_VAL + POLAR_GAP,
        color=colors,
        edgecolor='black',
        linewidth=0.8
    )

    ax2.plot(circle_theta, circle_r, color='black', linewidth=1, linestyle='-')
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_axis_off()

    # --- 修改点 2：外部标签字体大小从 14 改为 18，且位置基于缩放后的百分比 ---
    for angle, percent, name, raw_val in zip(theta, percentages, feature_names, mean_abs_shap):
        percent_scaled = percent * SECTOR_SCALE
        angle_deg = np.degrees(angle)
        visual_top = POLAR_BOTTOM_VAL + POLAR_GAP + percent_scaled

        if 0 <= angle_deg < 180:
            rotation = 90 - angle_deg
            alignment_ha = 'left'
            alignment_va = 'center'
            pos_outer = visual_top + 0.5
        else:
            rotation = 270 - angle_deg
            alignment_ha = 'right'
            alignment_va = 'center'
            pos_outer = visual_top + 0.5

        ax2.text(angle, pos_outer, f"{name}\n{percent:.1f}%",
                 ha=alignment_ha, va=alignment_va,
                 rotation=rotation, rotation_mode='anchor',
                 fontsize=10,                # ← 字体大小改为 18
                 fontweight='bold', family='Arial',
                 color='black')
                 # bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                 # bbox=dict( facecolor="white",
                 #           alpha=0.8, edgecolor="none"))

    # 内部数值标签（基于缩放后的位置）
    for angle, percent, raw_val in zip(theta, percentages, mean_abs_shap):
        percent_scaled = percent * SECTOR_SCALE
        angle_deg = np.degrees(angle)
        visual_top = POLAR_BOTTOM_VAL + POLAR_GAP + percent_scaled
        text_radius = visual_top - percent_scaled * 0.12
        rotation = -angle_deg if 0 <= angle_deg < 180 else 180 - angle_deg

        ax2.text(
            angle, text_radius, f"{raw_val:.3f}",
            ha='center', va='center',
            rotation=rotation, rotation_mode='anchor',
            fontsize=9, fontweight='bold', color='white'
        )

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = ax2.inset_axes([0.48, 0.42, 0.04, 0.12], transform=ax2.transAxes)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('SHAP Value', fontsize=8, fontweight='bold', family='Arial', rotation=0)
    cbar.ax.yaxis.set_label_coords(0.5, 1.3)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.tick_params(labelsize=10)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')

    output_path = os.path.join(output_dir, f"shap_beeswarm_rose_{model_name}_{color_scheme}.jpg")
    output_path_pdf = os.path.join(output_dir, f"shap_beeswarm_rose_{model_name}_{color_scheme}.pdf")
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ 已保存: {os.path.basename(output_path)}")
    return output_path

# ==========================================
# 13. 生成所有SHAP可视化
# ==========================================
print("\n" + "=" * 80)
print("步骤 5: 生成SHAP可视化")
print("=" * 80)

models_to_plot = [
    ('LightGBM', lightgbm_shap, lightgbm_imp, lightgbm_top),
    ('CatBoost', catboost_shap, catboost_imp, catboost_top),
    ('XGBoost', xgboost_shap, xgboost_imp, xgboost_top),
    ('RandomForest', rf_shap, rf_imp, rf_top),
    ('GradientBoosting', gb_shap, gb_imp, gb_top)
]

for model_name, shap_vals, imp_df, top_feats in models_to_plot:
    print(f"\n{model_name} 模型可视化:")
    for scheme in COLOR_SCHEMES:
        plot_shap_barplot_with_rose(shap_vals, X_test_desc_df, top_feats, model_name, scheme)
        plot_shap_beeswarm(shap_vals, X_test_desc_df, top_feats, model_name, scheme)
        plot_shap_violin(shap_vals, X_test_desc_df, top_feats, model_name, scheme)
        plot_shap_beeswarm_rose_combined(shap_vals, X_test_desc_df, imp_df, model_name, TOP_N_FEATURES, scheme, OUTPUT_DIR)

# ==========================================
# 14. 生成分析报告
# ==========================================
print("\n" + "=" * 80)
print("步骤 6: 生成分析报告")
print("=" * 80)

report = f"""
SHAP分析报告 - CAT数据集
{'=' * 80}
数据集信息:
- 原始样本数: {len(df)}
- 有效样本数: {len(X)}
- 训练集: {len(X_train)} 条 (85%)
- 测试集: {len(X_test)} 条 (15%)
- 总特征维度: {X_final.shape[1]}
- 分析的描述符数: {len(selected_desc_names)}
- 选取Top特征数: {TOP_N_FEATURES}

模型性能 (测试集):
1. LightGBM: R²={lightgbm_r2_test:.4f}, RMSE={lightgbm_rmse_test:.4f}, MAE={lightgbm_mae_test:.4f}
2. CatBoost: R²={catboost_r2_test:.4f}, RMSE={catboost_rmse_test:.4f}, MAE={catboost_mae_test:.4f}
3. XGBoost: R²={xgboost_r2_test:.4f}, RMSE={xgboost_rmse_test:.4f}, MAE={xgboost_mae_test:.4f}
4. RandomForest: R²={rf_r2_test:.4f}, RMSE={rf_rmse_test:.4f}, MAE={rf_mae_test:.4f}
5. GradientBoosting: R²={gb_r2_test:.4f}, RMSE={gb_rmse_test:.4f}, MAE={gb_mae_test:.4f}

输出目录: {OUTPUT_DIR}
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""
with open(os.path.join(OUTPUT_DIR, "SHAP_Analysis_Report.txt"), 'w', encoding='utf-8') as f:
    f.write(report)
print(report)

print("\n✅ 所有任务完成!")
print(f"📁 输出目录: {OUTPUT_DIR}")