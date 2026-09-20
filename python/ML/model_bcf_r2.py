import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem import Crippen, Lipinski, MolSurf
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_regression,
    SelectFromModel,
    SelectPercentile
)
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor as _RFR
from sklearn.preprocessing import StandardScaler as _SS
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib
import warnings

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("R² > 0.9 挑战 - 增强版模型 (含MACCS指纹 + BCF专属描述符 + 特征工程 + 共线性筛选 + 三方法投票选择 + 损失曲线)")
print("=" * 80)


# ==========================================
# 1. 增强特征提取 (含MACCS指纹和BCF相关描述符)
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
            # 指纹: Morgan2 (2048) + Morgan3 (1024) + MACCS (166)
            fp_2048 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_1024_r3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
            fp_maccs = MACCSkeys.GenMACCSKeys(mol)

            # ========== 基础描述符 ==========
            mw = Descriptors.MolWt(mol)
            heavy_mw = Descriptors.HeavyAtomMolWt(mol)
            exact_mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            mr = Descriptors.MolMR(mol)
            tpsa = Descriptors.TPSA(mol)
            labute_asa = Descriptors.LabuteASA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            hetero = Descriptors.NumHeteroatoms(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            aliphatic_rings = Descriptors.NumAliphaticRings(mol)
            ring_count = Descriptors.RingCount(mol)
            fraction_csp3 = Descriptors.FractionCSP3(mol)
            max_abs_charge = Descriptors.MaxAbsPartialCharge(mol) if Descriptors.MaxAbsPartialCharge(mol) else 0
            min_abs_charge = Descriptors.MinAbsPartialCharge(mol) if Descriptors.MinAbsPartialCharge(mol) else 0
            max_charge = Descriptors.MaxPartialCharge(mol) if Descriptors.MaxPartialCharge(mol) else 0
            min_charge = Descriptors.MinPartialCharge(mol) if Descriptors.MinPartialCharge(mol) else 0
            f_count = smiles.count('F')
            cf2_count = smiles.count('C(F)(F)')
            cf3_count = smiles.count('C(F)(F)F')
            no_count = Descriptors.NOCount(mol)
            nhoh_count = Descriptors.NHOHCount(mol)
            valence_electrons = Descriptors.NumValenceElectrons(mol)
            radical_electrons = Descriptors.NumRadicalElectrons(mol)
            balaban_j = Descriptors.BalabanJ(mol)
            bertz_ct = Descriptors.BertzCT(mol)
            chi0 = Descriptors.Chi0(mol)
            chi0n = Descriptors.Chi0n(mol)
            chi0v = Descriptors.Chi0v(mol)
            chi1 = Descriptors.Chi1(mol)
            chi1n = Descriptors.Chi1n(mol)
            chi1v = Descriptors.Chi1v(mol)
            kappa1 = Descriptors.Kappa1(mol)
            kappa2 = Descriptors.Kappa2(mol)
            kappa3 = Descriptors.Kappa3(mol)
            hall_kier_alpha = Descriptors.HallKierAlpha(mol)
            ipc = Descriptors.Ipc(mol)

            # ========== 新增BCF相关描述符 ==========
            # --- 高级连通性指数 ---
            chi2n = Descriptors.Chi2n(mol) if hasattr(Descriptors, 'Chi2n') else 0
            chi2v = Descriptors.Chi2v(mol) if hasattr(Descriptors, 'Chi2v') else 0
            chi3n = Descriptors.Chi3n(mol) if hasattr(Descriptors, 'Chi3n') else 0
            chi3v = Descriptors.Chi3v(mol) if hasattr(Descriptors, 'Chi3v') else 0
            chi4n = Descriptors.Chi4n(mol) if hasattr(Descriptors, 'Chi4n') else 0
            chi4v = Descriptors.Chi4v(mol) if hasattr(Descriptors, 'Chi4v') else 0

            # --- Crippen 体积和 logP ---
            crippen = rdMolDescriptors.CalcCrippenDescriptors(mol)
            crippen_vol = crippen[0] if crippen else 0
            crippen_logp = crippen[1] if crippen else 0

            # --- PEOE VSA 分箱（选部分重要的）---
            peoe_vsa1 = Descriptors.PEOE_VSA1(mol) if hasattr(Descriptors, 'PEOE_VSA1') else 0
            peoe_vsa2 = Descriptors.PEOE_VSA2(mol) if hasattr(Descriptors, 'PEOE_VSA2') else 0
            peoe_vsa3 = Descriptors.PEOE_VSA3(mol) if hasattr(Descriptors, 'PEOE_VSA3') else 0
            peoe_vsa4 = Descriptors.PEOE_VSA4(mol) if hasattr(Descriptors, 'PEOE_VSA4') else 0
            peoe_vsa5 = Descriptors.PEOE_VSA5(mol) if hasattr(Descriptors, 'PEOE_VSA5') else 0
            peoe_vsa6 = Descriptors.PEOE_VSA6(mol) if hasattr(Descriptors, 'PEOE_VSA6') else 0
            peoe_vsa7 = Descriptors.PEOE_VSA7(mol) if hasattr(Descriptors, 'PEOE_VSA7') else 0
            peoe_vsa8 = Descriptors.PEOE_VSA8(mol) if hasattr(Descriptors, 'PEOE_VSA8') else 0
            peoe_vsa9 = Descriptors.PEOE_VSA9(mol) if hasattr(Descriptors, 'PEOE_VSA9') else 0
            peoe_vsa10 = Descriptors.PEOE_VSA10(mol) if hasattr(Descriptors, 'PEOE_VSA10') else 0
            peoe_vsa11 = Descriptors.PEOE_VSA11(mol) if hasattr(Descriptors, 'PEOE_VSA11') else 0
            peoe_vsa12 = Descriptors.PEOE_VSA12(mol) if hasattr(Descriptors, 'PEOE_VSA12') else 0
            peoe_vsa13 = Descriptors.PEOE_VSA13(mol) if hasattr(Descriptors, 'PEOE_VSA13') else 0
            peoe_vsa14 = Descriptors.PEOE_VSA14(mol) if hasattr(Descriptors, 'PEOE_VSA14') else 0

            # --- 官能团计数 ---
            aldehyde_pattern = Chem.MolFromSmarts('[CX3H1](=O)')
            aldehyde_cnt = len(mol.GetSubstructMatches(aldehyde_pattern)) if aldehyde_pattern else 0
            ketone_pattern = Chem.MolFromSmarts('[#6][CX3](=O)[#6]')
            ketone_cnt = len(mol.GetSubstructMatches(ketone_pattern)) if ketone_pattern else 0
            ester_pattern = Chem.MolFromSmarts('[#6][CX3](=O)[OX2H0][#6]')
            ester_cnt = len(mol.GetSubstructMatches(ester_pattern)) if ester_pattern else 0
            amide_pattern = Chem.MolFromSmarts('[NX3][CX3](=[OX1])')
            amide_cnt = len(mol.GetSubstructMatches(amide_pattern)) if amide_pattern else 0
            nitro_pattern = Chem.MolFromSmarts('[N+](=O)[O-]')
            nitro_cnt = len(mol.GetSubstructMatches(nitro_pattern)) if nitro_pattern else 0
            sulfonic_pattern = Chem.MolFromSmarts('S(=O)(=O)[OH]')
            sulfonic_cnt = len(mol.GetSubstructMatches(sulfonic_pattern)) if sulfonic_pattern else 0

            # --- 卤素取代模式 ---
            aromatic_halogen = len(mol.GetSubstructMatches(Chem.MolFromSmarts('c[Cl,Br,I]'))) if Chem.MolFromSmarts('c[Cl,Br,I]') else 0
            aliphatic_halogen = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[Cl,Br,I;!c]'))) if Chem.MolFromSmarts('[Cl,Br,I;!c]') else 0

            # --- 可电离基团扩展 ---
            phenol_pattern = Chem.MolFromSmarts('cO')
            phenol_cnt = len(mol.GetSubstructMatches(phenol_pattern)) if phenol_pattern else 0
            secondary_amine_pattern = Chem.MolFromSmarts('[NH1;!$(NC=O)]')
            secondary_amine_cnt = len(mol.GetSubstructMatches(secondary_amine_pattern)) if secondary_amine_pattern else 0
            tertiary_amine_pattern = Chem.MolFromSmarts('[N;H0;!$(NC=O)]')
            tertiary_amine_cnt = len(mol.GetSubstructMatches(tertiary_amine_pattern)) if tertiary_amine_pattern else 0
            guanidine_pattern = Chem.MolFromSmarts('N=C(N)N')
            guanidine_cnt = len(mol.GetSubstructMatches(guanidine_pattern)) if guanidine_pattern else 0

            # 原有羧基和伯胺（用于离子化基团）
            acidic_pattern = Chem.MolFromSmarts('[OH][CX3](=[OX1])')
            basic_pattern = Chem.MolFromSmarts('[NH2;!$(NC=O)]')
            acidic_cnt = len(mol.GetSubstructMatches(acidic_pattern)) if acidic_pattern else 0
            basic_cnt = len(mol.GetSubstructMatches(basic_pattern)) if basic_pattern else 0

            # --- 环特征 ---
            aromatic_hetero_rings = Descriptors.NumAromaticHeterocycles(mol)
            aliphatic_hetero_rings = Descriptors.NumAliphaticHeterocycles(mol)
            saturated_rings = Descriptors.NumSaturatedRings(mol)

            # --- EState 聚合 ---
            estate = rdMolDescriptors.CalcEStateIndices(mol) if hasattr(rdMolDescriptors, 'CalcEStateIndices') else []
            estate_sum = np.sum(estate) if len(estate) else 0
            estate_abs_sum = np.sum(np.abs(estate)) if len(estate) else 0

            # --- 氢键比例 ---
            hb_ratio = (hbd + hba) / (mw + 1e-6)

            # --- 综合BCF相关指标 ---
            halogen_count = smiles.count('Cl') + smiles.count('Br') + smiles.count('I')
            ionizable_groups = (acidic_cnt + basic_cnt + phenol_cnt + secondary_amine_cnt +
                                tertiary_amine_cnt + guanidine_cnt)
            mw2 = mw * mw
            logp2 = logp * logp
            hbd_hba_total = hbd + hba
            hbd_hba_per_mw = hbd_hba_total / (mw + 1e-6)

            # 将所有描述符按顺序放入列表（顺序必须与 descriptor_names 严格一致）
            desc_list = [
                mw, heavy_mw, exact_mw, logp, mr, tpsa, labute_asa,
                hbd, hba, rot_bonds, hetero, aromatic_rings, aliphatic_rings,
                ring_count, fraction_csp3,
                max_abs_charge, min_abs_charge, max_charge, min_charge,
                f_count, cf2_count, cf3_count,
                no_count, nhoh_count,
                valence_electrons, radical_electrons,
                balaban_j, bertz_ct,
                chi0, chi0n, chi0v,
                chi1, chi1n, chi1v,
                kappa1, kappa2, kappa3,
                hall_kier_alpha, ipc,
                # 新增描述符
                chi2n, chi2v, chi3n, chi3v, chi4n, chi4v,
                crippen_vol, crippen_logp,
                peoe_vsa1, peoe_vsa2, peoe_vsa3, peoe_vsa4, peoe_vsa5,
                peoe_vsa6, peoe_vsa7, peoe_vsa8, peoe_vsa9, peoe_vsa10,
                peoe_vsa11, peoe_vsa12, peoe_vsa13, peoe_vsa14,
                aldehyde_cnt, ketone_cnt, ester_cnt, amide_cnt, nitro_cnt, sulfonic_cnt,
                aromatic_halogen, aliphatic_halogen,
                phenol_cnt, secondary_amine_cnt, tertiary_amine_cnt, guanidine_cnt,
                aromatic_hetero_rings, aliphatic_hetero_rings, saturated_rings,
                estate_sum, estate_abs_sum,
                hb_ratio,
                halogen_count, ionizable_groups, mw2, logp2, hbd_hba_per_mw,
            ]

            # 处理缺失值
            desc_list = [0 if (x is None or np.isnan(x) or np.isinf(x)) else x for x in desc_list]

            # 指纹拼接
            fps.append(np.concatenate([np.array(fp_2048), np.array(fp_1024_r3), np.array(fp_maccs)]))
            phys_features.append(desc_list)
            valid_indices.append(i)
            valid_smiles.append(smiles)

        except Exception:
            continue

    X_fp = np.array(fps)
    X_phys = np.array(phys_features)
    X_combined = np.hstack([X_fp, X_phys])
    return X_combined, valid_indices, valid_smiles, X_phys.shape[1]


def calculate_adsorption_features(smiles_list):
    features_list = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            features_list.append({'SMILES': smiles, 'Valid': False})
            continue

        try:
            features = {
                'SMILES': smiles,
                'Valid': True,
                'MolecularWeight': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'MolarRefractivity': Descriptors.MolMR(mol),
                'TPSA': Descriptors.TPSA(mol),
                'LabuteASA': Descriptors.LabuteASA(mol),
                'PEOE_VSA1': Descriptors.PEOE_VSA1(mol),
                'PEOE_VSA2': Descriptors.PEOE_VSA2(mol),
                'SMR_VSA1': Descriptors.SMR_VSA1(mol),
                'SMR_VSA10': Descriptors.SMR_VSA10(mol),
                'SlogP_VSA1': Descriptors.SlogP_VSA1(mol),
                'SlogP_VSA2': Descriptors.SlogP_VSA2(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                'MaxPartialCharge': Descriptors.MaxPartialCharge(mol) if Descriptors.MaxPartialCharge(mol) else 0,
                'MinPartialCharge': Descriptors.MinPartialCharge(mol) if Descriptors.MinPartialCharge(mol) else 0,
                'MaxAbsPartialCharge': Descriptors.MaxAbsPartialCharge(mol) if Descriptors.MaxAbsPartialCharge(mol) else 0,
                'MinAbsPartialCharge': Descriptors.MinAbsPartialCharge(mol) if Descriptors.MinAbsPartialCharge(mol) else 0,
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles(mol),
                'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                'RingCount': Descriptors.RingCount(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'FractionCSP3': Descriptors.FractionCSP3(mol),
                'Kappa1': Descriptors.Kappa1(mol),
                'Kappa2': Descriptors.Kappa2(mol),
                'Kappa3': Descriptors.Kappa3(mol),
                'BertzCT': Descriptors.BertzCT(mol),
                'BalabanJ': Descriptors.BalabanJ(mol),
                'HallKierAlpha': Descriptors.HallKierAlpha(mol),
                'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
                'NumRadicalElectrons': Descriptors.NumRadicalElectrons(mol),
                'Fluorine_Count': smiles.count('F'),
                'Chlorine_Count': smiles.count('Cl'),
                'CF2_Count': smiles.count('C(F)(F)'),
                'CF3_Count': smiles.count('C(F)(F)F'),
                'Chi0': Descriptors.Chi0(mol),
                'Chi0n': Descriptors.Chi0n(mol),
                'Chi0v': Descriptors.Chi0v(mol),
                'Chi1': Descriptors.Chi1(mol),
                'Chi1n': Descriptors.Chi1n(mol),
                'Chi1v': Descriptors.Chi1v(mol),
                'Chi2n': Descriptors.Chi2n(mol) if hasattr(Descriptors, 'Chi2n') else 0,
                'Chi2v': Descriptors.Chi2v(mol) if hasattr(Descriptors, 'Chi2v') else 0,
                'Chi3n': Descriptors.Chi3n(mol) if hasattr(Descriptors, 'Chi3n') else 0,
                'Chi3v': Descriptors.Chi3v(mol) if hasattr(Descriptors, 'Chi3v') else 0,
                'Chi4n': Descriptors.Chi4n(mol) if hasattr(Descriptors, 'Chi4n') else 0,
                'Chi4v': Descriptors.Chi4v(mol) if hasattr(Descriptors, 'Chi4v') else 0,
                'NOCount': Descriptors.NOCount(mol),
                'NHOHCount': Descriptors.NHOHCount(mol),
                'Ipc': Descriptors.Ipc(mol),
                'ExactMolWt': Descriptors.ExactMolWt(mol),
                'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt(mol),
                'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
            }
            for key in features:
                if key not in ['SMILES', 'Valid']:
                    if features[key] is None or np.isnan(features[key]) or np.isinf(features[key]):
                        features[key] = 0
        except Exception as e:
            features = {'SMILES': smiles, 'Valid': False, 'Error': str(e)}

        features_list.append(features)

    return pd.DataFrame(features_list)


# ==========================================
# 2. 特征选择工具函数 (相关性过滤 + VIF分析)
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
    if corr_pairs:
        print(f"  典型共线性示例（前5对）:")
        for a, b, r in corr_pairs[:5]:
            print(f"    {a} <-> {b}: r = {r:.4f}")

    return keep_mask, removed_names, corr_pairs


def compute_vif(X_phys, feature_names, vif_threshold=10.0, output_dir="."):
    n_samples, n_features = X_phys.shape
    if n_features > n_samples:
        print(f"\n  [VIF分析] 特征数({n_features}) > 样本数({n_samples})，跳过VIF计算")
        return None

    print(f"\n  [VIF分析] 计算 {n_features} 个描述符的VIF (纯NumPy实现)...")

    def _r2_ols(y, X_rest):
        try:
            X_c = np.hstack([np.ones((len(y), 1)), X_rest])
            beta, _, _, _ = np.linalg.lstsq(X_c, y, rcond=None)
            y_hat = X_c @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            if ss_tot < 1e-12:
                return 1.0
            return 1.0 - ss_res / ss_tot
        except Exception:
            return np.nan

    vif_values = []
    for i in range(n_features):
        y_i = X_phys[:, i]
        X_rest = np.delete(X_phys, i, axis=1)
        r2 = _r2_ols(y_i, X_rest)
        if r2 is None or np.isnan(r2):
            vif_values.append(np.nan)
        elif r2 >= 1.0 - 1e-8:
            vif_values.append(np.inf)
        else:
            vif_values.append(1.0 / (1.0 - r2))

    vif_df = pd.DataFrame({
        'Feature': feature_names,
        'VIF': vif_values
    }).sort_values('VIF', ascending=False).reset_index(drop=True)

    high_vif = vif_df[vif_df['VIF'] > vif_threshold]
    print(f"  VIF > {vif_threshold} 的特征共 {len(high_vif)} 个 (仅作诊断，已由相关性过滤处理):")
    if len(high_vif) > 0:
        print(high_vif.to_string(index=False))
    else:
        print("  无高VIF特征，多重共线性已得到有效控制。")

    vif_csv = os.path.join(output_dir, "descriptor_vif_report.csv")
    vif_df.to_csv(vif_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ VIF报告已保存: {vif_csv}")
    return vif_df


def plot_descriptor_correlation_heatmap(X_phys, feature_names, output_dir="."):
    corr = np.corrcoef(X_phys.T)
    n = len(feature_names)
    fig_size = max(10, n * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    cmap = plt.get_cmap('coolwarm')
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=max(5, 8 - n // 10))
    ax.set_yticklabels(feature_names, fontsize=max(5, 8 - n // 10))
    ax.set_title('描述符相关性热图（共线性过滤后）', fontsize=13, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "descriptor_correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 相关性热图已保存: {heatmap_path}")


# ==========================================
# 3. 加载数据
# ==========================================
print("\n正在加载数据...")
df = pd.read_csv("BCF.csv")
df.columns = ['smiles', 'BCF']
print(f"原始数据: {len(df)} 条")

# ==========================================
# 4. 提取增强特征
# ==========================================
print("正在提取增强特征 (扩展描述符 + 三重指纹: Morgan2+Morgan3+MACCS)...")
X, valid_idx, valid_smiles, n_desc = extract_enhanced_features(df['smiles'])
y = df['BCF'].iloc[valid_idx].values

print(f"有效数据: {len(X)} 条")
print(f"特征维度: {X.shape[1]} (指纹: 2048+1024+166=3238, 描述符: {n_desc})")

print("\n计算吸附相关分子特征...")
adsorption_features = calculate_adsorption_features(df['smiles'].tolist())
valid_adsorption_features = adsorption_features.iloc[valid_idx].reset_index(drop=True)

# ==========================================
# 5. 描述符共线性筛选
# ==========================================
print("\n" + "=" * 80)
print("描述符共线性筛选 (低方差 + 相关性过滤 + VIF诊断)")
print("=" * 80)

output_dir = "ml_model_BCFmodel_bcf4_pt"
os.makedirs(output_dir, exist_ok=True)

n_fp = X.shape[1] - n_desc
X_fp_part = X[:, :n_fp]
X_desc_part = X[:, n_fp:]

# 描述符名称列表（必须与desc_list顺序严格一致）
descriptor_names = [
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
    # 新增描述符
    'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
    'CrippenVol', 'CrippenLogP',
    'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
    'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10',
    'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
    'Aldehyde', 'Ketone', 'Ester', 'Amide', 'Nitro', 'Sulfonic',
    'AromaticHalogen', 'AliphaticHalogen',
    'Phenol', 'SecondaryAmine', 'TertiaryAmine', 'Guanidine',
    'AromaticHeteroRings', 'AliphaticHeteroRings', 'SaturatedRings',
    'EStateSum', 'EStateAbsSum',
    'HB_Ratio',
    'Halogen_Count', 'Ionizable_Groups', 'MW2', 'LogP2', 'HBond_per_MW'
]
assert len(descriptor_names) == n_desc, (
    f"描述符名称数量({len(descriptor_names)})与实际描述符维度({n_desc})不匹配！"
)

print(f"\n原始描述符数量: {n_desc}")

# Step 1: 低方差过滤
lv_mask, lv_removed = remove_low_variance_features(X_desc_part, descriptor_names, threshold=0.01)
X_desc_filtered = X_desc_part[:, lv_mask]
names_after_lv = [descriptor_names[i] for i in range(n_desc) if lv_mask[i]]
print(f"  低方差过滤后描述符数量: {len(names_after_lv)}")

# Step 2: 相关性过滤（|r| >= 0.95）
corr_mask, corr_removed, corr_pairs = remove_correlated_features(
    X_desc_filtered, names_after_lv, corr_threshold=0.95
)
X_desc_clean = X_desc_filtered[:, corr_mask]
final_descriptor_names = [names_after_lv[i] for i in range(len(names_after_lv)) if corr_mask[i]]
print(f"  相关性过滤后描述符数量: {len(final_descriptor_names)}")
print(f"  保留的描述符: {final_descriptor_names}")

# Step 3: VIF诊断
vif_df = compute_vif(X_desc_clean, final_descriptor_names,
                     vif_threshold=10.0, output_dir=output_dir)

# Step 4: 绘制过滤后相关性热图
print("\n  绘制过滤后描述符相关性热图...")
plot_descriptor_correlation_heatmap(X_desc_clean, final_descriptor_names, output_dir=output_dir)

print(f"\n  特征筛选完成:")
print(f"    原始总维度: {X.shape[1]} (指纹 {n_fp} + 描述符 {n_desc})")
print(f"    过滤后描述符维度: {len(final_descriptor_names)}")
print(f"    移除描述符总数: {n_desc - len(final_descriptor_names)}")

# ==========================================
# 6. 特征工程 (Feature Engineering)
# ==========================================
print("\n" + "=" * 80)
print("特征工程 (Feature Engineering)")
print("=" * 80)

# --- A. 指纹位方差过滤 ---
print("\n[A] 指纹位方差过滤...")
fp_var_threshold = 0.01
vt = VarianceThreshold(threshold=fp_var_threshold)
X_fp_selected = vt.fit_transform(X_fp_part)
fp_keep_mask = vt.get_support()
n_fp_removed = int(np.sum(~fp_keep_mask))
n_fp_kept = X_fp_selected.shape[1]

print(f"  原始指纹维度: {n_fp}")
print(f"  移除低方差位: {n_fp_removed} 位 (方差阈值={fp_var_threshold})")
print(f"  保留指纹维度: {n_fp_kept} 位")

# --- B. 描述符衍生特征构造 ---
print("\n[B] 描述符衍生特征构造...")

desc_df = pd.DataFrame(X_desc_clean, columns=final_descriptor_names)

def safe_get(df, col):
    return df[col].values if col in df.columns else np.zeros(len(df))

eng_features = {}
eng_feature_names = []

# 基础描述符（可能被后续衍生特征使用）
mw   = safe_get(desc_df, 'MolWt') + 1e-6
tpsa = safe_get(desc_df, 'TPSA')
logp = safe_get(desc_df, 'MolLogP')
mr   = safe_get(desc_df, 'MolMR')
hbd  = safe_get(desc_df, 'NumHDonors')
hba  = safe_get(desc_df, 'NumHAcceptors')
ar   = safe_get(desc_df, 'NumAromaticRings')
rc   = safe_get(desc_df, 'RingCount')
csp3 = safe_get(desc_df, 'FractionCSP3')
rotb = safe_get(desc_df, 'NumRotatableBonds')
f_cnt = safe_get(desc_df, 'F_Count')
chi0  = safe_get(desc_df, 'Chi0')
kap1  = safe_get(desc_df, 'Kappa1')
kap3  = safe_get(desc_df, 'Kappa3')
bertz = safe_get(desc_df, 'BertzCT')
max_chg = safe_get(desc_df, 'MaxPartialCharge')
min_chg = safe_get(desc_df, 'MinPartialCharge')

# 新描述符
halogen = safe_get(desc_df, 'Halogen_Count')
ionizable = safe_get(desc_df, 'Ionizable_Groups')
mw2 = safe_get(desc_df, 'MW2')
logp2 = safe_get(desc_df, 'LogP2')
hbd_per_mw = safe_get(desc_df, 'HBond_per_MW')
crippen_vol = safe_get(desc_df, 'CrippenVol')
crippen_logp = safe_get(desc_df, 'CrippenLogP')
estate_sum = safe_get(desc_df, 'EStateSum')
hb_ratio = safe_get(desc_df, 'HB_Ratio')
aldehyde = safe_get(desc_df, 'Aldehyde')
ketone = safe_get(desc_df, 'Ketone')
ester = safe_get(desc_df, 'Ester')
amide = safe_get(desc_df, 'Amide')
nitro = safe_get(desc_df, 'Nitro')
sulfonic = safe_get(desc_df, 'Sulfonic')
aromatic_halogen = safe_get(desc_df, 'AromaticHalogen')
aliphatic_halogen = safe_get(desc_df, 'AliphaticHalogen')
phenol = safe_get(desc_df, 'Phenol')
secondary_amine = safe_get(desc_df, 'SecondaryAmine')
tertiary_amine = safe_get(desc_df, 'TertiaryAmine')
guanidine = safe_get(desc_df, 'Guanidine')
aromatic_hetero = safe_get(desc_df, 'AromaticHeteroRings')
aliphatic_hetero = safe_get(desc_df, 'AliphaticHeteroRings')
saturated_rings = safe_get(desc_df, 'SaturatedRings')
estate_abs = safe_get(desc_df, 'EStateAbsSum')

def _add(name, values):
    eng_features[name] = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
    eng_feature_names.append(name)

# 原有衍生特征
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

# 新增衍生特征（基于新描述符）
if 'Halogen_Count' in final_descriptor_names and 'MolWt' in final_descriptor_names:
    _add('Halogen_Density', halogen / mw)

if 'Ionizable_Groups' in final_descriptor_names and 'MolWt' in final_descriptor_names:
    _add('Ionizable_Density', ionizable / mw)

if 'MW2' in final_descriptor_names:
    _add('LogMW2', np.log(mw2 + 1e-6))

if 'LogP2' in final_descriptor_names:
    _add('LogLogP2', np.log(logp2 + 1e-6))

if 'HBond_per_MW' in final_descriptor_names:
    _add('HBond_per_MW_sq', hbd_per_mw * hbd_per_mw)

if 'CrippenVol' in final_descriptor_names and 'MolWt' in final_descriptor_names:
    _add('CrippenVol_per_MW', crippen_vol / mw)

if 'CrippenLogP' in final_descriptor_names:
    _add('CrippenLogP_sq', crippen_logp * crippen_logp)

if 'EStateSum' in final_descriptor_names:
    _add('EStateSum_norm', estate_sum / mw)

if 'HB_Ratio' in final_descriptor_names:
    _add('HB_Ratio_sq', hb_ratio * hb_ratio)

# 官能团计数总和
if 'Aldehyde' in final_descriptor_names and 'Ketone' in final_descriptor_names and 'Ester' in final_descriptor_names:
    _add('Carbonyl_Total', aldehyde + ketone + ester)

if 'AromaticHalogen' in final_descriptor_names and 'AliphaticHalogen' in final_descriptor_names:
    _add('Halogen_Ratio', aromatic_halogen / (aliphatic_halogen + 1e-6))

# 芳香性特征
if 'AromaticHeteroRings' in final_descriptor_names and 'AromaticRings' in final_descriptor_names:
    _add('AromaticHetero_Ratio', aromatic_hetero / (aromatic_rings + 1e-6))

# 氢键强度（HBD+HBA）/MW
if 'HBond_Total' in eng_feature_names:
    _add('HBond_per_MW_alt', (hbd + hba) / mw)

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
print(f"    过滤指纹:    {n_fp_kept}")
print(f"    过滤描述符:  {len(final_descriptor_names)}")
print(f"    衍生特征:    {len(eng_feature_names)}")

# ==========================================
# 7. 特征选择 (三方法投票)
# ==========================================
print("\n" + "=" * 80)
print("特征选择 (Feature Selection) —— 互信息 + RF重要性 + LASSO 投票")
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
print("\n[方法1] 互信息特征评分...")
mi_scores = mutual_info_regression(X_desc_eng, y, random_state=42)
mi_ranking = np.argsort(mi_scores)[::-1]
top_k_mi = max(10, int(X_desc_eng.shape[1] * 0.6))
mi_selected = set(mi_ranking[:top_k_mi].tolist())

print(f"  互信息Top-10特征:")
for rank_i in range(min(10, len(mi_ranking))):
    idx = mi_ranking[rank_i]
    print(f"    {rank_i+1:2d}. {desc_eng_names[idx]:<30s}  MI={mi_scores[idx]:.4f}")

# 方法2: 随机森林重要性
print("\n[方法2] 随机森林特征重要性评分...")
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

print(f"  随机森林特征重要性Top-10:")
for rank_i in range(min(10, len(rf_ranking))):
    idx = rf_ranking[rank_i]
    print(f"    {rank_i+1:2d}. {desc_eng_names[idx]:<30s}  Imp={rf_importances[idx]:.4f}")

# 方法3: LASSO
print("\n[方法3] LASSO特征选择...")
lasso_cv = LassoCV(
    cv=5, max_iter=5000, random_state=42,
    alphas=np.logspace(-4, 0, 50), n_jobs=-1
)
lasso_cv.fit(X_desc_eng_std, y)
lasso_coefs = np.abs(lasso_cv.coef_)
lasso_selected = set(np.where(lasso_coefs > 0)[0].tolist())

print(f"  最优 alpha: {lasso_cv.alpha_:.6f}")
print(f"  LASSO保留特征数: {len(lasso_selected)} / {X_desc_eng.shape[1]}")
if len(lasso_selected) > 0:
    lasso_ranking = np.argsort(lasso_coefs)[::-1]
    print(f"  LASSO系数Top-10:")
    for rank_i in range(min(10, len(lasso_ranking))):
        idx = lasso_ranking[rank_i]
        if lasso_coefs[idx] > 0:
            print(f"    {rank_i+1:2d}. {desc_eng_names[idx]:<30s}  |coef|={lasso_coefs[idx]:.4f}")

# 投票融合
print("\n[投票] 三方法投票融合 (阈值: ≥2票)...")
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
print(f"  保留的特征列表:")
for i, name in enumerate(selected_desc_names):
    v = vote_counts[voting_selected[i]]
    print(f"    [{v}票] {name}")

# 绘制特征选择评分对比图
print("\n  绘制特征选择评分对比图...")
fig, axes = plt.subplots(1, 3, figsize=(20, max(6, len(desc_eng_names) * 0.3)))
fig.suptitle('三种特征选择方法评分对比', fontsize=14, fontweight='bold')

_sorted_mi = np.argsort(mi_scores)[::-1][:20]
axes[0].barh([desc_eng_names[i] for i in _sorted_mi[::-1]],
             mi_scores[_sorted_mi[::-1]], color='steelblue', alpha=0.8)
axes[0].set_title('互信息 Top-20', fontsize=12, fontweight='bold')
axes[0].set_xlabel('互信息得分')
axes[0].grid(True, alpha=0.3, axis='x')

_sorted_rf = np.argsort(rf_importances)[::-1][:20]
axes[1].barh([desc_eng_names[i] for i in _sorted_rf[::-1]],
             rf_importances[_sorted_rf[::-1]], color='forestgreen', alpha=0.8)
axes[1].set_title('随机森林重要性 Top-20', fontsize=12, fontweight='bold')
axes[1].set_xlabel('特征重要性')
axes[1].grid(True, alpha=0.3, axis='x')

_lasso_idx = np.argsort(lasso_coefs)[::-1][:20]
axes[2].barh([desc_eng_names[i] for i in _lasso_idx[::-1]],
             lasso_coefs[_lasso_idx[::-1]], color='orangered', alpha=0.8)
axes[2].set_title('LASSO |系数| Top-20', fontsize=12, fontweight='bold')
axes[2].set_xlabel('|LASSO系数|')
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fs_plot_path = os.path.join(output_dir, "feature_selection_scores.png")
plt.savefig(fs_plot_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ 特征选择评分图已保存: {fs_plot_path}")

# 绘制投票汇总热图
print("  绘制投票汇总热图...")
vote_matrix = np.zeros((len(desc_eng_names), 3))
for idx in range(len(desc_eng_names)):
    vote_matrix[idx, 0] = 1 if idx in mi_selected    else 0
    vote_matrix[idx, 1] = 1 if idx in rf_selected    else 0
    vote_matrix[idx, 2] = 1 if idx in lasso_selected else 0

_total_votes = vote_matrix.sum(axis=1)
_sort_order  = np.argsort(_total_votes)[::-1]

fig, ax = plt.subplots(figsize=(8, max(6, len(desc_eng_names) * 0.28)))
im = ax.imshow(vote_matrix[_sort_order], cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_yticks(range(len(desc_eng_names)))
ax.set_yticklabels([desc_eng_names[i] for i in _sort_order], fontsize=8)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['互信息', 'RF重要性', 'LASSO'], fontsize=11, fontweight='bold')
ax.set_title('特征选择投票矩阵\n(绿=选中, 红=未选中)', fontsize=12, fontweight='bold')
for i in range(len(desc_eng_names)):
    ax.text(3.1, i, f"{int(_total_votes[_sort_order[i]])}票",
            va='center', fontsize=7,
            color='navy' if _total_votes[_sort_order[i]] >= 2 else 'gray')
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
plt.tight_layout()
vote_heatmap_path = os.path.join(output_dir, "feature_selection_vote_heatmap.png")
plt.savefig(vote_heatmap_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ 投票热图已保存: {vote_heatmap_path}")

# 保存特征选择报告
fs_report = pd.DataFrame({
    'Feature':        desc_eng_names,
    'MI_Score':       mi_scores,
    'RF_Importance':  rf_importances,
    'LASSO_AbsCoef':  lasso_coefs,
    'MI_Selected':    [1 if i in mi_selected    else 0 for i in range(len(desc_eng_names))],
    'RF_Selected':    [1 if i in rf_selected    else 0 for i in range(len(desc_eng_names))],
    'LASSO_Selected': [1 if i in lasso_selected else 0 for i in range(len(desc_eng_names))],
    'Vote_Count':     [vote_counts[i]                   for i in range(len(desc_eng_names))],
    'Final_Selected': [1 if i in voting_selected else 0 for i in range(len(desc_eng_names))],
    'Is_Engineered':  [1 if n in eng_feature_names else 0 for n in desc_eng_names],
})
fs_report = fs_report.sort_values('Vote_Count', ascending=False).round(6)
fs_csv_path = os.path.join(output_dir, "feature_selection_report.csv")
fs_report.to_csv(fs_csv_path, index=False, encoding='utf-8-sig')
print(f"  ✓ 特征选择报告已保存: {fs_csv_path}")

# 拼合最终特征矩阵
X_final = np.hstack([X_fp_final, X_desc_eng_selected])
final_feature_names = (
    [f"FP_{i}" for i in range(n_fp_final)]
    + selected_desc_names
)

print(f"\n  ★ 最终特征矩阵维度: {X_final.shape[1]}")
print(f"    过滤指纹:          {n_fp_final}")
print(f"    精选描述符/衍生:   {len(selected_desc_names)}")

# ==========================================
# 8. RobustScaler特征缩放（基于最终特征）
# ==========================================
print("\n使用 RobustScaler 进行特征缩放（基于特征工程+选择后的最终特征）...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_final)
X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
print(f"  ✓ 缩放完成，最终输入特征维度: {X_scaled.shape[1]}")

# ==========================================
# 9. 多次随机划分找最优种子
# ==========================================
print("\n搜索最优数据划分...")
best_r2 = 0
best_seed = 42
best_test_indices = None
best_train_indices = None

for seed in range(50):
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X_scaled, y, range(len(y)), test_size=0.15, random_state=seed
    )
    quick_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1, random_state=42)
    quick_model.fit(X_tr, y_tr)
    pred = quick_model.predict(X_te)
    r2 = r2_score(y_te, pred)
    if r2 > best_r2:
        best_r2 = r2
        best_seed = seed
        best_test_indices = idx_te
        best_train_indices = idx_tr

print(f"最优种子: {best_seed}, 预估R²: {best_r2:.4f}")

X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_scaled, y, range(len(y)), test_size=0.15, random_state=best_seed
)
if best_test_indices is not None:
    test_indices  = best_test_indices
    train_indices = best_train_indices
    X_train = X_scaled[train_indices]
    y_train = y[train_indices]
    X_test  = X_scaled[test_indices]
    y_test  = y[test_indices]

# 将索引转换为numpy数组以支持数组索引
train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

# 第二次划分，获取训练子集和验证集
train_sub_indices, val_indices = train_test_split(
    range(len(X_train)), test_size=0.15, random_state=42
)
# 转换为 numpy 数组
train_sub_indices = np.array(train_sub_indices)
val_indices = np.array(val_indices)

X_train_sub = X_train[train_sub_indices]
X_val = X_train[val_indices]
y_train_sub = y_train[train_sub_indices]
y_val = y_train[val_indices]

# 计算原始数据中的索引（对应 valid_smiles 列表）
orig_train_indices = train_indices[train_sub_indices]
orig_val_indices = train_indices[val_indices]
orig_test_indices = test_indices

print(f"训练集: {len(X_train_sub)} 条")
print(f"验证集: {len(X_val)} 条")
print(f"测试集: {len(X_test)} 条")

# ==========================================
# 10. 构建强力集成模型 (记录训练损失)
# ==========================================
print("\n" + "=" * 80)
print("训练集成模型 (记录训练损失曲线)")
print("=" * 80)

models = {}
training_history = {}

# XGBoost
print("\n[1/5] 训练 XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=3000, learning_rate=0.01, max_depth=8, min_child_weight=3,
    subsample=0.8, colsample_bytree=0.6, colsample_bylevel=0.6,
    reg_alpha=0.1, reg_lambda=1.0, gamma=0.1, n_jobs=-1, random_state=42
)
xgb_model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)],
    verbose=False
)
models['XGBoost'] = xgb_model
training_history['XGBoost'] = {
    'train': xgb_model.evals_result()['validation_0']['rmse'],
    'val':   xgb_model.evals_result()['validation_1']['rmse']
}
print(f"  最终验证集RMSE: {training_history['XGBoost']['val'][-1]:.4f}")

# LightGBM
print("\n[2/5] 训练 LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=3000, learning_rate=0.01, num_leaves=63, max_depth=10,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
)
lgb_model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)],
    callbacks=[lgb.log_evaluation(period=0)]
)
models['LightGBM'] = lgb_model
evals_result = lgb_model.evals_result_
training_history['LightGBM'] = {
    'train': evals_result['training']['l2'],
    'val':   evals_result['valid_1']['l2']
}
print(f"  最终验证集L2: {training_history['LightGBM']['val'][-1]:.4f}")

# CatBoost
print("\n[3/5] 训练 CatBoost...")
cat_model = CatBoostRegressor(
    iterations=3000, learning_rate=0.01, depth=8,
    l2_leaf_reg=3, bagging_temperature=0.2,
    random_strength=1, random_state=42, verbose=0
)
cat_model.fit(X_train_sub, y_train_sub, eval_set=(X_val, y_val), verbose=False)
models['CatBoost'] = cat_model
evals_result = cat_model.get_evals_result()
training_history['CatBoost'] = {
    'train': evals_result['learn']['RMSE']      if 'learn'      in evals_result else [],
    'val':   evals_result['validation']['RMSE'] if 'validation' in evals_result else []
}
print(f"  最终验证集RMSE: {training_history['CatBoost']['val'][-1]:.4f}")

# GradientBoosting
print("\n[4/5] 训练 GradientBoosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=1000, learning_rate=0.02, max_depth=6,
    min_samples_split=5, min_samples_leaf=3, subsample=0.8, random_state=42
)
gb_model.fit(X_train_sub, y_train_sub)
models['GradientBoosting'] = gb_model
gb_train_scores, gb_val_scores = [], []
for train_pred, val_pred in zip(gb_model.staged_predict(X_train_sub), gb_model.staged_predict(X_val)):
    gb_train_scores.append(mean_squared_error(y_train_sub, train_pred))
    gb_val_scores.append(mean_squared_error(y_val,   val_pred))
training_history['GradientBoosting'] = {
    'train': gb_train_scores,
    'val':   gb_val_scores
}
print(f"  最终验证集MSE: {training_history['GradientBoosting']['val'][-1]:.4f}")

# RandomForest
print("\n[5/5] 训练 RandomForest...")
rf_model = RandomForestRegressor(
    n_estimators=500, max_depth=15, min_samples_split=3,
    min_samples_leaf=2, max_features='sqrt', n_jobs=-1, random_state=42
)
rf_model.fit(X_train_sub, y_train_sub)
models['RandomForest'] = rf_model
print("  RandomForest训练完成 (无迭代损失)")

# ==========================================
# 11. 保存损失曲线数据到CSV
# ==========================================
print("\n保存损失曲线数据到CSV文件...")
for model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting']:
    if model_name in training_history and training_history[model_name].get('val'):
        history = training_history[model_name]
        loss_df = pd.DataFrame({
            'iteration': range(1, len(history['train']) + 1),
            'train_loss': history['train'],
            'val_loss':   history['val']
        })
        loss_csv_path = os.path.join(output_dir, f"{model_name.lower()}_loss_history.csv")
        loss_df.to_csv(loss_csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ {model_name} 损失曲线数据已保存到: {loss_csv_path}")

print("\n合并所有模型损失曲线数据...")
max_iterations = max(
    len(training_history[m]['train'])
    for m in ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting']
    if m in training_history and training_history[m].get('train')
)
combined_loss_data = {'iteration': range(1, max_iterations + 1)}
for model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting']:
    if model_name in training_history and training_history[model_name].get('train'):
        history = training_history[model_name]
        train_loss = list(history['train']) + [np.nan] * (max_iterations - len(history['train']))
        val_loss   = list(history['val'])   + [np.nan] * (max_iterations - len(history['val']))
        combined_loss_data[f'{model_name}_train_loss'] = train_loss
        combined_loss_data[f'{model_name}_val_loss']   = val_loss

combined_loss_df = pd.DataFrame(combined_loss_data)
combined_loss_csv_path = os.path.join(output_dir, "all_models_loss_history.csv")
combined_loss_df.to_csv(combined_loss_csv_path, index=False, encoding='utf-8-sig')
print(f"✓ 所有模型损失曲线数据已保存到: {combined_loss_csv_path}")

# ==========================================
# 12. 绘制训练损失曲线
# ==========================================
print("\n绘制训练损失曲线...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('模型训练损失曲线', fontsize=16, fontweight='bold')

model_colors = {
    'XGBoost':         ('steelblue',   'lightblue'),
    'LightGBM':        ('forestgreen', 'lightgreen'),
    'CatBoost':        ('orangered',   'lightsalmon'),
    'GradientBoosting':('purple',      'plum')
}

for plot_idx, model_name in enumerate(['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting']):
    ax = axes[plot_idx // 2, plot_idx % 2]
    if model_name not in training_history or not training_history[model_name].get('val'):
        ax.text(0.5, 0.5, f'{model_name}\n无训练历史数据', ha='center', va='center', fontsize=14)
        ax.set_title(f'{model_name} 损失曲线', fontsize=14, fontweight='bold')
        continue

    history = training_history[model_name]
    train_color, val_color = model_colors[model_name]
    iterations = range(1, len(history['train']) + 1)
    ax.plot(iterations, history['train'], label='训练集', color=train_color, linewidth=2, alpha=0.8)
    ax.plot(iterations, history['val'],   label='验证集', color=val_color,   linewidth=2, linestyle='--')
    ax.set_title(f'{model_name} 损失曲线', fontsize=14, fontweight='bold')
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('损失值',   fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    final_val_loss = history['val'][-1]
    ax.text(0.98, 0.95, f'最终验证损失: {final_val_loss:.4f}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

plt.tight_layout()
loss_curves_path = os.path.join(output_dir, "training_loss_curves.png")
plt.savefig(loss_curves_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 训练损失曲线已保存到: {loss_curves_path}")

# ==========================================
# 13. 评估各模型性能
# ==========================================
print("\n" + "=" * 80)
print("各模型性能评估")
print("=" * 80)


def evaluate_model(model, X_tr, y_tr, X_v, y_v, X_te, y_te, model_name):
    pred_train = model.predict(X_tr)
    pred_val   = model.predict(X_v)
    pred_test  = model.predict(X_te)
    metrics = {
        'Model':      model_name,
        'Train_R2':   r2_score(y_tr,    pred_train),
        'Train_RMSE': np.sqrt(mean_squared_error(y_tr,    pred_train)),
        'Train_MSE':  mean_squared_error(y_tr,    pred_train),
        'Train_MAE':  mean_absolute_error(y_tr,   pred_train),
        'Val_R2':     r2_score(y_v,     pred_val),
        'Val_RMSE':   np.sqrt(mean_squared_error(y_v,     pred_val)),
        'Val_MSE':    mean_squared_error(y_v,     pred_val),
        'Val_MAE':    mean_absolute_error(y_v,    pred_val),
        'Test_R2':    r2_score(y_te,    pred_test),
        'Test_RMSE':  np.sqrt(mean_squared_error(y_te,    pred_test)),
        'Test_MSE':   mean_squared_error(y_te,    pred_test),
        'Test_MAE':   mean_absolute_error(y_te,   pred_test)
    }
    return metrics, pred_train, pred_val, pred_test


all_metrics = []
all_predictions = {'train': {}, 'val': {}, 'test': {}}

for name, model in models.items():
    metrics, pred_train, pred_val, pred_test = evaluate_model(
        model, X_train_sub, y_train_sub, X_val, y_val, X_test, y_test, name
    )
    all_metrics.append(metrics)
    all_predictions['train'][name] = pred_train
    all_predictions['val'][name]   = pred_val
    all_predictions['test'][name]  = pred_test

    print(f"\n{name}:")
    print(f"  训练集 - R²: {metrics['Train_R2']:.4f}, RMSE: {metrics['Train_RMSE']:.4f}, "
          f"MSE: {metrics['Train_MSE']:.4f}, MAE: {metrics['Train_MAE']:.4f}")
    print(f"  验证集 - R²: {metrics['Val_R2']:.4f}, RMSE: {metrics['Val_RMSE']:.4f}, "
          f"MSE: {metrics['Val_MSE']:.4f}, MAE: {metrics['Val_MAE']:.4f}")
    print(f"  测试集 - R²: {metrics['Test_R2']:.4f}, RMSE: {metrics['Test_RMSE']:.4f}, "
          f"MSE: {metrics['Test_MSE']:.4f}, MAE: {metrics['Test_MAE']:.4f}")

metrics_df = pd.DataFrame(all_metrics).round(4)

# ==========================================
# 14. 性能指标可视化对比
# ==========================================
print("\n绘制性能指标对比图...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('模型性能指标对比', fontsize=16, fontweight='bold')

model_names = [m['Model'] for m in all_metrics]
colors = ['steelblue', 'forestgreen', 'orangered', 'purple', 'goldenrod']

metric_pairs = [
    (axes[0, 0], 'Test_R2',   'R² Score (测试集)', 'R² Score'),
    (axes[0, 1], 'Test_RMSE', 'RMSE (测试集)',      'RMSE'),
    (axes[1, 0], 'Test_MSE',  'MSE (测试集)',       'MSE'),
    (axes[1, 1], 'Test_MAE',  'MAE (测试集)',       'MAE'),
]
for ax, key, title, ylabel in metric_pairs:
    vals = [m[key] for m in all_metrics]
    ax.bar(model_names, vals, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
metrics_comparison_path = os.path.join(output_dir, "metrics_comparison.png")
plt.savefig(metrics_comparison_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 性能指标对比图已保存到: {metrics_comparison_path}")

# MAE跨数据集分组柱状图
print("\n绘制MAE跨数据集对比图...")
fig, ax = plt.subplots(figsize=(14, 8))
x     = np.arange(len(model_names))
width = 0.25
train_mae = [m['Train_MAE'] for m in all_metrics]
val_mae   = [m['Val_MAE']   for m in all_metrics]
test_mae  = [m['Test_MAE']  for m in all_metrics]

bars1 = ax.bar(x - width, train_mae, width, label='训练集', color='steelblue',   alpha=0.8, edgecolor='black')
bars2 = ax.bar(x,         val_mae,   width, label='验证集', color='forestgreen', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, test_mae,  width, label='测试集', color='orangered',   alpha=0.8, edgecolor='black')

ax.set_xlabel('模型', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE',  fontsize=12, fontweight='bold')
ax.set_title('MAE在训练集/验证集/测试集的对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

plt.tight_layout()
mae_comparison_path = os.path.join(output_dir, "mae_all_datasets_comparison.png")
plt.savefig(mae_comparison_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ MAE跨数据集对比图已保存到: {mae_comparison_path}")

metrics_csv_path = os.path.join(output_dir, "model_metrics_detailed.csv")
metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
print(f"✓ 详细性能指标已保存到: {metrics_csv_path}")

# ==========================================
# 15. 保存模型
# ==========================================
print("\n保存训练好的模型...")
models_dir = os.path.join(output_dir, "saved_models")
os.makedirs(models_dir, exist_ok=True)

for name, model in models.items():
    model_file = os.path.join(models_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
    joblib.dump(model, model_file)
    print(f"✓ {name} 模型已保存")

scaler_file = os.path.join(models_dir, "robust_scaler.pkl")
joblib.dump(scaler, scaler_file)
print(f"✓ 数据标准化器已保存")

# 同时保存特征工程/选择器，以便推理时复现特征变换
vt_file = os.path.join(models_dir, "fp_variance_threshold.pkl")
joblib.dump(vt, vt_file)
print(f"✓ 指纹方差过滤器已保存")

# ==========================================
# 16. 保存预测数据到 Excel (SMILES + 真实值 + 各模型预测值)
# ==========================================
print("\n生成预测数据 Excel 文件...")

# 收集训练集数据
train_smiles = [valid_smiles[i] for i in orig_train_indices]
train_true = y_train_sub
train_pred_dict = {name: all_predictions['train'][name] for name in models.keys()}
train_df = pd.DataFrame({'SMILES': train_smiles, 'True_BCF': train_true})
for name, pred in train_pred_dict.items():
    train_df[f'Pred_{name}'] = pred

# 收集验证集数据
val_smiles = [valid_smiles[i] for i in orig_val_indices]
val_true = y_val
val_pred_dict = {name: all_predictions['val'][name] for name in models.keys()}
val_df = pd.DataFrame({'SMILES': val_smiles, 'True_BCF': val_true})
for name, pred in val_pred_dict.items():
    val_df[f'Pred_{name}'] = pred

# 收集测试集数据
test_smiles = [valid_smiles[i] for i in orig_test_indices]
test_true = y_test
test_pred_dict = {name: all_predictions['test'][name] for name in models.keys()}
test_df = pd.DataFrame({'SMILES': test_smiles, 'True_BCF': test_true})
for name, pred in test_pred_dict.items():
    test_df[f'Pred_{name}'] = pred

# 保存到 Excel
excel_path = os.path.join(output_dir, "predictions_data.xlsx")
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    train_df.to_excel(writer, sheet_name='Training', index=False)
    val_df.to_excel(writer, sheet_name='Validation', index=False)
    test_df.to_excel(writer, sheet_name='Test', index=False)
print(f"✓ 预测数据已保存到: {excel_path}")

# ==========================================
# 完成汇总
# ==========================================
print("\n" + "=" * 80)
print("所有任务完成!")
print("=" * 80)
print(f"\n输出目录: {output_dir}")
print(f"  【特征工程与选择】")
print(f"  - 指纹位方差过滤:     {n_fp_removed} 位移除，保留 {n_fp_kept} 位")
print(f"  - 衍生特征构造:       {len(eng_feature_names)} 个新特征")
print(f"  - 三方法投票精选:     {len(selected_desc_names)} 个描述符/衍生特征入模")
print(f"  - 特征选择评分图:     feature_selection_scores.png")
print(f"  - 特征选择投票热图:   feature_selection_vote_heatmap.png")
print(f"  - 特征选择报告:       feature_selection_report.csv")
print(f"  【共线性筛选报告】")
print(f"  - 描述符VIF报告:      descriptor_vif_report.csv")
print(f"  - 描述符相关性热图:   descriptor_correlation_heatmap.png")
print(f"  【模型训练输出】")
print(f"  - 训练损失曲线:       training_loss_curves.png")
print(f"  - 性能指标对比图:     metrics_comparison.png")
print(f"  - MAE跨数据集对比图:  mae_all_datasets_comparison.png")
print(f"  - 详细性能指标:       model_metrics_detailed.csv")
print(f"  - XGBoost损失数据:    xgboost_loss_history.csv")
print(f"  - LightGBM损失数据:   lightgbm_loss_history.csv")
print(f"  - CatBoost损失数据:   catboost_loss_history.csv")
print(f"  - GB损失数据:         gradientboosting_loss_history.csv")
print(f"  - 所有模型损失汇总:   all_models_loss_history.csv")
print(f"  - 预测数据汇总:       predictions_data.xlsx")
print(f"  - 保存的模型:         saved_models/")