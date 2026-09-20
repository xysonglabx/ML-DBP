import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, Crippen, Lipinski, MolSurf, rdMolDescriptors, MACCSkeys, GraphDescriptors
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor as _RFR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

print("=" * 80)
print("SHAP分析 - 从已保存的模型加载 (XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting)")
print("数据文件: AR.csv")
print("=" * 80)

# ==========================================
# 配置参数（沿用 shap_plot.py 风格）
# ==========================================
OUTPUT_DIR = r"F:\python-workspace\Our\YanEr\DBPs_Script\beiyesi\ml_model_ARmodel_AR5_pt\output"               # 输出目录
MODELS_DIR = r"F:\python-workspace\Our\YanEr\DBPs_Script\beiyesi\ml_model_ARmodel_AR5_pt\saved_models"  # 模型目录
TOP_N_FEATURES = 15                                    # Top特征数量

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n✓ 输出目录: {OUTPUT_DIR}")
print(f"✓ 模型目录: {MODELS_DIR}")

# 设置图形参数 - 全局字体为Arial,全部加粗
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
# 1. 增强特征提取（与 model_ar5_pt.py 完全一致）
# ==========================================
def extract_enhanced_features(smiles_list):
    fps = []
    phys_features = []
    valid_indices = []
    valid_smiles = []

    # 尝试导入 EState 指纹
    use_estate_fp = False
    try:
        from rdkit.Chem.EState import EStateFingerprinter
        estate_fingerprinter = EStateFingerprinter()
        use_estate_fp = True
        print("  ✓ EState 指纹可用，将添加至指纹特征")
    except ImportError:
        print("  ⚠️ EState 指纹不可用，跳过")

    # 子结构定义
    substructures = {
        'Carboxyl': 'C(=O)O',
        'Carboxyl_anion': 'C(=O)[O-]',
        'Nitro': '[N+](=O)[O-]',
        'Cyano': 'C#N',
        'Amino': '[NH2]',
        'Hydroxy': '[OH]',
        'Ether': 'C-O-C',
        'Amide': 'C(=O)N',
        'Ester': 'C(=O)O',
        'Sulfonyl': 'S(=O)(=O)',
        'Sulfonate': 'S(=O)(=O)[O-]',
        'Phenol': 'c1ccccc1O',
        'Aniline': 'c1ccccc1N',
        'Nitro_aromatic': 'c1ccccc1[N+](=O)[O-]',
        'Halogen_aromatic': 'c1ccccc1[F,Cl,Br,I]',
    }

    # 官能团碎片名称列表
    from rdkit.Chem import Fragments
    fragment_names = [
        'fr_Al_COO', 'fr_Al_OH', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH',
        'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_S', 'fr_HOCC',
        'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_alkyl_halide', 'fr_halogen',
    ]

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            # ---------- 指纹 ----------
            fp_2048 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_1024_r3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
            fp_2048_r4 = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
            fp_2048_r5 = AllChem.GetMorganFingerprintAsBitVect(mol, 5, nBits=2048)
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            fp_list = [np.array(fp_2048), np.array(fp_1024_r3), np.array(fp_2048_r4), np.array(fp_2048_r5), np.array(maccs_fp)]
            if use_estate_fp:
                estate_fp = estate_fingerprinter.GetFingerprint(mol)
                fp_list.append(np.array(estate_fp))
            combined_fp = np.concatenate(fp_list)

            # ---------- 2D 描述符 ----------
            mol_wt = Descriptors.MolWt(mol)
            heavy_atom_mol_wt = Descriptors.HeavyAtomMolWt(mol)
            exact_mol_wt = Descriptors.ExactMolWt(mol)
            mol_logp = Descriptors.MolLogP(mol)
            mol_mr = Descriptors.MolMR(mol)
            tpsa = Descriptors.TPSA(mol)
            labute_asa = Descriptors.LabuteASA(mol)
            num_h_donors = Descriptors.NumHDonors(mol)
            num_h_acceptors = Descriptors.NumHAcceptors(mol)
            num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            num_heteroatoms = Descriptors.NumHeteroatoms(mol)
            num_aromatic_rings = Descriptors.NumAromaticRings(mol)
            num_aliphatic_rings = Descriptors.NumAliphaticRings(mol)
            ring_count = Descriptors.RingCount(mol)
            fraction_csp3 = Descriptors.FractionCSP3(mol)
            max_abs_partial_charge = Descriptors.MaxAbsPartialCharge(mol) if Descriptors.MaxAbsPartialCharge(mol) else 0
            min_abs_partial_charge = Descriptors.MinAbsPartialCharge(mol) if Descriptors.MinAbsPartialCharge(mol) else 0
            max_partial_charge = Descriptors.MaxPartialCharge(mol) if Descriptors.MaxPartialCharge(mol) else 0
            min_partial_charge = Descriptors.MinPartialCharge(mol) if Descriptors.MinPartialCharge(mol) else 0
            f_count = smiles.count('F')
            cf2_count = smiles.count('C(F)(F)')
            cf3_count = smiles.count('C(F)(F)F')
            no_count = Descriptors.NOCount(mol)
            nhoh_count = Descriptors.NHOHCount(mol)
            num_valence_electrons = Descriptors.NumValenceElectrons(mol)
            num_radical_electrons = Descriptors.NumRadicalElectrons(mol)
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
            num_saturated_rings = Descriptors.NumSaturatedRings(mol)
            num_double_bonds = getattr(Descriptors, 'NumDoubleBonds', lambda m: 0)(mol)
            num_triple_bonds = getattr(Descriptors, 'NumTripleBonds', lambda m: 0)(mol)
            num_aromatic_bonds = getattr(Descriptors, 'NumAromaticBonds', lambda m: 0)(mol)
            o_count = smiles.count('O')
            n_count = smiles.count('N')
            s_count = smiles.count('S')
            cl_count = smiles.count('Cl')
            br_count = smiles.count('Br')
            i_count = smiles.count('I')

            desc_list = [
                mol_wt, heavy_atom_mol_wt, exact_mol_wt,
                mol_logp, mol_mr, tpsa, labute_asa,
                num_h_donors, num_h_acceptors, num_rotatable_bonds,
                num_heteroatoms, num_aromatic_rings, num_aliphatic_rings,
                ring_count, fraction_csp3,
                max_abs_partial_charge, min_abs_partial_charge,
                max_partial_charge, min_partial_charge,
                f_count, cf2_count, cf3_count,
                no_count, nhoh_count,
                num_valence_electrons, num_radical_electrons,
                balaban_j, bertz_ct,
                chi0, chi0n, chi0v,
                chi1, chi1n, chi1v,
                kappa1, kappa2, kappa3,
                hall_kier_alpha, ipc,
                num_saturated_rings,
                num_double_bonds, num_triple_bonds, num_aromatic_bonds,
                o_count, n_count, s_count, cl_count, br_count, i_count,
            ]

            # 更多 Chi 指数
            for chi_name in ['Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v']:
                desc_list.append(getattr(Descriptors, chi_name, lambda m: 0)(mol))

            # VSA 描述符
            for idx in range(1, 15):
                try:
                    val = getattr(Descriptors, f'PEOE_VSA{idx}')(mol)
                    if val is None:
                        val = 0
                except:
                    val = 0
                desc_list.append(val)
            for idx in range(1, 11):
                try:
                    val = getattr(Descriptors, f'SMR_VSA{idx}')(mol)
                    if val is None:
                        val = 0
                except:
                    val = 0
                desc_list.append(val)
            for idx in range(1, 13):
                try:
                    val = getattr(Descriptors, f'SlogP_VSA{idx}')(mol)
                    if val is None:
                        val = 0
                except:
                    val = 0
                desc_list.append(val)
            for idx in range(1, 12):
                try:
                    val = getattr(Descriptors, f'EState_VSA{idx}')(mol)
                    if val is None:
                        val = 0
                except:
                    val = 0
                desc_list.append(val)

            # 子结构计数
            for name, smarts in substructures.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    desc_list.append(0)
                    continue
                matches = mol.GetSubstructMatches(pattern)
                desc_list.append(len(matches))

            # 比例特征（安全版本）
            heavy_atoms = mol.GetNumHeavyAtoms()
            if heavy_atoms > 0:
                aromatic_atoms = 0
                try:
                    aromatic_atoms = len(mol.GetAromaticAtoms())
                except:
                    try:
                        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if hasattr(atom, 'GetIsAromatic') and atom.GetIsAromatic())
                    except:
                        aromatic_atoms = 0
                aromatic_atom_ratio = aromatic_atoms / heavy_atoms
                hetero_atom_ratio = num_heteroatoms / heavy_atoms
                halogen_count = f_count + cl_count + br_count + i_count
                halogen_ratio = halogen_count / heavy_atoms
                rot_bond_ratio = num_rotatable_bonds / heavy_atoms
                ring_info = mol.GetRingInfo()
                ring_atoms = set()
                for ring in ring_info.AtomRings():
                    ring_atoms.update(ring)
                ring_atom_ratio = len(ring_atoms) / heavy_atoms
                double_bond_ratio = num_double_bonds / heavy_atoms
                triple_bond_ratio = num_triple_bonds / heavy_atoms
            else:
                aromatic_atom_ratio = hetero_atom_ratio = halogen_ratio = rot_bond_ratio = ring_atom_ratio = double_bond_ratio = triple_bond_ratio = 0

            desc_list.extend([
                aromatic_atom_ratio, hetero_atom_ratio, halogen_ratio,
                rot_bond_ratio, ring_atom_ratio, double_bond_ratio, triple_bond_ratio
            ])

            # 电荷统计
            try:
                AllChem.ComputeGasteigerCharges(mol)
                charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
            except:
                charges = [0] * mol.GetNumAtoms()
            abs_charges = np.abs(charges)
            total_abs_charge = np.sum(abs_charges)
            rms_charge = np.sqrt(np.sum(np.square(charges)))
            max_abs_charge = np.max(abs_charges) if len(abs_charges) > 0 else 0
            if len(charges) > 1:
                charge_var = np.var(charges)
                charge_skew = (np.mean((charges - np.mean(charges))**3) / (np.std(charges)**3)) if np.std(charges) > 0 else 0
                charge_kurt = (np.mean((charges - np.mean(charges))**4) / (np.std(charges)**4)) - 3 if np.std(charges) > 0 else 0
                charge_range = max(charges) - min(charges)
                charge_q75 = np.percentile(charges, 75)
                charge_q25 = np.percentile(charges, 25)
                charge_iqr = charge_q75 - charge_q25
            else:
                charge_var = charge_skew = charge_kurt = charge_range = charge_iqr = 0
            desc_list.extend([total_abs_charge, rms_charge, max_abs_charge, charge_var, charge_skew, charge_kurt, charge_range, charge_iqr])

            # 环比例
            total_rings = ring_count
            if total_rings > 0:
                aromatic_ring_ratio = num_aromatic_rings / total_rings
                saturated_ring_ratio = num_saturated_rings / total_rings
            else:
                aromatic_ring_ratio = saturated_ring_ratio = 0
            desc_list.extend([aromatic_ring_ratio, saturated_ring_ratio])

            # 额外环类型
            extra_descs = [
                getattr(Descriptors, 'NumSaturatedHeterocycles', lambda m: 0)(mol),
                getattr(Descriptors, 'NumUnsaturatedHeterocycles', lambda m: 0)(mol),
                getattr(Descriptors, 'NumSaturatedCarbocycles', lambda m: 0)(mol),
                getattr(Descriptors, 'NumUnsaturatedCarbocycles', lambda m: 0)(mol),
            ]
            desc_list.extend(extra_descs)

            # 官能团碎片计数
            for frag_name in fragment_names:
                try:
                    frag_func = getattr(Fragments, frag_name)
                    val = frag_func(mol)
                    if val is None:
                        val = 0
                except:
                    val = 0
                desc_list.append(val)

            # 图论描述符
            try:
                graph_diameter = GraphDescriptors.GraphDiameter(mol)
                graph_radius = GraphDescriptors.GraphRadius(mol)
            except:
                graph_diameter = graph_radius = 0
            desc_list.extend([graph_diameter, graph_radius])

            # ---------- 3D 几何特征 ----------
            mol_3d = None
            try:
                mol_3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
                AllChem.MMFFOptimizeMolecule(mol_3d)
                mol_volume = rdMolDescriptors.CalcExactMolVolume(mol_3d)
                mol_surface_area = rdMolDescriptors.CalcExactMolSurfaceArea(mol_3d)
                inertia = AllChem.InertialMoments(mol_3d)
                if len(inertia) >= 3:
                    sorted_inertia = sorted(inertia)
                    asphericity = (3 * (inertia[0]*inertia[1] + inertia[0]*inertia[2] + inertia[1]*inertia[2]) -
                                   (inertia[0] + inertia[1] + inertia[2])**2) / (2 * (inertia[0] + inertia[1] + inertia[2])**2)
                    sphericity = (inertia[0] * inertia[1] * inertia[2])**(1/3) / ((inertia[0] + inertia[1] + inertia[2]) / 3)
                else:
                    asphericity = sphericity = 0
            except:
                mol_volume = mol_surface_area = asphericity = sphericity = 0
                inertia = [0, 0, 0]

            desc_list.extend([mol_volume, mol_surface_area, inertia[0], inertia[1], inertia[2], asphericity, sphericity])

            # 分子最远原子距离（直径）
            max_dist = 0
            try:
                if mol_3d is not None:
                    conf = mol_3d.GetConformer()
                    coords = conf.GetPositions()
                    if len(coords) > 1:
                        max_dist = 0
                        for j in range(len(coords)):
                            for k in range(j+1, len(coords)):
                                dist = np.linalg.norm(coords[j] - coords[k])
                                if dist > max_dist:
                                    max_dist = dist
            except:
                pass
            desc_list.append(max_dist)
            desc_list.append(max_dist ** 2)  # 直径平方

            # 氢键供体-受体对的数量、平均距离、最小距离
            hbond_pairs = 0
            hbond_avg_dist = 0
            hbond_min_dist = 0
            try:
                if mol_3d is not None:
                    h_donors = [atom for atom in mol_3d.GetAtoms() if atom.GetNumExplicitHs() > 0 and atom.GetAtomicNum() in [7,8]]
                    h_acceptors = [atom for atom in mol_3d.GetAtoms() if atom.GetAtomicNum() in [7,8] and atom.GetNumExplicitHs() < 2]
                    conf = mol_3d.GetConformer()
                    distances = []
                    for donor in h_donors:
                        for acceptor in h_acceptors:
                            if donor.GetIdx() == acceptor.GetIdx():
                                continue
                            d_pos = conf.GetAtomPosition(donor.GetIdx())
                            a_pos = conf.GetAtomPosition(acceptor.GetIdx())
                            dist = d_pos.Distance(a_pos)
                            if dist < 3.0:
                                distances.append(dist)
                    hbond_pairs = len(distances)
                    if distances:
                        hbond_avg_dist = np.mean(distances)
                        hbond_min_dist = np.min(distances)
            except:
                pass
            desc_list.extend([hbond_pairs, hbond_avg_dist, hbond_min_dist])

            # 溶剂可及表面积（SASA）
            try:
                sasa = rdMolDescriptors.CalcSASA(mol_3d) if mol_3d is not None else 0
            except:
                sasa = 0
            desc_list.append(sasa)

            # 分子表面静电势（ESP）统计
            try:
                esp = rdMolDescriptors.CalcElectrostaticPotential(mol, charges)  # 需要已有电荷
                if esp:
                    esp_max = np.max(esp)
                    esp_min = np.min(esp)
                    esp_mean = np.mean(esp)
                    esp_std = np.std(esp)
                else:
                    esp_max = esp_min = esp_mean = esp_std = 0
            except:
                esp_max = esp_min = esp_mean = esp_std = 0
            desc_list.extend([esp_max, esp_min, esp_mean, esp_std])

            # USR 形状描述符
            try:
                usr = rdMolDescriptors.GetUSR(mol)
                desc_list.extend(usr)  # 12 个值
            except:
                desc_list.extend([0] * 12)

            # 药效团指纹（取前 128 位）
            try:
                pharm2d_fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
                pharm2d_array = np.array(pharm2d_fp)
                desc_list.extend(pharm2d_array[:128])
            except:
                desc_list.extend([0] * 128)

            # 衍生特征：交叉项
            logp_tpsa = mol_logp * tpsa
            logp_sq = mol_logp ** 2
            tpsa_sq = tpsa ** 2
            hb_total = num_h_donors + num_h_acceptors
            hb_ratio = hb_total / (heavy_atoms + 1e-8)
            rot_ratio = num_rotatable_bonds / (heavy_atoms + 1e-8)
            mr_density = mol_mr / (mol_wt + 1e-8)
            desc_list.extend([logp_tpsa, logp_sq, tpsa_sq, hb_ratio, rot_ratio, mr_density])

            # 手性中心数量
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            desc_list.append(len(chiral_centers))

            # 芳香环的平均大小（安全版本）
            try:
                from rdkit.Chem import GetSymmSSSR
                rings = GetSymmSSSR(mol)
                aromatic_rings = []
                for ring in rings:
                    try:
                        is_aromatic = True
                        for idx in ring:
                            atom = mol.GetAtomWithIdx(idx)
                            if not (hasattr(atom, 'GetIsAromatic') and atom.GetIsAromatic()):
                                is_aromatic = False
                                break
                        if is_aromatic:
                            aromatic_rings.append(ring)
                    except:
                        continue
                if aromatic_rings:
                    avg_aromatic_ring_size = np.mean([len(ring) for ring in aromatic_rings])
                else:
                    avg_aromatic_ring_size = 0
            except:
                avg_aromatic_ring_size = 0
            desc_list.append(avg_aromatic_ring_size)

            # 处理缺失值
            desc_list = [0 if (x is None or np.isnan(x) or np.isinf(x)) else x for x in desc_list]

            fps.append(combined_fp)
            phys_features.append(desc_list)
            valid_indices.append(i)
            valid_smiles.append(smiles)

        except Exception as e:
            print(f"⚠️ 处理 SMILES '{smiles}' 时发生异常: {e}")
            continue

    if len(valid_indices) == 0:
        print("错误：没有有效的分子！请检查SMILES数据。")
        return None, [], [], 0

    X_fp = np.array(fps)
    X_phys = np.array(phys_features)
    X_combined = np.hstack([X_fp, X_phys])
    return X_combined, valid_indices, valid_smiles, X_phys.shape[1]

# ==========================================
# 2. 特征筛选函数（低方差 + 相关性）
# ==========================================
def remove_low_variance_features(X_phys, feature_names, threshold=0.01):
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
# 3. 特征工程（衍生特征构造，与 model_ar5_pt.py 一致）
# ==========================================
def construct_engineered_features(desc_df, final_descriptor_names):
    """
    构造衍生特征，返回衍生特征数组和名称列表
    """
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

    if eng_features:
        X_eng = np.column_stack([eng_features[k] for k in eng_feature_names])
        X_eng = np.nan_to_num(X_eng, nan=0, posinf=0, neginf=0)
        print(f"  构造衍生特征: {len(eng_feature_names)} 个")
        for n in eng_feature_names:
            print(f"    + {n}")
    else:
        X_eng = np.zeros((len(mw), 0))
        print("  无可构造的衍生特征（描述符不足）")
    return X_eng, eng_feature_names

# ==========================================
# 4. 特征选择（三方法投票）
# ==========================================
def select_features_voting(X_desc_eng, y, desc_eng_names, top_k_ratio=0.6, random_state=42):
    """
    互信息 + RF重要性 + LASSO 投票选择特征
    返回: X_selected, selected_names, voting_selected_indices
    """
    print(f"\n  输入: 描述符+衍生特征维度 = {X_desc_eng.shape[1]}")

    ss = StandardScaler()
    X_desc_eng_std = ss.fit_transform(X_desc_eng)
    X_desc_eng_std = np.nan_to_num(X_desc_eng_std, nan=0, posinf=0, neginf=0)

    # 方法1: 互信息
    print("[方法1] 互信息特征评分...")
    mi_scores = mutual_info_regression(X_desc_eng, y, random_state=random_state)
    mi_ranking = np.argsort(mi_scores)[::-1]
    top_k_mi = max(10, int(X_desc_eng.shape[1] * top_k_ratio))
    mi_selected = set(mi_ranking[:top_k_mi].tolist())

    # 方法2: 随机森林重要性
    print("[方法2] 随机森林特征重要性评分...")
    rf_selector = _RFR(n_estimators=300, max_depth=10, min_samples_leaf=2,
                       max_features='sqrt', n_jobs=-1, random_state=random_state)
    rf_selector.fit(X_desc_eng, y)
    rf_importances = rf_selector.feature_importances_
    rf_ranking = np.argsort(rf_importances)[::-1]
    top_k_rf = max(10, int(X_desc_eng.shape[1] * top_k_ratio))
    rf_selected = set(rf_ranking[:top_k_rf].tolist())

    # 方法3: LASSO
    print("[方法3] LASSO特征选择...")
    lasso_cv = LassoCV(cv=5, max_iter=5000, random_state=random_state,
                       alphas=np.logspace(-4, 0, 50), n_jobs=-1)
    lasso_cv.fit(X_desc_eng_std, y)
    lasso_coefs = np.abs(lasso_cv.coef_)
    lasso_selected = set(np.where(lasso_coefs > 0)[0].tolist())

    # 投票融合
    vote_counts = {}
    for idx in range(X_desc_eng.shape[1]):
        votes = (1 if idx in mi_selected else 0) + (1 if idx in rf_selected else 0) + (1 if idx in lasso_selected else 0)
        vote_counts[idx] = votes

    voting_selected = sorted([i for i, v in vote_counts.items() if v >= 2])
    if len(voting_selected) < 5:
        print(f"  [警告] 投票选出特征过少({len(voting_selected)})，回退到RF Top-15")
        voting_selected = sorted(rf_ranking[:15].tolist())

    X_selected = X_desc_eng[:, voting_selected]
    selected_names = [desc_eng_names[i] for i in voting_selected]

    print(f"\n  投票结果: {len(voting_selected)} 个描述符/衍生特征被保留")
    for i, name in enumerate(selected_names):
        v = vote_counts[voting_selected[i]]
        print(f"    [{v}票] {name}")

    return X_selected, selected_names, voting_selected

# ==========================================
# 5. 加载数据（使用 AR.csv）
# ==========================================
print("\n" + "=" * 80)
print("步骤 1: 加载数据")
print("=" * 80)

DATA_FILE = "AR.csv"
possible_paths = [
    DATA_FILE,
    os.path.join(os.getcwd(), DATA_FILE),
    r"F:\python-workspace\Our\YanEr\DBPs_Script\beiyesi\AR.csv"
]

df = None
for path in possible_paths:
    try:
        df = pd.read_csv(path)
        df.columns = ['smiles', 'AR']
        print(f"✓ 成功加载数据: {path}")
        break
    except:
        continue

if df is None:
    print("❌ 无法找到 AR.csv 文件!")
    exit(1)

print(f"✓ 数据行数: {len(df)}")
print(f"✓ 目标值范围: [{df['AR'].min():.2f}, {df['AR'].max():.2f}]")

# ==========================================
# 6. 提取增强特征
# ==========================================
print("\n正在提取增强特征 (扩展描述符 + 多重指纹 + 3D/药效团等)...")
X, valid_idx, valid_smiles, n_desc = extract_enhanced_features(df['smiles'])

if X is None or len(valid_idx) == 0:
    print("错误：没有有效的分子，无法继续。请检查 SMILES 格式。")
    exit(1)

y = df['AR'].iloc[valid_idx].values
print(f"✓ 有效样本数: {len(X)}")
print(f"✓ 特征维度: {X.shape[1]} (指纹: {X.shape[1] - n_desc}, 描述符: {n_desc})")

# ==========================================
# 7. 描述符共线性筛选（低方差 + 相关性）
# ==========================================
print("\n" + "=" * 80)
print("描述符共线性筛选 (低方差 + 相关性过滤)")
print("=" * 80)

n_fp = X.shape[1] - n_desc
X_fp_part = X[:, :n_fp]
X_desc_part = X[:, n_fp:]

# 完整描述符名称列表（必须与 extract_enhanced_features 中 desc_list 顺序一致）
descriptor_names = [
    'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'MolLogP', 'MolMR', 'TPSA', 'LabuteASA',
    'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'NumHeteroatoms', 'NumAromaticRings',
    'NumAliphaticRings', 'RingCount', 'FractionCSP3', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',
    'MaxPartialCharge', 'MinPartialCharge', 'F_Count', 'CF2_Count', 'CF3_Count', 'NOCount', 'NHOHCount',
    'NumValenceElectrons', 'NumRadicalElectrons', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v',
    'Chi1', 'Chi1n', 'Chi1v', 'Kappa1', 'Kappa2', 'Kappa3', 'HallKierAlpha', 'Ipc', 'NumSaturatedRings',
    'NumDoubleBonds', 'NumTripleBonds', 'NumAromaticBonds', 'O_Count', 'N_Count', 'S_Count', 'Cl_Count',
    'Br_Count', 'I_Count',
    'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
    'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8',
    'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
    'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SMR_VSA10',
    'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8',
    'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
    'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8',
    'EState_VSA9', 'EState_VSA10', 'EState_VSA11',
    'Carboxyl_Count', 'Carboxyl_anion_Count', 'Nitro_Count', 'Cyano_Count', 'Amino_Count', 'Hydroxy_Count',
    'Ether_Count', 'Amide_Count', 'Ester_Count', 'Sulfonyl_Count', 'Sulfonate_Count', 'Phenol_Count',
    'Aniline_Count', 'Nitro_aromatic_Count', 'Halogen_aromatic_Count',
    'AromaticAtomRatio', 'HeteroAtomRatio', 'HalogenRatio', 'RotBondRatio', 'RingAtomRatio',
    'DoubleBondRatio', 'TripleBondRatio',
    'TotalAbsCharge', 'RMSCharge', 'MaxAbsCharge', 'ChargeVariance', 'ChargeSkewness', 'ChargeKurtosis',
    'ChargeRange', 'ChargeIQR',
    'AromaticRingRatio', 'SaturatedRingRatio',
    'NumSaturatedHeterocycles', 'NumUnsaturatedHeterocycles', 'NumSaturatedCarbocycles', 'NumUnsaturatedCarbocycles',
    'fr_Al_COO', 'fr_Al_OH', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_S',
    'fr_HOCC', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_alkyl_halide', 'fr_halogen',
    'GraphDiameter', 'GraphRadius',
    'MolVolume', 'MolSurfaceArea', 'InertiaX', 'InertiaY', 'InertiaZ', 'Asphericity', 'Sphericity',
    'MaxDistance', 'MaxDistanceSq',
    'HBondPairs', 'HBondAvgDist', 'HBondMinDist',
    'SASA',
    'ESP_Max', 'ESP_Min', 'ESP_Mean', 'ESP_Std',
    *[f'USR_{i}' for i in range(12)],
    *[f'Pharm2D_{i}' for i in range(128)],
    'LogP_TPSA', 'LogP_Sq', 'TPSA_Sq',
    'HB_Ratio', 'RotBond_Ratio', 'MR_Density',
    'ChiralCenters', 'AvgAromaticRingSize',
]

assert len(descriptor_names) == n_desc, (
    f"描述符名称数量({len(descriptor_names)})与实际描述符维度({n_desc})不匹配！"
)

print(f"\n原始描述符数量: {n_desc}")

# Step 1: 低方差过滤（阈值 0.01，与 model_ar5_pt.py 一致）
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

print(f"\n  特征筛选完成:")
print(f"    原始总维度: {X.shape[1]} (指纹 {n_fp} + 描述符 {n_desc})")
print(f"    过滤后描述符维度: {len(final_descriptor_names)}")

# ==========================================
# 8. 特征工程 (指纹方差过滤 + 衍生特征)
# ==========================================
print("\n" + "=" * 80)
print("特征工程 (指纹方差过滤 + 衍生特征构造)")
print("=" * 80)

# --- A. 指纹位方差过滤 ---
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
X_eng, eng_feature_names = construct_engineered_features(desc_df, final_descriptor_names)

# --- C. 拼合全部特征 ---
X_engineered = np.hstack([X_fp_selected, X_desc_clean, X_eng])
all_feature_names = (
    [f"FP_{i}" for i in range(n_fp_kept)]
    + final_descriptor_names
    + eng_feature_names
)
print(f"\n  特征工程后总维度: {X_engineered.shape[1]}")

# ==========================================
# 9. 特征选择 (三方法投票)
# ==========================================
print("\n" + "=" * 80)
print("特征选择 (三方法投票)")
print("=" * 80)

n_fp_final = n_fp_kept
X_fp_final = X_engineered[:, :n_fp_final]
X_desc_eng = X_engineered[:, n_fp_final:]
desc_eng_names = all_feature_names[n_fp_final:]

X_desc_eng_selected, selected_desc_names, voting_selected = select_features_voting(
    X_desc_eng, y, desc_eng_names, top_k_ratio=0.6, random_state=42
)

# 拼合最终特征矩阵
X_final = np.hstack([X_fp_final, X_desc_eng_selected])
final_feature_names = (
    [f"FP_{i}" for i in range(n_fp_final)]
    + selected_desc_names
)
print(f"\n  ★ 最终特征矩阵维度: {X_final.shape[1]}")

# ==========================================
# 10. 加载模型和配置
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
models_to_load = {
    'LightGBM': 'lightgbm_model.pkl',
    'CatBoost': 'catboost_model.pkl',
    'XGBoost': 'xgboost_model.pkl',
    'RandomForest': 'randomforest_model.pkl',
    'GradientBoosting': 'gradientboosting_model.pkl'
}

loaded_models = {}
for name, filename in models_to_load.items():
    model_path = os.path.join(MODELS_DIR, filename)
    try:
        loaded_models[name] = joblib.load(model_path)
        print(f"✓ 加载{name}模型: {model_path}")
    except Exception as e:
        print(f"❌ 无法加载{name}模型: {e}")
        exit(1)

# ==========================================
# 11. 数据预处理和划分（复现训练时的划分）
# ==========================================
print("\n数据预处理...")
X_scaled = scaler.transform(X_final)
X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

# 使用固定种子 42（与 model_ar5_pt.py 一致）
print("使用训练时的数据划分 (seed=42, test_size=0.15)...")
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_scaled, y, range(len(y)), test_size=0.15, random_state=42
)

# 第二次划分（训练/验证）
X_train_sub, X_val, y_train_sub, y_val, train_sub_indices, val_indices = train_test_split(
    X_train, y_train, train_indices, test_size=0.15, random_state=42
)

# 构建原始索引到 valid_smiles 位置的映射
orig_to_valid_pos = {orig_idx: pos for pos, orig_idx in enumerate(valid_idx)}
train_positions = [orig_to_valid_pos[idx] for idx in train_sub_indices]
val_positions   = [orig_to_valid_pos[idx] for idx in val_indices]
test_positions  = [orig_to_valid_pos[idx] for idx in test_indices]

train_smiles = [valid_smiles[i] for i in train_positions]
val_smiles   = [valid_smiles[i] for i in val_positions]
test_smiles  = [valid_smiles[i] for i in test_positions]

print(f"✓ 训练集: {len(X_train_sub)} 条")
print(f"✓ 验证集: {len(X_val)} 条")
print(f"✓ 测试集: {len(X_test)} 条")

# ==========================================
# 12. 模型预测和评估（可选，仅用于验证）
# ==========================================
print("\n" + "=" * 80)
print("步骤 3: 模型评估")
print("=" * 80)

for name, model in loaded_models.items():
    pred_train = model.predict(X_train_sub)
    pred_test = model.predict(X_test)
    r2_train = r2_score(y_train_sub, pred_train)
    r2_test = r2_score(y_test, pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    mae_test = mean_absolute_error(y_test, pred_test)
    print(f"\n{name}:")
    print(f"  训练集 R²: {r2_train:.4f}")
    print(f"  测试集 R²: {r2_test:.4f}")
    print(f"  测试集 RMSE: {rmse_test:.4f}")
    print(f"  测试集 MAE: {mae_test:.4f}")

# ==========================================
# 13. SHAP分析 (针对精选描述符/衍生特征)
# ==========================================
print("\n" + "=" * 80)
print("步骤 4: SHAP分析 (精选描述符/衍生特征)")
print("=" * 80)

# 提取描述符/衍生特征部分
X_train_desc = X_train[:, n_fp_final:]
X_test_desc = X_test[:, n_fp_final:]

print(f"✓ 描述符/衍生特征维度 (精选后): {X_test_desc.shape[1]}")
print(f"✓ 特征名称: {selected_desc_names}")

X_test_desc_df = pd.DataFrame(X_test_desc, columns=selected_desc_names)

def get_top_features_shap(model, X_data_full, X_data_df, feature_names, top_n=20, model_name="Model"):
    """使用SHAP获取最重要的特征"""
    print(f"\n计算 {model_name} 的SHAP值...")
    # 针对 XGBoost 特殊处理，使用 PermutationExplainer 避免解析错误
    if model_name == 'XGBoost':
        print("  针对 XGBoost 使用 PermutationExplainer (模型无关解释器)...")
        explainer = shap.PermutationExplainer(model.predict, X_data_full[:100])
        shap_values = explainer.shap_values(X_data_full)
    else:
        try:
            explainer = shap.TreeExplainer(model, check_additivity=False)
            shap_values = explainer.shap_values(X_data_full)
        except Exception as e:
            print(f"  TreeExplainer 失败，回退到 PermutationExplainer: {e}")
            explainer = shap.PermutationExplainer(model.predict, X_data_full[:100])
            shap_values = explainer.shap_values(X_data_full)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)
    shap_values_desc = shap_values[:, n_fp_final:]  # 只取描述符部分
    mean_abs_shap = np.abs(shap_values_desc).mean(axis=0)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    top_features = feature_importance.head(top_n)['Feature'].tolist()
    print(f"  ✓ 已选取前 {top_n} 个最重要的描述符特征")
    return shap_values_desc, feature_importance, top_features

# 计算各模型的SHAP值
shap_results = {}
for name, model in loaded_models.items():
    shap_vals, imp_df, top_feats = get_top_features_shap(
        model, X_test, X_test_desc_df, selected_desc_names,
        TOP_N_FEATURES, name
    )
    shap_results[name] = (shap_vals, imp_df, top_feats)

# 保存特征重要性
all_importance = pd.concat([imp_df.assign(Model=name) for name, (_, imp_df, _) in shap_results.items()])
all_importance_path = os.path.join(OUTPUT_DIR, "feature_importance_all.csv")
all_importance.to_csv(all_importance_path, index=False)
print(f"\n✓ 特征重要性已保存: {all_importance_path}")

# 保存Top特征对比
top_comparison = pd.DataFrame({'Rank': range(1, TOP_N_FEATURES + 1)})
for name, (_, _, top_feats) in shap_results.items():
    top_comparison[f'{name}_Feature'] = top_feats
    top_comparison[f'{name}_SHAP'] = [shap_results[name][1][shap_results[name][1]['Feature'] == f]['Mean_Abs_SHAP'].values[0] for f in top_feats]
top_comparison_path = os.path.join(OUTPUT_DIR, f"top{TOP_N_FEATURES}_features.csv")
top_comparison.to_csv(top_comparison_path, index=False)
print(f"✓ Top{TOP_N_FEATURES}特征对比已保存: {top_comparison_path}")

# ==========================================
# 14. SHAP可视化函数（与 shap_plot.py 完全一致）
# ==========================================
def create_optimized_cmap(base_cmap, start=0.2, end=0.9):
    colors = base_cmap(np.linspace(start, end, 256))
    return LinearSegmentedColormap.from_list('optimized', colors)

def plot_shap_barplot_with_rose(shap_values, X_data, feature_names, model_name, color_scheme='viridis'):
    feature_indices = [list(X_data.columns).index(f) for f in feature_names]
    shap_subset = shap_values[:, feature_indices]
    mean_abs_shap = np.abs(shap_subset).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_names)
    shap_series.sort_values(ascending=False, inplace=True)
    sorted_features = shap_series.index.tolist()
    sorted_shap_values = shap_series.values

    base_length, fixed_increment, colored_ring_width = 2.0, 0.25, 1.0
    num_vars = len(feature_names)
    one_oclock_offset = np.pi / 21
    percentages = (sorted_shap_values / sorted_shap_values.sum()) * 100
    widths = (sorted_shap_values / sorted_shap_values.sum()) * 2 * np.pi
    thetas = np.cumsum([0] + widths[:-1].tolist()) - one_oclock_offset
    total_lengths = [base_length + i * fixed_increment for i in range(num_vars)]
    inner_heights = [max(0, tl - colored_ring_width) for tl in total_lengths]
    inner_colors = ['#F5F5F5', '#FFFFFF'] * (num_vars // 2 + 1)

    cmap_base = COLOR_SCHEMES[color_scheme]
    cmap = create_optimized_cmap(cmap_base)
    color_norm = mcolors.Normalize(vmin=np.quantile(sorted_shap_values, 0.25),
                                   vmax=np.quantile(sorted_shap_values, 0.75))
    colors = cmap(color_norm(sorted_shap_values))

    fig_width = 10
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=600, facecolor='white')

    left_margin = 0.10
    right_margin = 0.22
    bottom_margin = 0.12
    top_margin = 0.12
    main_plot_width = 1 - left_margin - right_margin
    plot_bottom = bottom_margin
    plot_height = 1 - bottom_margin - top_margin

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

    main_ax_left = cbar_left + colorbar_width + 0.05
    ax0 = fig.add_axes([main_ax_left, plot_bottom, main_plot_width, plot_height])
    ax0.xaxis.tick_bottom()
    ax0.xaxis.set_label_position("bottom")
    ax0.invert_xaxis()

    bar_height = 0.65
    bar_positions = range(len(sorted_features))
    bars = ax0.barh(y=bar_positions, width=sorted_shap_values, color=colors,
                    height=bar_height, edgecolor='white', linewidth=1.2)

    x_range = max(sorted_shap_values) - min(sorted_shap_values)
    offset = x_range * 0.01
    for i, (bar, value) in enumerate(zip(bars, sorted_shap_values)):
        label_x = value - offset
        label_y = i
        ax0.text(label_x, label_y, f'{value:.4f}',
                 ha='left', va='center',
                 fontsize=12, fontweight='bold', family='Arial',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                           edgecolor='#CCCCCC', alpha=0.9))

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

    label_fontsize = 24
    label_x = 1 - right_margin + 0.02
    for i, feature in enumerate(sorted_features):
        display_coords = ax0.transData.transform((0, i))
        fig_coords = fig.transFigure.inverted().transform(display_coords)
        y_position = fig_coords[1]
        fig.text(label_x, y_position, feature,
                 ha='left', va='center',
                 color='black', fontsize=label_fontsize, fontweight='bold', family='Arial')

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, f'shap_barplot_rose_{model_name}_{color_scheme}.jpg')
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return output_file

def plot_shap_beeswarm(shap_values, X_data, feature_names, model_name, color_scheme='viridis'):
    feature_indices = [list(X_data.columns).index(f) for f in feature_names]
    shap_subset = shap_values[:, feature_indices]
    X_subset = X_data[feature_names]
    mean_abs_shap = np.abs(shap_subset).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_names)
    shap_series.sort_values(ascending=False, inplace=True)
    sorted_features = shap_series.index.tolist()
    sorted_indices = [feature_names.index(f) for f in sorted_features]
    shap_subset_sorted = shap_subset[:, sorted_indices]
    X_subset_sorted = X_subset[sorted_features]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=600, facecolor='white')
    shap.summary_plot(shap_subset_sorted, X_subset_sorted,
                      plot_type="dot", show=False,
                      max_display=len(sorted_features), cmap=color_scheme)
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
    feature_indices = [list(X_data.columns).index(f) for f in feature_names]
    shap_subset = shap_values[:, feature_indices]
    X_subset = X_data[feature_names]
    mean_abs_shap = np.abs(shap_subset).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_names)
    shap_series.sort_values(ascending=False, inplace=True)
    sorted_features = shap_series.index.tolist()
    sorted_indices = [feature_names.index(f) for f in sorted_features]
    shap_subset_sorted = shap_subset[:, sorted_indices]
    X_subset_sorted = X_subset[sorted_features]

    fig, ax = plt.subplots(figsize=(20, 10), dpi=600, facecolor='white')
    shap.summary_plot(shap_subset_sorted, X_subset_sorted,
                      plot_type="layered_violin", cmap=color_scheme,
                      show=False, max_display=len(feature_names))
    plt.xlabel('SHAP Value', fontsize=28, fontweight='bold', family='Arial', labelpad=12)
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
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
    current_xlim = ax.get_xlim()
    ax.spines['left'].set_bounds(ax.get_ylim()[0], ax.get_ylim()[1])
    ax.spines['bottom'].set_bounds(current_xlim[0], current_xlim[1])
    for spine in ax.spines.values():
        spine.set_linewidth(2)

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
    if output_dir is None:
        output_dir = OUTPUT_DIR

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

    top_features = importance_df.sort_values('Mean_Abs_SHAP', ascending=False).head(top_n)
    feature_names = top_features['Feature'].tolist()
    mean_abs_shap = top_features['Mean_Abs_SHAP'].values

    feature_indices = [list(X_data_df.columns).index(f) for f in feature_names]
    shap_subset = shap_values[:, feature_indices]
    X_subset = X_data_df[feature_names]

    percentages = mean_abs_shap / mean_abs_shap.sum() * 100
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
    vmax = np.ceil(max(mean_abs_shap))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(mean_abs_shap))

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
        coll.set_sizes([40])
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
    cbar_ax.set_ylabel("Feature value", fontsize=16, fontweight='bold', family='Arial', rotation=90)
    cbar_ax.yaxis.set_label_coords(2.0, 0.5)
    cbar_ax.tick_params(labelsize=14, width=2, length=6, direction='in')
    for label in cbar_ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')

    ax2_pos = [POLAR_X, POLAR_Y, POLAR_SIZE, POLAR_SIZE]
    ax2 = fig.add_axes(ax2_pos, projection='polar')
    norm = mcolors.Normalize(vmin=min(mean_abs_shap), vmax=max(mean_abs_shap))
    colors = cmap(norm(mean_abs_shap))
    ax2.bar(
        theta,
        percentages,
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

    for angle, percent, name, raw_val in zip(theta, percentages, feature_names, mean_abs_shap):
        angle_deg = np.degrees(angle)
        visual_top = POLAR_BOTTOM_VAL + POLAR_GAP + percent
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
        ax2.text(angle, pos_outer,
                 f"{name}\n{percent:.1f}%",
                 ha=alignment_ha, va=alignment_va,
                 rotation=rotation, rotation_mode='anchor',
                 fontsize=14, fontweight='bold', family='Arial',
                 color='black',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))

    for angle, percent, raw_val in zip(theta, percentages, mean_abs_shap):
        angle_deg = np.degrees(angle)
        visual_top = POLAR_BOTTOM_VAL + POLAR_GAP + percent
        text_radius = visual_top - percent * 0.12
        rotation = - angle_deg if 0 <= angle_deg < 180 else 180 - angle_deg
        ax2.text(angle, text_radius, f"{raw_val:.3f}",
                 ha='center', va='center',
                 rotation=rotation, rotation_mode='anchor',
                 fontsize=9, fontweight='bold', color='white')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = ax2.inset_axes([0.48, 0.40, 0.04, 0.13], transform=ax2.transAxes)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('SHAP Value', fontsize=8, fontweight='bold', family='Arial', rotation=0)
    cbar.ax.yaxis.set_label_coords(0.5, 1.3)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.tick_params(labelsize=12)
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
# 15. 生成所有SHAP可视化
# ==========================================
print("\n" + "=" * 80)
print("步骤 5: 生成SHAP可视化")
print("=" * 80)

models_to_plot = [
    (name, shap_vals, imp_df, top_feats)
    for name, (shap_vals, imp_df, top_feats) in shap_results.items()
]

generated_files = []
for model_name, shap_vals, importance_df, top_features in models_to_plot:
    print(f"\n{model_name} 模型可视化:")
    for scheme_name in COLOR_SCHEMES.keys():
        output_file = plot_shap_barplot_with_rose(shap_vals, X_test_desc_df, top_features,
                                                  model_name, scheme_name)
        generated_files.append(output_file)
        print(f"    ✓ {scheme_name}: {os.path.basename(output_file)}")
        output_file = plot_shap_beeswarm(shap_vals, X_test_desc_df, top_features,
                                         model_name, scheme_name)
        generated_files.append(output_file)
        print(f"    ✓ {scheme_name}: {os.path.basename(output_file)}")
        output_file = plot_shap_violin(shap_vals, X_test_desc_df, top_features,
                                       model_name, scheme_name)
        generated_files.append(output_file)
        print(f"    ✓ {scheme_name}: {os.path.basename(output_file)}")
        output_file = plot_shap_beeswarm_rose_combined(
            shap_vals, X_test_desc_df, importance_df, model_name,
            top_n=TOP_N_FEATURES, color_scheme=scheme_name, output_dir=OUTPUT_DIR
        )
        generated_files.append(output_file)
        print(f"    ✓ {scheme_name}: {os.path.basename(output_file)}")

# ==========================================
# 16. 生成汇总报告
# ==========================================
print("\n" + "=" * 80)
print("步骤 6: 生成分析报告")
print("=" * 80)

n_models = len(models_to_plot)
n_schemes = len(COLOR_SCHEMES)
n_bar_rose = n_models * n_schemes
n_beeswarm = n_models * n_schemes
n_violin = n_models * n_schemes
n_beeswarm_rose = n_models * n_schemes
total_viz = n_bar_rose + n_beeswarm + n_violin + n_beeswarm_rose

summary_report = f"""
{'=' * 80}
SHAP分析报告 - 从已保存模型加载
数据文件: AR.csv
{'=' * 80}

数据集信息:
- 原始样本数: {len(df)}
- 有效样本数: {len(X)}
- 训练集: {len(X_train_sub)} 条 (85%)
- 测试集: {len(X_test)} 条 (15%)
- 总特征维度: {X.shape[1]} (指纹: {n_fp}, 描述符: {n_desc})
- 分析的描述符数: {X_test_desc.shape[1]}
- 选取Top特征数: {TOP_N_FEATURES}

模型性能 (测试集):
{'=' * 80}
"""
for name, model in loaded_models.items():
    pred_test = model.predict(X_test)
    r2 = r2_score(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mae = mean_absolute_error(y_test, pred_test)
    summary_report += f"""
{name}:
   - R² Score: {r2:.4f}
   - RMSE: {rmse:.4f}
   - MAE: {mae:.4f}
"""

summary_report += f"""
输出文件:
{'=' * 80}
数据文件:
- feature_importance_all.csv: 所有描述符的SHAP重要性
- top{TOP_N_FEATURES}_features.csv: Top {TOP_N_FEATURES} 描述符对比

可视化文件 (每个模型x每个配色方案):
- 条形图+玫瑰图组合: {n_bar_rose}个文件
- 蜂窝图(单独): {n_beeswarm}个文件
- 小提琴图: {n_violin}个文件
- 蜂巢图+玫瑰图组合(新增): {n_beeswarm_rose}个文件
- 总共: {total_viz}个可视化文件

配色方案说明:
{'=' * 80}
- viridis: 绿-黄渐变 (高对比度,色盲友好)
- plasma: 紫-橙渐变 (鲜艳,高可见性)
- coolwarm: 蓝-红渐变 (经典科学配色)
- RdYlBu: 红-黄-蓝 (三色渐变)
- RdBu_r: 红-蓝反向 (双色对比)

输出目录: {OUTPUT_DIR}
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""

report_path = os.path.join(OUTPUT_DIR, "SHAP_Analysis_Report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(summary_report)
print(f"✓ 分析报告已保存: {report_path}")

print("\n" + summary_report)
print("\n" + "=" * 80)
print("✅ 所有任务完成!")
print("=" * 80)
print(f"\n📁 所有文件已保存到: {OUTPUT_DIR}")
print(f"\n📊 共生成 {total_viz + 2} 个文件:")
print(f"   - {total_viz} 个可视化图表")
print(f"   - 2 个数据/报告文件")
print("\n📊 可视化文件分类:")
print(f"   - 条形图+玫瑰图组合: {n_bar_rose}个")
print(f"   - 蜂窝图(单独): {n_beeswarm}个")
print(f"   - 小提琴图: {n_violin}个")
print(f"   - 蜂巢图+玫瑰图组合(新增): {n_beeswarm_rose}个")
print("\n" + "=" * 80)