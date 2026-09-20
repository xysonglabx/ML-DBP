import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, Crippen, Lipinski, MolSurf, rdMolDescriptors
from rdkit.Chem import MACCSkeys, GraphDescriptors
# 移除 EStateFingerprinter 的导入，将在函数内部动态导入
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
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
print("R² > 0.9 挑战 - 增强版模型 (含特征工程 + 共线性筛选 + 三方法投票选择 + 损失曲线)")
print("=" * 80)


# ==========================================
# 1. 增强特征提取（包含所有新增描述符）
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
# 2. 辅助函数（计算吸附特征，原代码）
# ==========================================
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
                'Chi2n': Descriptors.Chi2n(mol),
                'Chi2v': Descriptors.Chi2v(mol),
                'Chi3n': Descriptors.Chi3n(mol),
                'Chi3v': Descriptors.Chi3v(mol),
                'Chi4n': Descriptors.Chi4n(mol),
                'Chi4v': Descriptors.Chi4v(mol),
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
# 3. 特征选择工具函数（低方差、相关性、VIF、热图）
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
# 4. 主程序
# ==========================================
print("\n正在加载数据...")
df = pd.read_csv("AR.csv")
df.columns = ['smiles', 'AR']
print(f"原始数据: {len(df)} 条")

print("正在提取增强特征 (扩展描述符 + 多重指纹 + 3D/药效团等)...")
X, valid_idx, valid_smiles, n_desc = extract_enhanced_features(df['smiles'])

if X is None or len(valid_idx) == 0:
    print("错误：没有有效的分子，无法继续。请检查 SMILES 格式。")
    exit(1)

y = df['AR'].iloc[valid_idx].values
print(f"有效数据: {len(X)} 条")
print(f"特征维度: {X.shape[1]} (指纹: {X.shape[1] - n_desc}, 描述符: {n_desc})")

print("\n计算吸附相关分子特征...")
adsorption_features = calculate_adsorption_features(df['smiles'].tolist())
valid_adsorption_features = adsorption_features.iloc[valid_idx].reset_index(drop=True)

# ==========================================
# 5. 描述符共线性筛选
# ==========================================
print("\n" + "=" * 80)
print("描述符共线性筛选 (低方差 + 相关性过滤 + VIF诊断)")
print("=" * 80)

output_dir = "ml_model_ARmodel_AR5_pt"
os.makedirs(output_dir, exist_ok=True)

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
    'ChargeRange', 'ChargeIQR',   # 新增
    'AromaticRingRatio', 'SaturatedRingRatio',
    'NumSaturatedHeterocycles', 'NumUnsaturatedHeterocycles', 'NumSaturatedCarbocycles', 'NumUnsaturatedCarbocycles',
    'fr_Al_COO', 'fr_Al_OH', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_S',
    'fr_HOCC', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_alkyl_halide', 'fr_halogen',
    'GraphDiameter', 'GraphRadius',
    'MolVolume', 'MolSurfaceArea', 'InertiaX', 'InertiaY', 'InertiaZ', 'Asphericity', 'Sphericity',
    'MaxDistance', 'MaxDistanceSq',   # 新增
    'HBondPairs', 'HBondAvgDist', 'HBondMinDist',   # 新增
    'SASA',   # 新增
    'ESP_Max', 'ESP_Min', 'ESP_Mean', 'ESP_Std',   # 新增
    *[f'USR_{i}' for i in range(12)],
    *[f'Pharm2D_{i}' for i in range(128)],
    'LogP_TPSA', 'LogP_Sq', 'TPSA_Sq',   # 新增 LogP_Sq 和 TPSA_Sq
    'HB_Ratio', 'RotBond_Ratio', 'MR_Density',
    'ChiralCenters', 'AvgAromaticRingSize',
]

# 检查长度
assert len(descriptor_names) == n_desc, f"描述符名称数量({len(descriptor_names)})与实际描述符维度({n_desc})不匹配！"

print(f"\n原始描述符数量: {n_desc}")

# Step 1: 低方差过滤
lv_mask, lv_removed = remove_low_variance_features(X_desc_part, descriptor_names, threshold=0.01)
X_desc_filtered = X_desc_part[:, lv_mask]
names_after_lv = [descriptor_names[i] for i in range(n_desc) if lv_mask[i]]
print(f"  低方差过滤后描述符数量: {len(names_after_lv)}")

# Step 2: 相关性过滤
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

# Step 4: 相关性热图
print("\n  绘制过滤后描述符相关性热图...")
plot_descriptor_correlation_heatmap(X_desc_clean, final_descriptor_names, output_dir=output_dir)

print(f"\n  特征筛选完成:")
print(f"    原始总维度: {X.shape[1]} (指纹 {n_fp} + 描述符 {n_desc})")
print(f"    过滤后描述符维度: {len(final_descriptor_names)}")
print(f"    移除描述符总数: {n_desc - len(final_descriptor_names)}")

# ==========================================
# 6. 特征工程
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
# 8. RobustScaler特征缩放
# ==========================================
print("\n使用 RobustScaler 进行特征缩放...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_final)
X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
print(f"  ✓ 缩放完成，最终输入特征维度: {X_scaled.shape[1]}")

# ==========================================
# 9. 数据划分
# ==========================================
best_seed = 42
print("\n使用固定划分种子: 42")
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_scaled, y, range(len(y)), test_size=0.15, random_state=best_seed
)

# ====== 修改部分：将索引转换为 numpy 数组，并构建原始索引到 valid_smiles 位置的映射 ======
train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

# 构建原始索引到 valid_smiles 位置的映射
orig_to_valid_pos = {orig_idx: pos for pos, orig_idx in enumerate(valid_idx)}

X_train_sub, X_val, y_train_sub, y_val, train_sub_indices, val_indices = train_test_split(
    X_train, y_train, train_indices, test_size=0.15, random_state=42
)

# 注意：train_sub_indices 和 val_indices 已经是原始数据索引（因为传入的 indices 是 train_indices 原始索引）
# 无需再次转换，直接使用即可
train_sub_indices = np.array(train_sub_indices)
val_indices = np.array(val_indices)

print(f"训练集: {len(X_train_sub)} 条")
print(f"验证集: {len(X_val)} 条")
print(f"测试集: {len(X_test)} 条")

# ==========================================
# 10. 构建集成模型
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
    X_train, y_train,
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
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
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
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
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
gb_model.fit(X_train, y_train)
models['GradientBoosting'] = gb_model
gb_train_scores, gb_val_scores = [], []
for train_pred, val_pred in zip(gb_model.staged_predict(X_train), gb_model.staged_predict(X_val)):
    gb_train_scores.append(mean_squared_error(y_train, train_pred))
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
rf_model.fit(X_train, y_train)
models['RandomForest'] = rf_model
print("  RandomForest训练完成 (无迭代损失)")

# ==========================================
# 11. 保存损失曲线数据
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
vt_file = os.path.join(models_dir, "fp_variance_threshold.pkl")
joblib.dump(vt, vt_file)
print(f"✓ 指纹方差过滤器已保存")

# ==========================================
# 16. 保存预测数据到 Excel (SMILES + 真实值 + 各模型预测值)
# ==========================================
print("\n生成预测数据 Excel 文件...")

# 使用映射获取 valid_smiles 中的位置索引
train_positions = [orig_to_valid_pos[idx] for idx in train_sub_indices]
val_positions   = [orig_to_valid_pos[idx] for idx in val_indices]
test_positions  = [orig_to_valid_pos[idx] for idx in test_indices]

# 收集训练集数据
train_smiles = [valid_smiles[i] for i in train_positions]
train_true = y_train_sub
train_pred_dict = {name: all_predictions['train'][name] for name in models.keys()}
train_df = pd.DataFrame({'SMILES': train_smiles, 'True_AR': train_true})
for name, pred in train_pred_dict.items():
    train_df[f'Pred_{name}'] = pred

# 收集验证集数据
val_smiles = [valid_smiles[i] for i in val_positions]
val_true = y_val
val_pred_dict = {name: all_predictions['val'][name] for name in models.keys()}
val_df = pd.DataFrame({'SMILES': val_smiles, 'True_AR': val_true})
for name, pred in val_pred_dict.items():
    val_df[f'Pred_{name}'] = pred

# 收集测试集数据
test_smiles = [valid_smiles[i] for i in test_positions]
test_true = y_test
test_pred_dict = {name: all_predictions['test'][name] for name in models.keys()}
test_df = pd.DataFrame({'SMILES': test_smiles, 'True_AR': test_true})
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