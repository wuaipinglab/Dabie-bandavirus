 
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.preprocessing import StandardScaler
from Bio.PDB import PDBParser

# 导入自定义模块
from weighted_contact_number import *
from seq_utils import *

 
import Bio
print(f"Current Biopython version: {Bio.__version__}")

# 如果版本 >= 1.80，建议降级到1.79
 
# 氨基酸性质表
aa_charge_hydro = '../data/aa_properties/dissimilarity_metrics.csv'

# EVE预测文件
h1_eve = '../results/evol_indices/Bunya_20000_samples.csv'

# PDB结构信息
h1_pdb_id = '8ilq'
h1_pdb_path = '../data/structures/8ilq.pdb'
h1_chains = ['A', 'B']
h1_trimer_chains = ['A', 'B']

# 目标序列
h1_target_seq_path = '../data/sequences/M_ref.fa'
print(h1_target_seq_path)

 
# def process_eve_smm(eve_path):
eve = pd.read_csv(h1_eve)
eve = eve[1:]  # 去掉第一行（可能是冗余的 header）
# eve
eve.columns = eve.columns.str.replace("_ensemble", "")
# eve.columns
eve['wt'] = eve.mutations.str[0]
eve['mut'] = eve.mutations.str[-1]

eve['i'] = eve.mutations.str[1:-1].astype(int)

eve['evol_indices'] = -eve.evol_indices  # EVE越低表示越不利
to_drop = ['protein_name', 'mutations']
to_drop.extend([col for col in eve.columns if "semantic_change" in col])
eve = eve.drop(columns=to_drop)


 
eve
 
def process_eve_smm(eve_path):
    eve = pd.read_csv(eve_path)
    eve = eve[1:]  # 去掉第一行（可能是冗余的 header）
    eve.columns = eve.columns.str.replace("_ensemble", "")
    eve['wt'] = eve.mutations.str[0]
    eve['mut'] = eve.mutations.str[-1]
    eve['i'] = eve.mutations.str[1:-1].astype(int)
    eve['evol_indices'] = -eve.evol_indices  # EVE越低表示越不利
    to_drop = ['protein_name', 'mutations']
    to_drop.extend([col for col in eve.columns if "semantic_change" in col])
    eve = eve.drop(columns=to_drop)
    return eve

 
def add_model_outputs(exps, eve_path):
    exps = exps.merge(process_eve_smm(eve_path),
                      on=['wt', 'mut', 'i'],
                      how='outer')
    return exps

 
def get_wcn(exps, pdb_path, trimer_chains, target_chains, map_table):
    wcn = add_wcn_to_site_annotations(pdb_path, ''.join(trimer_chains))
    wcn = wcn.rename(columns={'pdb_position': 'i', 'pdb_aa': 'wt'})
    wcn['i'] = wcn.i.apply(lambda x: alphanumeric_index_to_numeric_index(x)
                           if (x != '') else x)
    wcn['i'] = wcn.i.replace('', np.nan)
    wcn = remap_struct_df_to_target_seq(wcn, target_chains, map_table)

    exps = exps.merge(wcn[['i', 'wcn_sc']], how='left', on='i')
    exps = exps.sort_values('i')
    exps['wcn_bfil'] = exps.wcn_sc.fillna(method='bfill')
    exps['wcn_ffil'] = exps.wcn_sc.fillna(method='ffill')
    exps['wcn_fill'] = (
        exps[['wcn_ffil', 'wcn_bfil']].sum(axis=1, min_count=2) / 2)
    exps = exps.drop(columns=['wcn_bfil', 'wcn_ffil'])
    return exps

 
def hydrophobicity_charge(exps, table):
    props = pd.read_csv(table, index_col=0)

    scale = StandardScaler()
    props['eisenberg_weiss_diff_std'] = scale.fit_transform(
        props['eisenberg_weiss_diff'].abs().values.reshape(-1, 1))
    props['charge_diff_std'] = scale.fit_transform(
        props['charge_diff'].abs().values.reshape(-1, 1))
    exps = exps.merge(props, how='left', on=['wt', 'mut'])

    exps['charge_ew-hydro'] = exps[[
        'eisenberg_weiss_diff_std', 'charge_diff_std'
    ]].sum(axis=1)
    exps = exps.drop(columns=['eisenberg_weiss_diff_std', 'charge_diff_std'])
    return exps

 
def norm_to_wt(df, prefvar):
    newvar = 'norm_' + prefvar

    def grp_func(grp):
        ref = grp[grp['wt'] == grp['mut']][prefvar].mean()
        grp[newvar] = grp[prefvar] / ref
        return grp

    df[newvar] = df[prefvar]
    df = df.groupby(['i', 'wt']).apply(grp_func)
    return df

 
def load_H1():
    # 初始化突变表
    data = make_mut_table(h1_target_seq_path)

    # 添加模型得分
    data = add_model_outputs(data, h1_eve)

    # 去掉 wildtype 自己和自己的比对
    data = data[data.wt != data.mut]

    # 建立结构序列映射表
    map_table = remap_pdb_seq_to_target_seq(h1_pdb_path, h1_chains,
                                            h1_target_seq_path)

    # 添加 WCN 特征
    data = get_wcn(data, h1_pdb_path, h1_trimer_chains, h1_chains, map_table)

    # 添加氨基酸理化性质特征
    data = hydrophobicity_charge(data, aa_charge_hydro)
    data = data.sort_values(['i', 'mut'])

    return data, map_table

 
h1, _ = load_H1()
h1.to_csv('../results/bunya_scores.csv', index=False)
 

 
