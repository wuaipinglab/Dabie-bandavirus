import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.impute import SimpleImputer

temperatures = {"fitness": 1, "surfacc": 1, "exchangability": 2}
# 读取突变评分数据
sftsv = pd.read_csv('../results/bunya_scores.csv')


# # 找出包含突变存活分数的抗体列
# flu_ablist = [col for col in flu.columns.values if "mutfracsurvive" in col]
# flu_all_ab = flu_ablist.copy()  # 后续用于删除

def logistic(x):
    return 1 / (1 + np.exp(-x))


def standardization(x):
    """Assumes input is numpy array or pandas series"""
    return (x - x.mean()) / x.std()


def make_predictors(summary_init, thresh, ablist, scores=True):
    summary = summary_init.copy()

    #Drop extraneous WCN columns
    summary = summary.drop(
        columns=[col for col in summary.columns if "wcn_fill_" in col])
    summary = summary.drop(
        columns=[col for col in summary.columns if "wcn_sc" in col])
    summary = summary.drop(
        columns=[col for col in summary.columns if "diff" in col])

    #Reverse WCN direction so that larger values are more accessible
    summary["wcn_fill_r"] = -summary.wcn_fill
    summary = summary.drop(columns="wcn_fill")

    if scores:
        #Calculate max escape for each mutant
        summary["max_escape_experiment"] = summary[ablist].max(axis=1)
        #Calculate if escape>threshold for each mutant
        summary[
            "is_escape_experiment"] = summary["max_escape_experiment"] > thresh

    #Impute missing values for columns used to calculate EVEscape scores
    impute_cols = ["i", "evol_indices", "wcn_fill_r", "charge_ew-hydro"]

    df_imp = summary[impute_cols].copy()
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    df_imp = pd.DataFrame(imp.fit_transform(df_imp),
                          columns=df_imp.columns,
                          index=df_imp.index)
    df_imp = pd.concat([df_imp, summary[["wt", "mut"]]], axis=1)

    #Compute EVEscape scores
    summary["evescape"] = 0
    summary["evescape"] += np.log(
        logistic(
            standardization(df_imp["evol_indices"]) * 1 /
            temperatures["fitness"]))
    summary["evescape"] += np.log(
        logistic(
            standardization(df_imp["wcn_fill_r"]) * 1 /
            temperatures["surfacc"]))
    summary["evescape"] += np.log(
        logistic(
            standardization(df_imp["charge_ew-hydro"]) * 1 /
            temperatures["exchangability"]))

    summary = summary.drop(
        columns=[col for col in summary.columns if col == "wcn_fill"])

    summary = summary.rename(
        columns={
            "evol_indices": "fitness_eve",
            "wcn_fill_r": "accessibility_wcn",
            "charge_ew-hydro": "dissimilarity_charge_hydro"
        })

    summary = summary.round(decimals=7)

    return (summary)


sftsv = make_predictors(sftsv, None, None, scores=False)

sftsv.to_csv('../results/bunya_evescape.csv', index=False)


def make_site(summary_init):
    summary = summary_init.copy()
    summary = summary.groupby(['i', 'wt']).agg('mean').reset_index()

    return (summary)


sftsv_site = make_site(sftsv)

sftsv_site.to_csv('../results/bunya_evescape_site.csv', index=False)

