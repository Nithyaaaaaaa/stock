import pickle
import numpy as np
import pandas as pd

with open("/Users/nithyashreevsuresh/Downloads/st/predictive_insights/experiments/directional_study_results.pkl", "rb") as f:
    results = pickle.load(f)

print("=" * 130)
print(f"{'Experiment Name':<42} | {'Acc':<6} | {'BalAcc':<6} | {'MacF1':<6} | {'UP Prec':<7} | {'UP Rec':<7} | {'UP F1':<6} | {'DN Prec':<7} | {'DN Rec':<7} | {'DN F1':<6}")
print("=" * 130)

rows = []
for r in results:
    p = r["pooled"]
    rows.append({
        "ID": r["id"],
        "Name": r["name"],
        "Accuracy": p["Accuracy"],
        "Balanced_Accuracy": p["Balanced_Accuracy"],
        "Macro_F1": p["Macro_F1"],
        "Weighted_F1": p["Weighted_F1"],
        "UP_Precision": p["UP_Precision"],
        "UP_Recall": p["UP_Recall"],
        "UP_F1": p["UP_F1"],
        "UP_Support": p["UP_Support"],
        "DOWN_Precision": p["DOWN_Precision"],
        "DOWN_Recall": p["DOWN_Recall"],
        "DOWN_F1": p["DOWN_F1"],
        "DOWN_Support": p["DOWN_Support"],
        "TN": p["Confusion_Matrix"]["TN"],
        "FP": p["Confusion_Matrix"]["FP"],
        "FN": p["Confusion_Matrix"]["FN"],
        "TP": p["Confusion_Matrix"]["TP"],
        "Mean_Fold_Acc": r["mean_fold_acc"],
        "Mean_Fold_BalAcc": r["mean_fold_bal_acc"],
        "Mean_Fold_MacroF1": r["mean_fold_macro_f1"],
        "Mean_Fold_UP_F1": r["mean_fold_up_f1"],
        "Mean_Fold_DOWN_F1": r["mean_fold_down_f1"],
        "Mean_Fold_UP_Rec": r["mean_fold_up_rec"],
        "Mean_Fold_DOWN_Rec": r["mean_fold_down_rec"],
    })
    print(f"{r['name'][:42]:<42} | {p['Accuracy']:5.2f}% | {p['Balanced_Accuracy']:5.2f}% | {p['Macro_F1']:5.2f}% | {p['UP_Precision']:6.2f}% | {p['UP_Recall']:6.2f}% | {p['UP_F1']:5.2f}% | {p['DOWN_Precision']:6.2f}% | {p['DOWN_Recall']:6.2f}% | {p['DOWN_F1']:5.2f}%")

df_res = pd.DataFrame(rows)
df_res.to_csv("/Users/nithyashreevsuresh/Downloads/st/predictive_insights/experiments/directional_study_summary.csv", index=False)

print("\n" + "=" * 130)
print("DETAILED FOLD-BY-FOLD BREAKDOWN FOR TOP CANDIDATES")
print("=" * 130)

top_ids = [
    "EXP0_ALWAYS_UP",
    "EXP1_CATBOOST_REG_DEFAULT",
    "EXP2_CATBOOST_REG_TUNED_TAU",
    "EXP4_CATBOOST_CLF_BALANCED",
    "EXP7_CATBOOST_CLF_TUNED_PROB",
    "EXP10_RANDOM_FOREST_CLF_UNWEIGHTED",
    "EXP13_CATBOOST_CLF_EXPANDED_BALANCED"
]

for r in results:
    if r["id"] in top_ids:
        print(f"\n--- {r['id']}: {r['name']} ---")
        if "tuned_thresholds" in r and r["tuned_thresholds"]:
            print(f"Tuned Thresholds across Folds 1-5: {[round(t, 5) for t in r['tuned_thresholds']]}")
        print(f"{'Fold':<5} | {'Acc':<6} | {'BalAcc':<6} | {'MacF1':<6} | {'UP Prec':<7} | {'UP Rec':<7} | {'UP F1':<6} | {'DN Prec':<7} | {'DN Rec':<7} | {'DN F1':<6} | {'CM (TN,FP,FN,TP)'}")
        print("-" * 110)
        for fm in r["fold_metrics"]:
            cm = fm["Confusion_Matrix"]
            print(f"Fold {fm['fold']:<1} | {fm['Accuracy']:5.2f}% | {fm['Balanced_Accuracy']:5.2f}% | {fm['Macro_F1']:5.2f}% | {fm['UP_Precision']:6.2f}% | {fm['UP_Recall']:6.2f}% | {fm['UP_F1']:5.2f}% | {fm['DOWN_Precision']:6.2f}% | {fm['DOWN_Recall']:6.2f}% | {fm['DOWN_F1']:5.2f}% | ({cm['TN']:<3}, {cm['FP']:<3}, {cm['FN']:<3}, {cm['TP']:<3})")
