import pickle
from sklearn.metrics import classification_report
import numpy as np

with open("/Users/nithyashreevsuresh/Downloads/st/predictive_insights/experiments/directional_study_results.pkl", "rb") as f:
    results = pickle.load(f)

exp_map = {r["id"]: r for r in results}

target_ids = [
    "EXP0_ALWAYS_UP",
    "EXP1_CATBOOST_REG_DEFAULT",
    "EXP2_CATBOOST_REG_TUNED_TAU",
    "EXP4_CATBOOST_CLF_BALANCED",
    "EXP7_CATBOOST_CLF_TUNED_PROB",
    "EXP10_RANDOM_FOREST_CLF_UNWEIGHTED",
    "EXP13_CATBOOST_CLF_EXPANDED_BALANCED"
]

for eid in target_ids:
    r = exp_map[eid]
    p = r["pooled"]
    cm = p["Confusion_Matrix"]
    name = r["name"]
    print("=" * 80)
    print(f"REPORT FOR: {name} ({eid})")
    print("=" * 80)
    print(f"Accuracy: {p['Accuracy']:.4f}% | Balanced Accuracy: {p['Balanced_Accuracy']:.4f}% | Macro F1: {p['Macro_F1']:.4f}%")
    print(f"UP Class:   Precision = {p['UP_Precision']:.4f}%, Recall = {p['UP_Recall']:.4f}%, F1 = {p['UP_F1']:.4f}%, Support = {p['UP_Support']}")
    print(f"DOWN Class: Precision = {p['DOWN_Precision']:.4f}%, Recall = {p['DOWN_Recall']:.4f}%, F1 = {p['DOWN_F1']:.4f}%, Support = {p['DOWN_Support']}")
    print(f"Confusion Matrix: [[{cm['TN']}, {cm['FP']}], [{cm['FN']}, {cm['TP']}]]\n")
