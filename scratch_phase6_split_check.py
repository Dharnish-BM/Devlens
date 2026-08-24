import sys
sys.path.append('d:/GIT/Devlens')
from devlens.db.session import get_session
from devlens.db.repository import get_all_current_features
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

with get_session() as session:
    df = get_all_current_features(session)

labels = []
for username, row in df.iterrows():
    ml_sig = row.get("lang_ml_signal", 0.0)
    mobile_sig = row.get("lang_mobile_signal", 0.0)
    devops_sig = row.get("lang_devops_signal", 0.0)
    frontend_sig = row.get("lang_frontend_signal", 0.0)
    primary_js = row.get("lang_primary_is_javascript", 0.0) == 1.0
    primary_c = row.get("lang_primary_is_c", 0.0) == 1.0
    primary_java_kotlin = row.get("lang_primary_is_java_kotlin", 0.0) == 1.0
    entropy = row.get("lang_diversity_entropy", 0.0)

    if ml_sig > 0.05:
        lbl = "ML Specialist"
    elif mobile_sig > 0.02:
        lbl = "Mobile Developer"
    elif devops_sig > 0.10:
        lbl = "DevOps Engineer"
    elif frontend_sig > 0.50 or primary_js:
        lbl = "Frontend Developer"
    elif primary_c or (primary_java_kotlin and mobile_sig <= 0.02):
        lbl = "Backend Developer"
    elif entropy >= 1.8:
        lbl = "Full-Stack Developer"
    else:
        lbl = "Unclassified / Low Signal"
    
    labels.append(lbl)

df["archetype_label"] = labels

# Exclude Backend Developer (n=2) and Full-Stack Developer (n=1)
excluded_classes = ["Backend Developer", "Full-Stack Developer"]
train_df = df[~df["archetype_label"].isin(excluded_classes)].copy()

print(f"Total Cohort: {len(df)}")
print(f"Excluded from Training ({', '.join(excluded_classes)}): {len(df) - len(train_df)}")
print(f"Training Set Size: {len(train_df)}")

# Evaluate 80/20 and 75/25 Stratified Train/Test Splits
for test_ratio in [0.20, 0.25]:
    X = train_df.drop(columns=["archetype_label"])
    y = train_df["archetype_label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )
    
    print("\n" + "="*70)
    print(f"Stratified Train/Test Split (test_size={test_ratio*100:.0f}%, random_state=42)")
    print("="*70)
    print(f"{'Class Name':<30} {'Total':<10} {'Train Count':<12} {'Test Count':<12} {'Test %'}")
    print("-" * 70)
    
    classes_sorted = y.value_counts().index
    for cls in classes_sorted:
        tot = (y == cls).sum()
        tr_c = (y_train == cls).sum()
        te_c = (y_test == cls).sum()
        pct = te_c / tot * 100
        print(f"{cls:<30} {tot:<10} {tr_c:<12} {te_c:<12} {pct:.1f}%")
    
    print("-" * 70)
    print(f"{'TOTAL':<30} {len(y):<10} {len(y_train):<12} {len(y_test):<12}")
    
    # Check Unclassified / Low Signal test count
    low_sig_test = (y_test == "Unclassified / Low Signal").sum()
    mobile_test = (y_test == "Mobile Developer").sum()
    if low_sig_test < 4:
        print(f"WARNING: Unclassified/Low Signal has only {low_sig_test} test example(s) (< 3-4 examples). Precision/recall will be statistically unstable!")
    if mobile_test < 4:
        print(f"WARNING: Mobile Developer has only {mobile_test} test example(s) (< 3-4 examples).")
