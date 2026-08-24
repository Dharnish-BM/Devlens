import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('devlens.db')
query = '''
SELECT d.username, f.feature_name, f.feature_value
FROM features f
JOIN snapshots s ON f.snapshot_id = s.id
JOIN developers d ON s.developer_id = d.id
WHERE s.id IN (
    SELECT MAX(id)
    FROM snapshots
    GROUP BY developer_id
)
'''
df = pd.read_sql_query(query, conn)
conn.close()

# Pivot
df_pivot = df.pivot(index='username', columns='feature_name', values='feature_value')

# Calculate stats
stats = df_pivot.describe().T
stats['median'] = df_pivot.median()

def calc_default_pct(col):
    return ((col == 0.0) | col.isna()).mean() * 100

stats['pct_default_or_null'] = df_pivot.apply(calc_default_pct)

def check_nzv(col):
    val_counts = col.value_counts(normalize=True, dropna=False)
    if not val_counts.empty and val_counts.iloc[0] > 0.90:
        return True
    return False

stats['nzv_flag'] = df_pivot.apply(check_nzv)

z_scores = (df_pivot - df_pivot.mean()) / df_pivot.std()
outliers = z_scores > 3
outlier_counts = outliers.sum(axis=1)
flagged_users = outlier_counts[outlier_counts >= 3]

with open('scratch_dq_report.md', 'w', encoding='utf-8') as f:
    f.write("# DevLens Data Quality Report\n\n")
    f.write("## Feature Statistics\n")
    f.write("| Feature | Min | Max | Mean | Std | Median | % Default/Null | NZV Flag |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for feature in stats.index:
        row = stats.loc[feature]
        flag = "🚩 YES" if row['nzv_flag'] else "No"
        f.write(f"| {feature} | {row['min']:.4f} | {row['max']:.4f} | {row['mean']:.4f} | {row['std']:.4f} | {row['median']:.4f} | {row['pct_default_or_null']:.1f}% | {flag} |\n")

    f.write("\n## Near-Zero Variance Features (>90% identical values)\n")
    nzv_features = stats[stats['nzv_flag']].index.tolist()
    if nzv_features:
        for feat in nzv_features:
            f.write(f"- **{feat}**\n")
    else:
        f.write("- None detected.\n")

    f.write("\n## Outlier Users (Flagged on 3+ Features)\n")
    if not flagged_users.empty:
        for user, count in flagged_users.items():
            user_outliers = outliers.loc[user]
            flagged_feats = user_outliers[user_outliers].index.tolist()
            f.write(f"- **{user}**: {count} features\n")
            f.write(f"  - Features: {', '.join(flagged_feats)}\n")
    else:
        f.write("- None detected.\n")
