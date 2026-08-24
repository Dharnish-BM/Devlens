import sys
sys.path.append('d:/GIT/Devlens')
from devlens.db.session import get_session
from devlens.db.repository import get_all_current_features
import pandas as pd

with get_session() as session:
    df = get_all_current_features(session)

# Fallback filter criteria:
ml_mask = df['lang_ml_signal'] > 0.05
mobile_mask = df['lang_mobile_signal'] > 0.02
devops_mask = df['lang_devops_signal'] > 0.10
frontend_mask = (df['lang_primary_is_javascript'] == 1.0) | (df['lang_frontend_signal'] > 0.50)
backend_mask = (df['lang_primary_is_c'] == 1.0) | (df['lang_primary_is_java_kotlin'] == 1.0)

categorized_mask = ml_mask | mobile_mask | devops_mask | frontend_mask | backend_mask
fallback_df = df[~categorized_mask]

print(f'Total cohort: {len(df)}')
print(f'Categorized by domain/primary language: {categorized_mask.sum()}')
print(f'Fallback count (uncategorized by primary/domain rules): {len(fallback_df)}')

print('\nFallback Entropy Distribution:')
print(fallback_df['lang_diversity_entropy'].describe())

high_entropy = fallback_df[fallback_df['lang_diversity_entropy'] >= 1.8]
low_entropy = fallback_df[fallback_df['lang_diversity_entropy'] < 1.8]

print(f'\nFallback with Entropy >= 1.8 (Genuine Full-Stack/Polyglot): {len(high_entropy)}')
print(f'Fallback with Entropy < 1.8 (Low Entropy / Unclassifiable): {len(low_entropy)}')

print('\nLow Entropy Fallback Details:')
for idx, row in low_entropy.iterrows():
    print(f"{idx}: entropy={row['lang_diversity_entropy']:.4f}, primary_py={row['lang_primary_is_python']}, total_commits={row['activity_total_commits']}")
