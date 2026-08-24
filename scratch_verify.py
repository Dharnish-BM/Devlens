import sys
sys.path.append('d:/GIT/Devlens')
from devlens.db.session import get_session
from devlens.db.repository import get_all_current_features
from devlens.features.feature_engineering import prepare_scaled_feature_matrix
import pandas as pd

with get_session() as session:
    df = get_all_current_features(session)
    
if not df.empty:
    scaled_df, scaler, dropped_cols = prepare_scaled_feature_matrix(df, drop_zero_variance=True)
    print(f'Original features: {df.shape[1]}')
    print(f'Scaled features: {scaled_df.shape[1]}')
    print(f'Dropped features: {dropped_cols}')
    print('\nFinal Feature List:')
    for col in sorted(scaled_df.columns):
        print(f'- {col}')
else:
    print('DF is empty')
