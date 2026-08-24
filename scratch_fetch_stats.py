import sqlite3
import pandas as pd

conn = sqlite3.connect('devlens.db')
query = '''
SELECT f.feature_name, f.feature_value
FROM features f
JOIN snapshots s ON f.snapshot_id = s.id
JOIN developers d ON s.developer_id = d.id
WHERE d.username = 'Sabari-Vasan-SM'
  AND f.feature_name IN ('repo_mean_size_kb', 'repo_std_size_kb')
  AND s.id = (SELECT MAX(id) FROM snapshots WHERE developer_id = d.id)
'''
df = pd.read_sql_query(query, conn)
for index, row in df.iterrows():
    print(f"{row['feature_name']}: {row['feature_value']}")
conn.close()
