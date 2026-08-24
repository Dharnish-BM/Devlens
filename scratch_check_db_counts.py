import sys
sys.path.append('d:/GIT/Devlens')
from devlens.db.session import get_session
from devlens.db.models import ArchetypePrediction
import pandas as pd

with get_session() as session:
    rows = session.query(ArchetypePrediction.archetype_label).all()
    df = pd.DataFrame(rows, columns=["label"])

print("=== ACTUAL STORED DATABASE COUNTS IN archetype_predictions (188 TOTAL) ===")
print(df["label"].value_counts())
