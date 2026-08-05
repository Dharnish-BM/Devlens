import sqlite3

conn = sqlite3.connect("devlens.db")
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]
print("TABLES:", tables)
print()

for table in tables:
    cur.execute(f"PRAGMA table_info({table})")
    cols = cur.fetchall()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    print(f"[{table}]  ({count} rows)")
    for col in cols:
        pk      = " PK"       if col[5] else ""
        nn      = " NOT NULL" if col[3] else ""
        default = f" DEFAULT {col[4]}" if col[4] else ""
        print(f"  {str(col[1]):<35} {col[2]}{pk}{nn}{default}")

    cur.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=?",
        (table,)
    )
    for idx in cur.fetchall():
        if idx[1]:
            print(f"  INDEX: {idx[0]}")
    print()

cur.execute("SELECT name, sql FROM sqlite_master WHERE type='index' ORDER BY name")
all_indexes = cur.fetchall()
print(f"Total indexes: {len(all_indexes)}")

conn.close()
