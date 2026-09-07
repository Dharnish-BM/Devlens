import time
import requests
import json
import sqlite3
from datetime import datetime

# 1. Test Cached vs Fresh Collection on kennethreitz
print("=== ASYNC PIPELINE PERFORMANCE PROFILE & CACHE BENCHMARK ===")

# First, test cached response (cache hit)
from devlens.app.pipeline_worker import start_collection_job, get_job_status

print("\n--- Test 1: Cached Retrieval (Recruiter re-visiting kennethreitz) ---")
t0 = time.time()
cached_job_id = start_collection_job("kennethreitz", force_refresh=False, max_age_days=7)

while True:
    st = get_job_status(cached_job_id)
    if st["status"] in ("done", "failed"):
        break
    time.sleep(0.1)
elapsed_cached = time.time() - t0

print(f"Cached Job Status: {st['status']}")
print(f"Cached Flag:       {st.get('cached')}")
print(f"Progress Message:  {st['progress']}")
print(f"Time to serve:     {elapsed_cached:.4f} seconds (near-instantaneous)")

# Second, test fresh parallelized collection (bypassing cache)
print("\n--- Test 2: Fresh Live Collection (force_refresh=True, ThreadPoolExecutor max_workers=6) ---")
t0 = time.time()
fresh_job_id = start_collection_job("kennethreitz", force_refresh=True, max_age_days=7)

last_pct = 0
while True:
    st = get_job_status(fresh_job_id)
    pct = st.get("percent", 0)
    if pct != last_pct:
        print(f"  [{time.time()-t0:5.1f}s] [{st['stage']:18s}] {pct:3d}% - {st['progress']}")
        last_pct = pct
    if st["status"] in ("done", "failed"):
        break
    time.sleep(0.5)

elapsed_fresh = time.time() - t0

print(f"\nFresh Collection Result:")
print(f"  Status:       {st['status']}")
print(f"  Stage:        {st['stage']}")
print(f"  Total Time:   {elapsed_fresh:.2f} seconds")
print(f"  Baseline:     ~205.00 seconds")
print(f"  Speedup:      {205.0 / elapsed_fresh:.2f}x faster")

# Verify DB state
conn = sqlite3.connect("devlens.db")
c = conn.cursor()
c.execute("""
    SELECT d.username, d.source, s.id, ap.archetype_label, ap.confidence, ap.shap_top_features, s.collected_at
    FROM developers d
    JOIN snapshots s ON d.id = s.developer_id
    JOIN archetype_predictions ap ON s.id = ap.snapshot_id
    WHERE d.username = 'kennethreitz'
    ORDER BY s.id DESC LIMIT 1
""")
row = c.fetchone()
print(f"\nVerified Final DB Snapshot for kennethreitz:")
print(f"  Snapshot ID:  {row[2]}")
print(f"  Archetype:    {row[3]}")
print(f"  Confidence:   {row[4]:.4f}")
print(f"  Source:       {row[1]}")
print(f"  Collected At: {row[6]}")
