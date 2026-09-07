import time
import requests
import json

base_url = "http://127.0.0.1:5000"
test_user = "kennethreitz"

print(f"=== 1. TRIGGERING ASYNC COLLECTION PIPELINE FOR: {test_user} ===")
resp = requests.post(f"{base_url}/collect/start/{test_user}")
print(f"Start Response Status: {resp.status_code}")
job_data = resp.json()
print("Response JSON:", job_data)
job_id = job_data.get("job_id")
assert job_id, "No job_id returned"

print(f"\n=== 2. POLLING PIPELINE STAGES FOR JOB: {job_id} ===")
last_stage = ""
last_progress = ""

start_t = time.time()
while True:
    status_resp = requests.get(f"{base_url}/collect/status/{job_id}")
    status = status_resp.json()
    
    stage = status.get("stage")
    progress = status.get("progress")
    percent = status.get("percent")
    
    if stage != last_stage or progress != last_progress:
        elapsed = time.time() - start_t
        print(f"[{elapsed:5.1f}s] [{percent:3d}%] Stage: {stage:<20} | Progress: {progress}")
        last_stage = stage
        last_progress = progress
        
    if stage in ("done", "failed"):
        break
        
    time.sleep(1.5)

print("\n=== 3. FINAL JOB STATUS ===")
print(json.dumps(status, indent=2))

if stage == "done":
    print(f"\n=== 4. FETCHING RENDERED PROFILE: /profile/{test_user} ===")
    p_resp = requests.get(f"{base_url}/profile/{test_user}")
    print(f"Profile Page HTTP Status: {p_resp.status_code}")
    print("Profile HTML Snippets:")
    for line in p_resp.text.splitlines():
        if any(term in line for term in ["Functional Archetype", "Engagement Tier", "Confidence", "DocScore", "EngScore", "SNAPSHOT-ID", "Python", "Centroid Dist"]):
            print("  ", line.strip().encode("ascii", "replace").decode())
else:
    print(f"Pipeline failed: {status.get('error')}")
