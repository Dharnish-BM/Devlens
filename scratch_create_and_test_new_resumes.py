import os
import requests
import sys
sys.path.append('d:/GIT/Devlens')
from devlens.data_collection.create_sample_resumes import create_pdf
from devlens.data_collection.resume_extractor import ResumeExtractor

# 1. Create two new distinct test resumes
os.makedirs("data/sample_resumes", exist_ok=True)

# Resume 1: Explicit Direct URL
path_direct = "data/sample_resumes/resume_direct_kenneth.pdf"
create_pdf(
    path_direct,
    [
        "Kenneth Reitz - Python Core & API Architect",
        "Email: me@kennethreitz.org",
        "Portfolio & Code: https://github.com/kennethreitz",
        "Location: Boulder, CO",
        "Summary: Author of Python Requests, Pipenv, and open source tools.",
        "Skills: Python, HTTP Architecture, API Design, Distributed Systems"
    ]
)

# Resume 2: Inferred Only (NO GitHub link, NO GitHub keyword)
path_inferred = "data/sample_resumes/resume_inferred_mitsuhiko.pdf"
create_pdf(
    path_inferred,
    [
        "Armin Ronacher",
        "Email: mitsuhiko@gmail.com",
        "Location: Vienna, Austria",
        "Role: Systems Architect & Open Source Developer",
        "Summary: Specializing in Python, Rust, SDK design, and web frameworks.",
        "Experience: Sentry Architecture Lead, Open Source Maintainer."
    ]
)

print("Generated two new distinct test resumes:")
print("  •", path_direct)
print("  •", path_inferred)

# 2. Extract and print raw text from both
ext = ResumeExtractor()
print("\n=== RAW TEXT: resume_direct_kenneth.pdf ===")
print(ext.extract_raw_text(path_direct))

print("\n=== RAW TEXT: resume_inferred_mitsuhiko.pdf ===")
print(ext.extract_raw_text(path_inferred))

# 3. Test through the Flask Web App endpoint POST /upload/process
print("\n=== RUNNING THROUGH FLASK /upload/process ENDPOINT ===")

# Test Resume 1 (Direct URL)
with open(path_direct, "rb") as f:
    files = {"resume_file": (os.path.basename(path_direct), f, "application/pdf")}
    data = {"upload_mode": "single_file"}
    resp1 = requests.post("http://127.0.0.1:5000/upload/process", files=files, data=data)

print(f"\n1. Direct URL Resume ({os.path.basename(path_direct)}) -> HTTP {resp1.status_code}")
for line in resp1.text.splitlines():
    if "kennethreitz" in line or "direct_url" in line or "Ready to Stage" in line:
        print("  ", line.strip().encode("ascii", "replace").decode())

# Test Resume 2 (Inferred Email)
with open(path_inferred, "rb") as f:
    files = {"resume_file": (os.path.basename(path_inferred), f, "application/pdf")}
    data = {"upload_mode": "single_file"}
    resp2 = requests.post("http://127.0.0.1:5000/upload/process", files=files, data=data)

print(f"\n2. Inferred Email Resume ({os.path.basename(path_inferred)}) -> HTTP {resp2.status_code}")
for line in resp2.text.splitlines():
    if "mitsuhiko" in line or "inferred_verified" in line or "Verification Required" in line or "Confirm" in line:
        print("  ", line.strip().encode("ascii", "replace").decode())
