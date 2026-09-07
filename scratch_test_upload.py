import requests

# 1. Test GET /upload
r_get = requests.get("http://127.0.0.1:5000/upload")
print(f"GET /upload Status Code: {r_get.status_code}")
assert "Candidate Resume Ingestion" in r_get.text, "Upload page title missing"

# 2. Test POST /upload/process with direct_url PDF
pdf_path = "data/sample_resumes/resume_direct_url.pdf"
with open(pdf_path, "rb") as f:
    files = {"resume_file": (pdf_path, f, "application/pdf")}
    data = {"upload_mode": "single_file"}
    r_post = requests.post("http://127.0.0.1:5000/upload/process", files=files, data=data)

print(f"POST /upload/process (Direct URL PDF) Status Code: {r_post.status_code}")
print("Response Snippet:")
for line in r_post.text.splitlines():
    if "torvalds" in line or "direct_url" in line or "Ready to Stage" in line:
        print("  ", line.strip().encode("ascii", "replace").decode())

# 3. Test POST /upload/process with inferred fallback PDF
pdf_inf = "data/sample_resumes/resume_inferred.pdf"
with open(pdf_inf, "rb") as f:
    files = {"resume_file": (pdf_inf, f, "application/pdf")}
    data = {"upload_mode": "single_file"}
    r_inf = requests.post("http://127.0.0.1:5000/upload/process", files=files, data=data)

print(f"\nPOST /upload/process (Inferred Fallback PDF) Status Code: {r_inf.status_code}")
for line in r_inf.text.splitlines():
    if "inferred_verified" in line or "Confirm" in line or "Verification Required" in line:
        print("  ", line.strip().encode("ascii", "replace").decode())
