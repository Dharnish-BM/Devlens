from flask import Flask, render_template, request
import sys
import os

# Tell Python where to find score_profile.py (now inside scripts/)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from score_profile import score_username

import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/score-single", methods=["POST"])
def score_single():
    username = request.form.get("username", "").strip()
    if not username:
        return render_template("index.html", error="Please enter a username")
    
    result = score_username(username)
    
    if "error" in result:
        return render_template("index.html", error=result["error"])
    
    return render_template("result_single.html", result=result)

@app.route("/score-batch", methods=["POST"])
def score_batch():
    file = request.files.get("csv_file")
    if not file:
        return render_template("index.html", error="Please upload a CSV file")
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    df = pd.read_csv(filepath)
    # Assumes CSV has a column literally named "username"
    if "username" not in df.columns:
        return render_template("index.html", error="CSV must have a column named 'username'")
    
    results = []
    for uname in df["username"].dropna().str.strip():
        res = score_username(uname)
        if "error" not in res:
            results.append(res)
    
    # Sort by score, highest first
    results.sort(key=lambda x: x["overall_skill_score"], reverse=True)
    
    return render_template("result_batch.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)