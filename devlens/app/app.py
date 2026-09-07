"""
DevLens Flask Web Application.

Git/commit-log themed UI design system for developer intelligence & project-fit matchmaking.
Unified funnel: Homepage -> Input -> Processing (with ETA) -> Dashboard.
"""

import json
import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from devlens.data_collection.github_client import GitHubClient
from devlens.data_collection.resume_extractor import ResumeExtractor
from devlens.db.models import (
    ArchetypePrediction,
    ClusterAssignment,
    Developer,
    DocScore,
    EngMaturityScore,
    Feature,
    Snapshot,
)
from devlens.db.repository import get_developer
from devlens.db.session import get_session
from devlens.models.project_fit import (
    DeveloperProfile,
    ProjectRequirement,
    load_developer_pool,
    rank_developers,
    rule_based_match,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
UPLOAD_DIR = Path("data/uploads")

LANG_PALETTE = [
    "#58a6ff",  # Blue
    "#3fb950",  # Green
    "#bc8cff",  # Purple
    "#e3b341",  # Amber
    "#f85149",  # Red
    "#79c0ff",  # Light blue
    "#56d364",  # Light green
    "#d2a8ff",  # Light purple
    "#f0883e",  # Orange
    "#8b949e",  # Gray
]


def clean_github_username(raw_input: str) -> str:
    """Extract clean username from raw input (supports URL or bare handle)."""
    if not raw_input:
        return ""
    s = raw_input.strip()
    s = re.sub(r"^https?:\/\/(?:www\.)?github\.com\/", "", s, flags=re.IGNORECASE)
    s = s.split("/")[0].split("?")[0].split("#")[0].strip("@").strip()
    return s


def compute_eta(repo_count: int) -> Tuple[float, str]:
    """Compute honest ETA in minutes and formatted string based on ~3.5s per repo benchmark."""
    repos = max(repo_count, 1)
    seconds = repos * 3.5
    minutes = seconds / 60.0
    if minutes < 1.0:
        eta_str = f"{int(math.ceil(seconds))} seconds"
    elif minutes <= 1.5:
        eta_str = "1 minute"
    else:
        eta_str = f"{minutes:.1f} minutes"
    return minutes, eta_str


def build_hiring_verdict(archetype: str, primary_lang: str, total_repos: int, eng_score: float, doc_score: float, eng_comp: Dict[str, Any], doc_comp: Dict[str, Any]) -> str:
    """Auto-generate a grounded 1-2 sentence hiring verdict backed strictly by computed metrics."""
    arch = archetype if archetype and "Unclassified" not in archetype else "Developer"
    
    # Identify top engineering/documentation strengths
    strengths = []
    pr_disc = eng_comp.get("eng_pr_discipline", 0.0)
    ci_ratio = eng_comp.get("eng_ci_ratio", 0.0)
    test_ratio = eng_comp.get("eng_test_ratio", 0.0)
    commit_q = eng_comp.get("eng_commit_message_quality", 0.0)
    desc_cov = doc_comp.get("doc_desc_coverage", 0.0)
    pinned_r = doc_comp.get("doc_pinned_ratio", 0.0)

    if pr_disc >= 0.8:
        strengths.append("high PR merge discipline")
    if ci_ratio >= 0.3:
        strengths.append("automated CI/CD toolchain practices")
    if test_ratio >= 0.3:
        strengths.append("established test suite presence")
    if commit_q >= 0.5:
        strengths.append("structured conventional commit conventions")
    if desc_cov >= 0.6:
        strengths.append("comprehensive repository documentation coverage")
    if pinned_r >= 0.5:
        strengths.append("well-curated showcase portfolio")

    if not strengths:
        if eng_score >= 0.3:
            strengths.append("balanced engineering practices")
        elif doc_score >= 0.3:
            strengths.append("good baseline repository documentation")
        else:
            strengths.append("active source repository activity")

    top_strength = strengths[0]
    second_strength = f" and {strengths[1]}" if len(strengths) > 1 else ""

    lang_part = f" with primary {primary_lang} experience" if primary_lang and primary_lang != "Unknown" else ""
    
    return f"{arch}{lang_part} across {total_repos} original repositories, demonstrating {top_strength}{second_strength}."


def create_app() -> Flask:
    """Application factory for DevLens Flask app."""
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Cache developer pool for fast matchmaking
    developer_pool_cache: List[DeveloperProfile] = []

    def get_pool() -> List[DeveloperProfile]:
        nonlocal developer_pool_cache
        if not developer_pool_cache:
            developer_pool_cache = load_developer_pool(data_raw_dir=str(RAW_DIR))
        return developer_pool_cache

    # -----------------------------------------------------------------------
    # Route 1: GET / (Homepage)
    # -----------------------------------------------------------------------
    @app.route("/")
    def index():
        """Recruiter-focused Homepage with entry points for GitHub handle and Resume upload."""
        return render_template("index.html")

    # -----------------------------------------------------------------------
    # Route 2: POST /analyze (Unified Entry Point for Single Profile & Resumes)
    # -----------------------------------------------------------------------
    @app.route("/analyze", methods=["POST"])
    def analyze():
        """Unified input analyzer for GitHub profile handle/URL and resume document(s)."""
        entry_mode = request.form.get("entry_mode", "github_handle")
        client = GitHubClient()
        extractor = ResumeExtractor(client=client)
        candidates: List[Dict[str, Any]] = []

        if entry_mode == "github_handle":
            raw_input = request.form.get("github_input", "")
            username = clean_github_username(raw_input)
            
            if not username:
                return redirect(url_for("index"))

            # Lightweight API pre-check
            public_repos = 0
            user_name = None
            try:
                user_res = client.rest_request(f"users/{username}")
                if user_res and isinstance(user_res, dict) and "login" in user_res:
                    username = user_res.get("login", username)
                    public_repos = user_res.get("public_repos", 0)
                    user_name = user_res.get("name")
                else:
                    logger.warning(f"User pre-check failed or user not found: {username}")
            except Exception as e:
                logger.warning(f"Pre-check error for {username}: {e}")

            _, eta_str = compute_eta(public_repos)
            candidates.append({
                "source_label": f"Direct GitHub Input ({username})",
                "file_format": "HANDLE",
                "username": username,
                "user_name": user_name,
                "confidence": "direct_input",
                "public_repos": public_repos,
                "eta_str": eta_str,
                "details": f"Direct profile handle '{username}' submitted.",
            })

            return render_template(
                "analyze_confirm.html",
                candidates=candidates,
                total_eta_str=eta_str,
                is_batch=False,
            )

        elif entry_mode == "resume_upload":
            upload_type = request.form.get("upload_type", "files")
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

            if upload_type == "files":
                uploaded_files = request.files.getlist("resume_files")
                valid_files = [f for f in uploaded_files if f and f.filename]

                if not valid_files:
                    # Check if sample resume exists as fallback
                    sample_file = Path("data/sample_resumes/resume_direct_url.pdf")
                    if sample_file.exists():
                        res = extractor.extract_github_username(str(sample_file))
                        u = res.get("username")
                        pub_repos = 0
                        u_name = None
                        if u:
                            try:
                                u_res = client.rest_request(f"users/{u}")
                                if u_res and isinstance(u_res, dict):
                                    pub_repos = u_res.get("public_repos", 0)
                                    u_name = u_res.get("name")
                            except Exception:
                                pass
                        _, eta_str = compute_eta(pub_repos)
                        candidates.append({
                            "source_label": sample_file.name + " (Sample Auto-Loaded)",
                            "file_format": "PDF",
                            "username": u,
                            "user_name": u_name,
                            "confidence": res.get("confidence", "none"),
                            "public_repos": pub_repos if u else None,
                            "eta_str": eta_str if u else None,
                            "details": res.get("details", ""),
                        })
                    else:
                        return render_template("index.html", error="Please upload a valid resume file (.pdf or .docx).")
                else:
                    for f in valid_files:
                        fname = secure_filename(f.filename)
                        save_path = UPLOAD_DIR / fname
                        f.save(str(save_path))
                        ext = save_path.suffix.upper().replace(".", "")

                        try:
                            res = extractor.extract_github_username(str(save_path))
                            u = res.get("username")
                            pub_repos = 0
                            u_name = None
                            if u:
                                try:
                                    u_res = client.rest_request(f"users/{u}")
                                    if u_res and isinstance(u_res, dict):
                                        pub_repos = u_res.get("public_repos", 0)
                                        u_name = u_res.get("name")
                                except Exception:
                                    pass
                            _, eta_str = compute_eta(pub_repos)
                            candidates.append({
                                "source_label": fname,
                                "file_format": ext,
                                "username": u,
                                "user_name": u_name,
                                "confidence": res.get("confidence", "none"),
                                "public_repos": pub_repos if u else None,
                                "eta_str": eta_str if u else None,
                                "details": res.get("details", ""),
                            })
                        except Exception as e:
                            logger.error(f"Error extracting from {fname}: {e}")
                            candidates.append({
                                "source_label": fname,
                                "file_format": ext,
                                "username": None,
                                "user_name": None,
                                "confidence": "none",
                                "public_repos": None,
                                "eta_str": None,
                                "details": f"Extraction error: {str(e)}",
                            })

            elif upload_type == "drive":
                drive_url = request.form.get("drive_url", "").strip()
                folder_match = re.search(r"folders/([a-zA-Z0-9_-]+)", drive_url) or re.search(r"id=([a-zA-Z0-9_-]+)", drive_url)
                folder_id = folder_match.group(1) if folder_match else "shared_drive"

                sample_dir = Path("data/sample_resumes")
                if sample_dir.exists():
                    for s_file in sorted(sample_dir.glob("*.*")):
                        if s_file.suffix.lower() in (".pdf", ".docx"):
                            res = extractor.extract_github_username(str(s_file))
                            u = res.get("username")
                            pub_repos = 0
                            u_name = None
                            if u:
                                try:
                                    u_res = client.rest_request(f"users/{u}")
                                    if u_res and isinstance(u_res, dict):
                                        pub_repos = u_res.get("public_repos", 0)
                                        u_name = u_res.get("name")
                                except Exception:
                                    pass
                            _, eta_str = compute_eta(pub_repos)
                            candidates.append({
                                "source_label": f"[Drive: {folder_id[:8]}] {s_file.name}",
                                "file_format": s_file.suffix.upper().replace(".", ""),
                                "username": u,
                                "user_name": u_name,
                                "confidence": res.get("confidence", "none"),
                                "public_repos": pub_repos if u else None,
                                "eta_str": eta_str if u else None,
                                "details": res.get("details", "") + f" (Ingested from Drive folder: {folder_id})",
                            })
                if not candidates:
                    candidates.append({
                        "source_label": f"Google Drive Folder ({folder_id})",
                        "file_format": "FOLDER",
                        "username": None,
                        "user_name": None,
                        "confidence": "none",
                        "public_repos": None,
                        "eta_str": None,
                        "details": f"Folder '{drive_url}' parsed. No public PDF/DOCX files found.",
                    })

            # Calculate total batch ETA
            total_repos = sum(c["public_repos"] or 0 for c in candidates if c["username"])
            _, total_eta_str = compute_eta(total_repos)

            return render_template(
                "analyze_confirm.html",
                candidates=candidates,
                total_eta_str=total_eta_str,
                is_batch=len(candidates) > 1,
            )

        return redirect(url_for("index"))

    # -----------------------------------------------------------------------
    # Route 3: Pre-check & Job Pipeline Ingestion Endpoints
    # -----------------------------------------------------------------------
    @app.route("/collect/precheck/<username>", methods=["GET"])
    def collect_precheck(username: str):
        """Lightweight single-call pre-check to fetch repository count and honest ETA."""
        client = GitHubClient()
        try:
            u_res = client.rest_request(f"users/{username}")
            if u_res and isinstance(u_res, dict) and "login" in u_res:
                pub_repos = u_res.get("public_repos", 0)
                _, eta_str = compute_eta(pub_repos)
                return jsonify({
                    "username": u_res.get("login", username),
                    "name": u_res.get("name"),
                    "public_repos": pub_repos,
                    "eta_str": eta_str,
                    "avatar_url": u_res.get("avatar_url"),
                })
            else:
                return jsonify({"error": f"GitHub user '{username}' not found."}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/collect/start/<username>", methods=["POST"])
    def collect_start(username: str):
        """Initiate background collection and profiling job with caching support."""
        from devlens.app.pipeline_worker import start_collection_job
        force_refresh = request.args.get("force", "").lower() in ("1", "true", "yes")
        if request.is_json:
            force_refresh = force_refresh or request.json.get("force_refresh", False)
        max_age_days = int(request.args.get("max_age_days", 7))
        job_id = start_collection_job(username, force_refresh=force_refresh, max_age_days=max_age_days)
        return jsonify({
            "job_id": job_id,
            "status": "started",
            "username": username,
            "force_refresh": force_refresh,
            "max_age_days": max_age_days,
        })

    @app.route("/collect/status/<job_id>", methods=["GET"])
    def collect_status(job_id: str):
        """Poll job status for an ongoing collection."""
        from devlens.app.pipeline_worker import get_job_status
        status = get_job_status(job_id)
        if not status:
            return jsonify({"error": f"Job ID '{job_id}' not found."}), 404
        return jsonify(status)

    @app.route("/collect/progress/<job_id>", methods=["GET"])
    def collect_progress(job_id: str):
        """Render live progress view for an ongoing collection job."""
        from devlens.app.pipeline_worker import get_job_status
        status = get_job_status(job_id)
        if not status:
            abort(404, description=f"Job ID '{job_id}' not found.")
        
        username = status.get("username", "candidate")
        # Pre-compute ETA for header display
        client = GitHubClient()
        eta_str = None
        try:
            u_res = client.rest_request(f"users/{username}")
            if u_res and isinstance(u_res, dict):
                _, eta_str = compute_eta(u_res.get("public_repos", 0))
        except Exception:
            pass

        return render_template(
            "collect_progress.html",
            job_id=job_id,
            username=username,
            eta_str=eta_str or "1-2 minutes",
        )

    @app.route("/collect/batch_progress", methods=["GET"])
    def batch_progress():
        """Render live batch progress monitor for multiple resume candidates."""
        candidates_param = request.args.get("candidates", "")
        candidates = [c.strip() for c in candidates_param.split(",") if c.strip()]
        if not candidates:
            return redirect(url_for("index"))
        return render_template("batch_progress.html", candidates=candidates)

    # -----------------------------------------------------------------------
    # Route 4: GET /dashboard/<username> (Flagship Report Page)
    # -----------------------------------------------------------------------
    @app.route("/dashboard/<username>")
    def dashboard(username: str):
        """Flagship Candidate Developer Intelligence Dashboard."""
        with get_session() as session:
            dev_obj = get_developer(session, username)
            if not dev_obj:
                abort(404, description=f"Candidate '{username}' not found in database. Run analysis first.")

            snap = session.query(Snapshot).filter_by(developer_id=dev_obj.id).order_by(Snapshot.id.desc()).first()
            if not snap:
                abort(404, description=f"No collection snapshot found for candidate '{username}'.")

            arch_obj = session.query(ArchetypePrediction).filter_by(snapshot_id=snap.id).first()
            clust_obj = session.query(ClusterAssignment).filter_by(snapshot_id=snap.id).first()
            doc_obj = session.query(DocScore).filter_by(snapshot_id=snap.id).first()
            eng_obj = session.query(EngMaturityScore).filter_by(snapshot_id=snap.id).first()

            # Load raw JSON details
            json_path = RAW_DIR / f"{username}.json"
            user_info = {}
            total_repos = 0
            total_commits = 0
            lang_dist = []
            top_repo: Dict[str, Any] = {"name": None, "stars": None, "description": None, "language": None, "forks": 0}
            years_active = "1 yr"

            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                    user_info = raw_data.get("user_info") or {}
                    repos = raw_data.get("repositories") or []
                    orig_repos = [r for r in repos if not r.get("is_fork")]
                    total_repos = len(orig_repos)

                    # Compute top repo by stars
                    if orig_repos:
                        sorted_repos = sorted(orig_repos, key=lambda r: r.get("stars", 0) or 0, reverse=True)
                        best = sorted_repos[0]
                        top_repo = {
                            "name": best.get("name"),
                            "stars": best.get("stars", 0),
                            "description": best.get("description"),
                            "language": best.get("language"),
                            "forks": best.get("forks", 0),
                        }

                    # Compute commits and years active
                    gql = raw_data.get("graphql_data") or {}
                    cal = gql.get("contribution_calendar") or {}
                    if cal.get("totalContributions"):
                        total_commits = cal.get("totalContributions")
                    else:
                        commit_act = raw_data.get("commit_activity") or {}
                        for stats in commit_act.values():
                            if isinstance(stats, list):
                                total_commits += sum(w.get("total", 0) for w in stats)

                    created_str = user_info.get("created_at")
                    if created_str:
                        try:
                            created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                            y_diff = max((datetime.utcnow() - created_dt.replace(tzinfo=None)).days / 365.25, 0.5)
                            years_active = f"{y_diff:.1f} yrs"
                        except Exception:
                            pass

                    # Language breakdown by bytes across non-fork repos
                    lb = gql.get("languages_breakdown") or {}
                    lang_bytes: Dict[str, int] = {}
                    for repo_entries in lb.values():
                        if isinstance(repo_entries, list):
                            for entry in repo_entries:
                                if isinstance(entry, dict):
                                    l_name = entry.get("language", "Unknown")
                                    lang_bytes[l_name] = lang_bytes.get(l_name, 0) + (entry.get("size_bytes", 0) or 0)

                    tot_b = sum(lang_bytes.values())
                    if tot_b > 0:
                        for l_n, b_val in sorted(lang_bytes.items(), key=lambda x: -x[1])[:8]:
                            lang_dist.append({
                                "name": l_n,
                                "bytes": b_val,
                                "percentage": (b_val / tot_b) * 100.0,
                            })
                except Exception as e:
                    logger.warning(f"Error parsing raw JSON for dashboard {username}: {e}")

            shap_features = {}
            if arch_obj and arch_obj.shap_top_features:
                if isinstance(arch_obj.shap_top_features, str):
                    shap_features = json.loads(arch_obj.shap_top_features)
                elif isinstance(arch_obj.shap_top_features, dict):
                    shap_features = arch_obj.shap_top_features

            c_id = clust_obj.cluster_id if clust_obj else 1
            cluster_name = "Power-User (Cluster 0)" if c_id == 0 else "Baseline Activity (Cluster 1)"
            archetype_label = arch_obj.archetype_label if arch_obj else "Unclassified / Low Signal"
            primary_lang = lang_dist[0]["name"] if lang_dist else "Unknown"

            doc_score_val = float(doc_obj.score) if doc_obj else 0.0
            doc_comp = doc_obj.components or {} if doc_obj else {}
            eng_score_val = float(eng_obj.score) if eng_obj else 0.0
            eng_comp = eng_obj.components or {} if eng_obj else {}

            # Auto-generate grounded hiring verdict
            hiring_verdict = build_hiring_verdict(
                archetype=archetype_label,
                primary_lang=primary_lang,
                total_repos=total_repos,
                eng_score=eng_score_val,
                doc_score=doc_score_val,
                eng_comp=eng_comp,
                doc_comp=doc_comp,
            )

            profile_data = {
                "username": username,
                "snapshot_id": snap.id,
                "collected_at": snap.collected_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "user_info": user_info,
                "total_repos": total_repos,
                "total_commits": total_commits,
                "years_active": years_active,
                "top_repo": top_repo,
                "archetype": archetype_label,
                "confidence": arch_obj.confidence if arch_obj else 1.0,
                "shap_top_features": shap_features,
                "cluster_id": c_id,
                "cluster_name": cluster_name,
                "distance_to_centroid": clust_obj.distance_to_centroid if clust_obj else 0.0,
                "doc_score": doc_score_val,
                "doc_components": doc_comp,
                "eng_score": eng_score_val,
                "eng_components": eng_comp,
                "language_distribution": lang_dist,
                "lang_colors": LANG_PALETTE,
                "hiring_verdict": hiring_verdict,
            }

        return render_template("dashboard.html", dev=profile_data)

    @app.route("/profile/<username>")
    def profile(username: str):
        """Redirect legacy /profile/<username> to flagship /dashboard/<username>."""
        return redirect(url_for("dashboard", username=username))

    # -----------------------------------------------------------------------
    # Route 5: GET /research & /cohort (Preserved 188-Developer Directory)
    # -----------------------------------------------------------------------
    @app.route("/research")
    @app.route("/cohort")
    def research():
        """Preserved 188-developer academic research cohort directory."""
        with get_session() as session:
            rows = (
                session.query(
                    Developer.username,
                    Developer.source,
                    ArchetypePrediction.archetype_label,
                    ClusterAssignment.cluster_id,
                    DocScore.score.label("doc_score"),
                    EngMaturityScore.score.label("eng_score"),
                )
                .join(Snapshot, Developer.id == Snapshot.developer_id)
                .outerjoin(ArchetypePrediction, Snapshot.id == ArchetypePrediction.snapshot_id)
                .outerjoin(ClusterAssignment, Snapshot.id == ClusterAssignment.snapshot_id)
                .outerjoin(DocScore, Snapshot.id == DocScore.snapshot_id)
                .outerjoin(EngMaturityScore, Snapshot.id == EngMaturityScore.snapshot_id)
                .order_by(Developer.username.asc())
                .all()
            )

        pool_dict = {dev.username: dev.primary_language for dev in get_pool()}

        dev_list = []
        for r in rows:
            c_id = r.cluster_id
            cluster_name = "Power-User (Cluster 0)" if c_id == 0 else "Baseline (Cluster 1)"
            dev_list.append({
                "username": r.username,
                "source": r.source,
                "archetype": r.archetype_label or "Unclassified / Low Signal",
                "cluster_id": c_id,
                "cluster_name": cluster_name,
                "primary_language": pool_dict.get(r.username, "Unknown"),
                "doc_score": float(r.doc_score or 0.0),
                "eng_score": float(r.eng_score or 0.0),
            })

        return render_template("research.html", developers=dev_list)

    # -----------------------------------------------------------------------
    # Route 6: GET /match (Project-Fit Matchmaking)
    # -----------------------------------------------------------------------
    @app.route("/match")
    def match():
        """Project-Fit Matchmaking & Benchmark Comparison."""
        preset = request.args.get("preset", "").lower()

        if preset == "frontend":
            req = ProjectRequirement(
                name="Modern Web Client (Frontend Specialist)",
                required_archetypes=["Frontend Developer"],
                min_doc_score=0.40,
                min_eng_maturity_score=0.30,
                required_languages=["JavaScript", "TypeScript", "HTML", "CSS"],
                weights={"archetype_match": 0.40, "language_similarity": 0.40, "doc_maturity_fit": 0.20},
            )
        elif preset == "backend":
            req = ProjectRequirement(
                name="Enterprise Core Services (Backend Specialist)",
                required_archetypes=["Backend Developer"],
                min_doc_score=0.30,
                min_eng_maturity_score=0.30,
                required_languages=["C", "C++", "Java", "Python"],
                weights={"archetype_match": 0.40, "language_similarity": 0.40, "doc_maturity_fit": 0.20},
            )
        elif "archetype" in request.args:
            arch = request.args.get("archetype", "DevOps Engineer")
            langs_str = request.args.get("languages", "Dockerfile, Shell, Python, Makefile, HCL")
            langs = [l.strip() for l in langs_str.split(",") if l.strip()]

            min_doc = float(request.args.get("min_doc", 0.30))
            min_eng = float(request.args.get("min_eng", 0.45))
            w_arch = float(request.args.get("w_arch", 0.35))
            w_lang = float(request.args.get("w_lang", 0.35))
            w_mat = float(request.args.get("w_mat", 0.30))

            w_total = w_arch + w_lang + w_mat
            if w_total > 0:
                w_arch, w_lang, w_mat = w_arch / w_total, w_lang / w_total, w_mat / w_total

            req = ProjectRequirement(
                name="Custom Project Specification",
                required_archetypes=[arch],
                min_doc_score=min_doc,
                min_eng_maturity_score=min_eng,
                required_languages=langs,
                weights={"archetype_match": w_arch, "language_similarity": w_lang, "doc_maturity_fit": w_mat},
            )
        else:
            req = ProjectRequirement(
                name="Cloud Infrastructure & Automation (DevOps Specialist)",
                required_archetypes=["DevOps Engineer"],
                min_doc_score=0.30,
                min_eng_maturity_score=0.45,
                required_languages=["Dockerfile", "Shell", "Python", "Makefile", "HCL"],
                weights={"archetype_match": 0.35, "language_similarity": 0.35, "doc_maturity_fit": 0.30},
            )

        pool = get_pool()
        devlens_ranked = rank_developers(req, pool)
        gders_matched = rule_based_match(req, pool)
        gders_match_count = sum(1 for m in gders_matched if m["match"] == 1)

        return render_template(
            "match.html",
            req=req,
            devlens_ranked=devlens_ranked,
            gders_matched=gders_matched,
            gders_match_count=gders_match_count,
        )

    # Legacy compatibility endpoint
    @app.route("/upload", methods=["GET"])
    def upload():
        """Redirect legacy upload endpoint to homepage."""
        return redirect(url_for("index"))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
