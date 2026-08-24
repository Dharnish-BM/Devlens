"""
DevLens Flask Web Application.

Git/commit-log themed UI design system for developer intelligence & project-fit matchmaking.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, abort, jsonify, render_template, request

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

    @app.route("/")
    def index():
        """Homepage & Developer Portfolio Directory."""
        with get_session() as session:
            rows = (
                session.query(
                    Developer.username,
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

        # Get primary language per user from pool
        pool_dict = {dev.username: dev.primary_language for dev in get_pool()}

        dev_list = []
        for r in rows:
            c_id = r.cluster_id
            cluster_name = "Power-User (Cluster 0)" if c_id == 0 else "Baseline (Cluster 1)"
            dev_list.append({
                "username": r.username,
                "archetype": r.archetype_label or "Unclassified / Low Signal",
                "cluster_id": c_id,
                "cluster_name": cluster_name,
                "primary_language": pool_dict.get(r.username, "Unknown"),
                "doc_score": float(r.doc_score or 0.0),
                "eng_score": float(r.eng_score or 0.0),
            })

        return render_template("index.html", developers=dev_list)

    @app.route("/profile/<username>")
    def profile(username: str):
        """Developer Detail Profile Page."""
        with get_session() as session:
            dev_obj = get_developer(session, username)
            if not dev_obj:
                abort(404, description=f"Developer '{username}' not found in database.")

            snap = session.query(Snapshot).filter_by(developer_id=dev_obj.id).order_by(Snapshot.id.desc()).first()
            if not snap:
                abort(404, description=f"No collection snapshot found for '{username}'.")

            arch_obj = session.query(ArchetypePrediction).filter_by(snapshot_id=snap.id).first()
            clust_obj = session.query(ClusterAssignment).filter_by(snapshot_id=snap.id).first()
            doc_obj = session.query(DocScore).filter_by(snapshot_id=snap.id).first()
            eng_obj = session.query(EngMaturityScore).filter_by(snapshot_id=snap.id).first()

            # Load raw JSON details
            json_path = RAW_DIR / f"{username}.json"
            user_info = {}
            total_repos = 0
            lang_dist = []

            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                    user_info = raw_data.get("user_info") or {}
                    repos = raw_data.get("repositories") or []
                    total_repos = len([r for r in repos if not r.get("is_fork")])

                    gql = raw_data.get("graphql_data") or {}
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
                    logger.warning(f"Error parsing raw JSON for {username}: {e}")

            shap_features = {}
            if arch_obj and arch_obj.shap_top_features:
                if isinstance(arch_obj.shap_top_features, str):
                    shap_features = json.loads(arch_obj.shap_top_features)
                elif isinstance(arch_obj.shap_top_features, dict):
                    shap_features = arch_obj.shap_top_features

            c_id = clust_obj.cluster_id if clust_obj else 1
            cluster_name = "Power-User / Showcase (Cluster 0)" if c_id == 0 else "Baseline Activity (Cluster 1)"

            profile_data = {
                "username": username,
                "snapshot_id": snap.id,
                "collected_at": snap.collected_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "user_info": user_info,
                "total_repos": total_repos,
                "archetype": arch_obj.archetype_label if arch_obj else "Unclassified / Low Signal",
                "confidence": arch_obj.confidence if arch_obj else 1.0,
                "shap_top_features": shap_features,
                "cluster_id": c_id,
                "cluster_name": cluster_name,
                "distance_to_centroid": clust_obj.distance_to_centroid if clust_obj else 0.0,
                "doc_score": float(doc_obj.score) if doc_obj else 0.0,
                "doc_components": doc_obj.components or {} if doc_obj else {},
                "eng_score": float(eng_obj.score) if eng_obj else 0.0,
                "eng_components": eng_obj.components or {} if eng_obj else {},
                "language_distribution": lang_dist,
            }

        return render_template("profile.html", dev=profile_data)

    @app.route("/match")
    def match():
        """Project-Fit Matchmaking & Benchmark Comparison."""
        preset = request.args.get("preset", "").lower()

        # Handle Presets or Custom Inputs
        if preset == "frontend":
            req = ProjectRequirement(
                name="Modern Web Client (Frontend Specialist)",
                required_archetypes=["Frontend Developer"],
                min_doc_score=0.40,
                min_eng_maturity_score=0.30,
                required_languages=["JavaScript", "TypeScript", "HTML", "CSS"],
                weights={"archetype_match": 0.40, "language_similarity": 0.40, "doc_maturity_fit": 0.20}
            )
        elif preset == "backend":
            req = ProjectRequirement(
                name="Enterprise Core Services (Backend Specialist)",
                required_archetypes=["Backend Developer"],
                min_doc_score=0.30,
                min_eng_maturity_score=0.30,
                required_languages=["C", "C++", "Java", "Python"],
                weights={"archetype_match": 0.40, "language_similarity": 0.40, "doc_maturity_fit": 0.20}
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

            # Normalize weights to sum to 1.0
            w_total = w_arch + w_lang + w_mat
            if w_total > 0:
                w_arch, w_lang, w_mat = w_arch / w_total, w_lang / w_total, w_mat / w_total

            req = ProjectRequirement(
                name="Custom Project Specification",
                required_archetypes=[arch],
                min_doc_score=min_doc,
                min_eng_maturity_score=min_eng,
                required_languages=langs,
                weights={"archetype_match": w_arch, "language_similarity": w_lang, "doc_maturity_fit": w_mat}
            )
        else:
            # Default to Phase 8 DevOps Benchmark
            req = ProjectRequirement(
                name="Cloud Infrastructure & Automation (DevOps Specialist)",
                required_archetypes=["DevOps Engineer"],
                min_doc_score=0.30,
                min_eng_maturity_score=0.45,
                required_languages=["Dockerfile", "Shell", "Python", "Makefile", "HCL"],
                weights={"archetype_match": 0.35, "language_similarity": 0.35, "doc_maturity_fit": 0.30}
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
            gders_match_count=gders_match_count
        )

    @app.route("/api/collect", methods=["POST"])
    def api_collect():
        """Demo build notice for live collection endpoint."""
        return jsonify({
            "status": "disabled",
            "message": (
                "Live on-demand GitHub collection is disabled in this showcase build "
                "due to GitHub API rate limits and evaluation reproducibility constraints. "
                "Please query from the 188 pre-collected developer profiles."
            )
        }), 501

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
