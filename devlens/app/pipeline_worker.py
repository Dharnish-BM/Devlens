"""
Asynchronous Background Collection & Profiling Worker for DevLens.

Orchestrates full candidate ingestion:
1. GitHub REST + GraphQL collection (Phase 1 + 1B)
2. 52-metric Feature Engineering (Phase 4)
3. Persisted K-Means Engagement Tier Assignment (Phase 5)
4. Deterministic Hierarchical Archetype Rules (Phase 6)
5. Persisted XGBoost SHAP Feature Attribution (Phase 7)
6. SQLite Database Persistence (tagged source="live_upload")
"""

import json
import logging
import math
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap

from devlens.data_collection.collect_profile import GitHubProfileCollector
from devlens.db.models import Developer, Snapshot
from devlens.db.repository import (
    get_developer,
    get_latest_snapshot,
    insert_archetype_prediction,
    insert_features,
    upsert_cluster_assignment,
    upsert_developer,
    upsert_doc_score,
    upsert_eng_maturity_score,
)
from devlens.db.session import get_session
from devlens.features.feature_engineering import (
    build_feature_vector,
    compute_documentation_score,
    compute_engineering_maturity_score,
)
from devlens.models.explainability import explain_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global in-memory job store
JOBS: Dict[str, Dict[str, Any]] = {}

MODEL_DIR = Path("models/artifacts")
RAW_DIR = Path("data/raw")


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve status of an ongoing or completed collection job."""
    return JOBS.get(job_id)


def start_collection_job(username: str, force_refresh: bool = False, max_age_days: int = 7) -> str:
    """Initialize and spawn an asynchronous background collection job with caching."""
    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    JOBS[job_id] = {
        "job_id": job_id,
        "username": username,
        "status": "running",
        "stage": "collecting_profile",
        "progress": "Checking cache and initializing GitHub collector...",
        "percent": 5,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "cached": False,
    }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, username, force_refresh, max_age_days),
        daemon=True
    )
    thread.start()
    logger.info(f"Started background collection job {job_id} for user '{username}' (force_refresh={force_refresh}, max_age_days={max_age_days}).")
    return job_id


def _run_pipeline(job_id: str, username: str, force_refresh: bool = False, max_age_days: int = 7) -> None:
    """Execute collection, feature calculation, modeling, and DB write with caching and thread-pool enrichment."""
    try:
        job = JOBS[job_id]

        # -------------------------------------------------------------------
        # Step 0: Snapshot Cache Check (Serve existing if collected within N days)
        # -------------------------------------------------------------------
        if not force_refresh:
            with get_session() as session:
                dev = get_developer(session, username)
                if dev:
                    latest_snap = get_latest_snapshot(session, dev.id)
                    if latest_snap and latest_snap.collected_at:
                        age_days = (datetime.utcnow() - latest_snap.collected_at).total_seconds() / 86400.0
                        if age_days <= max_age_days:
                            logger.info(f"Serving cached snapshot for '{username}' (age: {age_days:.1f} days <= {max_age_days} days).")
                            job["cached"] = True
                            job["status"] = "done"
                            job["stage"] = "done"
                            job["progress"] = f"Served existing profile for '{username}' from cache (collected {age_days:.1f} days ago)."
                            job["percent"] = 100
                            job["completed_at"] = datetime.utcnow().isoformat()
                            return

        # -------------------------------------------------------------------
        # Step 1: GitHub REST + GraphQL Collection (Phase 1 + 1B)
        # -------------------------------------------------------------------
        job["stage"] = "collecting_profile"
        job["progress"] = f"Fetching GitHub profile and public repositories for '{username}'..."
        job["percent"] = 15

        collector = GitHubProfileCollector()
        user_info = collector.collect_user_info(username)
        if not user_info:
            raise ValueError(f"GitHub user '{username}' does not exist or API request failed.")

        repos = collector.collect_public_repos(username)
        non_fork_repos = [r for r in repos if not r.get("is_fork")]

        job["stage"] = "enriching_repos"
        job["progress"] = f"Enriching {len(non_fork_repos)} repositories with concurrent thread pool..."
        job["percent"] = 25

        # Parallelized Phase 1B Repo Enrichment using ThreadPoolExecutor (max_workers=6)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _enrich_repo(repo_item: Dict[str, Any]) -> None:
            r_name = repo_item["name"]
            
            # 1. README
            readme_res = collector.client.rest_request(f"repos/{username}/{r_name}/readme", max_retries=2)
            r_len = 0
            if readme_res and isinstance(readme_res, dict) and "content" in readme_res:
                try:
                    import base64
                    content_str = base64.b64decode(readme_res["content"]).decode("utf-8", errors="ignore")
                    r_len = len(content_str)
                except Exception:
                    r_len = readme_res.get("size", 0)
            repo_item["readme_length_chars"] = r_len

            # 2. CI Config
            has_ci = False
            wf_res = collector.client.rest_request(f"repos/{username}/{r_name}/contents/.github/workflows", max_retries=2)
            if wf_res and isinstance(wf_res, list) and len(wf_res) > 0:
                has_ci = True
            else:
                for ci_file in (".travis.yml", "circle.yml", "Jenkinsfile", ".gitlab-ci.yml", "azure-pipelines.yml"):
                    if collector.client.rest_request(f"repos/{username}/{r_name}/contents/{ci_file}", max_retries=1):
                        has_ci = True
                        break
            repo_item["has_ci_config"] = has_ci

            # 3. Test Presence
            has_tests = False
            contents_res = collector.client.rest_request(f"repos/{username}/{r_name}/contents", max_retries=2)
            if isinstance(contents_res, list):
                test_dirs = {"tests", "test", "__tests__", "spec", "testing"}
                test_patterns = ("test_", "_test.", ".test.", ".spec.")
                for item in contents_res:
                    if not isinstance(item, dict):
                        continue
                    i_name = item.get("name", "").lower()
                    i_type = item.get("type", "")
                    if i_type == "dir" and i_name in test_dirs:
                        has_tests = True
                        break
                    elif i_type == "file" and any(p in i_name for p in test_patterns):
                        has_tests = True
                        break
            repo_item["has_test_presence"] = has_tests

            # 4. Commit Messages
            commit_msgs = []
            commits_res = collector.client.rest_request(
                f"repos/{username}/{r_name}/commits",
                params={"per_page": 30},
                max_retries=2
            )
            if isinstance(commits_res, list):
                for c_item in commits_res:
                    if isinstance(c_item, dict):
                        raw_msg = c_item.get("commit", {}).get("message", "")
                        subj = raw_msg.split("\n")[0].strip() if raw_msg else ""
                        if subj:
                            commit_msgs.append(subj)
            repo_item["recent_commit_messages"] = commit_msgs

        completed_count = 0
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_repo = {executor.submit(_enrich_repo, r): r for r in non_fork_repos}
            for future in as_completed(future_to_repo):
                completed_count += 1
                r_obj = future_to_repo[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Error enriching {r_obj.get('name')}: {e}")
                pct = 25 + int(30 * completed_count / max(len(non_fork_repos), 1))
                job["progress"] = f"{completed_count}/{len(non_fork_repos)} repos enriched (checked {r_obj.get('name')})"
                job["percent"] = min(pct, 55)

        # PRs, issues, GraphQL calendar, and languages
        commit_activity = collector.collect_commit_activity(username, repos)
        pr_history = collector.collect_pr_history(username)
        issue_activity = collector.collect_issue_activity(username)
        graphql_data = collector.collect_graphql_data(username)

        total_commits = 0
        if graphql_data and "contribution_calendar" in graphql_data:
            total_commits = graphql_data["contribution_calendar"].get("totalContributions", 0)
        else:
            for stats in commit_activity.values():
                if isinstance(stats, list):
                    total_commits += sum(w.get("total", 0) for w in stats)

        summary_metrics = {
            "repo_count": len(repos),
            "total_commits": total_commits,
            "api_calls_used": collector.client.get_api_calls_summary(),
        }

        profile_data = {
            "username": username,
            "collected_at": datetime.utcnow().isoformat() + "Z",
            "user_info": user_info,
            "repositories": repos,
            "commit_activity": commit_activity,
            "pull_requests": pr_history,
            "issues": issue_activity,
            "graphql_data": graphql_data,
            "summary_metrics": summary_metrics,
        }

        # Save raw JSON
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = RAW_DIR / f"{username}.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)

        # -------------------------------------------------------------------
        # Step 2: Feature Engineering (Phase 4)
        # -------------------------------------------------------------------
        job["stage"] = "computing_features"
        job["progress"] = "Computing 52-metric engineering & documentation feature vector..."
        job["percent"] = 65

        features_dict = build_feature_vector(profile_data)
        doc_comp = compute_documentation_score(profile_data)
        eng_comp = compute_engineering_maturity_score(profile_data)

        # -------------------------------------------------------------------
        # Step 3: K-Means Engagement Tier Assignment (Phase 5 - Preserved Model)
        # -------------------------------------------------------------------
        job["stage"] = "scoring"
        job["progress"] = "Scoring K-Means engagement tier and XGBoost archetype..."
        job["percent"] = 80

        kmeans = joblib.load(MODEL_DIR / "kmeans_model.joblib")
        scaler = joblib.load(MODEL_DIR / "scaler.joblib")

        # Prepare 41-feature vector aligned with scaler
        feature_names_in = getattr(scaler, "feature_names_in_", None)
        if feature_names_in is None:
            feature_names_in = getattr(kmeans, "feature_names_in_", [])

        # Binarize sparse features if required
        features_for_scaler = dict(features_dict)
        features_for_scaler["repo_has_stars"] = 1.0 if features_for_scaler.get("repo_total_stars", 0.0) > 0 else 0.0
        features_for_scaler["collab_issues_opened"] = 1.0 if features_for_scaler.get("collab_issues_opened", 0.0) > 0 else 0.0
        features_for_scaler["collab_issues_commented"] = 1.0 if features_for_scaler.get("collab_issues_commented", 0.0) > 0 else 0.0

        feat_row = [features_for_scaler.get(fname, 0.0) for fname in feature_names_in]
        df_41 = pd.DataFrame([feat_row], columns=feature_names_in)
        scaled_vec = scaler.transform(df_41)

        cluster_id = int(kmeans.predict(scaled_vec)[0])
        dist_to_center = float(np.linalg.norm(scaled_vec[0] - kmeans.cluster_centers_[cluster_id]))

        # -------------------------------------------------------------------
        # Step 4: Phase 6 Rule-Based Archetype Classification (Guarded)
        # -------------------------------------------------------------------
        lang_ml = features_dict.get("lang_ml_signal", 0.0)
        lang_mob = features_dict.get("lang_mobile_signal", 0.0)
        lang_devops = features_dict.get("lang_devops_signal", 0.0)
        lang_front = features_dict.get("lang_frontend_signal", 0.0)
        lang_js = features_dict.get("lang_primary_is_javascript", 0.0)
        lang_c = features_dict.get("lang_primary_is_c", 0.0)
        lang_java = features_dict.get("lang_primary_is_java_kotlin", 0.0)
        entropy = features_dict.get("lang_diversity_entropy", 0.0)

        # Compute competing shares from non-fork repos
        MOBILE_LANGS = {"Swift", "Dart", "Kotlin", "Objective-C"}
        ML_LANGS = {"Jupyter Notebook", "R", "Julia"}
        DEVOPS_LANGS = {"Shell", "HCL", "Dockerfile", "Makefile"}

        repos = [r for r in profile_data.get("repositories", []) if not r.get("is_fork", False)]
        n_orig = len(repos)
        counts: Dict[str, int] = {}
        for r in repos:
            l = r.get("language") or "None"
            counts[l] = counts.get(l, 0) + 1
        l_shares = {k: v / n_orig for k, v in counts.items()} if n_orig > 0 else {}

        top_competing_mob = max([v for k, v in l_shares.items() if k not in MOBILE_LANGS] or [0.0])
        top_competing_ml = max([v for k, v in l_shares.items() if k not in ML_LANGS] or [0.0])
        top_competing_devops = max([v for k, v in l_shares.items() if k not in DEVOPS_LANGS] or [0.0])

        def _is_valid_spec_sig(sig: float, base: float, comp: float, mult: float = 1.20, min_c: float = 0.10, dom_thresh: float = 0.55) -> bool:
            if sig <= base:
                return False
            ceil = max(min_c, base * mult)
            if sig >= ceil:
                return True
            if comp > dom_thresh:
                return False
            return True

        is_ml = _is_valid_spec_sig(lang_ml, 0.05, top_competing_ml)
        is_mob = _is_valid_spec_sig(lang_mob, 0.02, top_competing_mob)
        is_devops = _is_valid_spec_sig(lang_devops, 0.10, top_competing_devops)

        if is_ml:
            archetype = "ML Specialist"
        elif is_mob:
            archetype = "Mobile Developer"
        elif is_devops:
            archetype = "DevOps Engineer"
        elif lang_js == 1.0 or lang_front > 0.50:
            archetype = "Frontend Developer"
        elif lang_c == 1.0 or (lang_java == 1.0 and lang_mob <= 0.02):
            archetype = "Backend Developer"
        elif entropy >= 1.8:
            archetype = "Full-Stack Developer"
        else:
            archetype = "Unclassified / Low Signal"

        # -------------------------------------------------------------------
        # Step 5: Phase 7 SHAP Interpretability (Preserved XGBoost Model)
        # -------------------------------------------------------------------
        job["progress"] = "Generating SHAP feature attributions..."
        job["percent"] = 90

        xgb_model = joblib.load(MODEL_DIR / "xgboost_archetype_model.joblib")
        le = joblib.load(MODEL_DIR / "archetype_label_encoder.joblib")
        explainer = shap.TreeExplainer(xgb_model)

        shap_explanation = explain_prediction(
            feature_vector=pd.Series(scaled_vec[0], index=feature_names_in),
            model=xgb_model,
            explainer=explainer,
            le=le,
            top_n=5,
            target_class=archetype if archetype in le.classes_ else None
        )
        confidence = shap_explanation["confidence"]
        shap_top_features = shap_explanation["shap_top_features"]

        # -------------------------------------------------------------------
        # Step 6: Database Persistence (Tagged source="live_upload")
        # -------------------------------------------------------------------
        job["progress"] = "Persisting candidate profile to SQLite database..."
        job["percent"] = 95

        with get_session() as session:
            dev = upsert_developer(session, username=username, source="live_upload")
            snap = Snapshot(
                developer_id=dev.id,
                collected_at=datetime.utcnow(),
                raw_json_path=str(raw_path)
            )
            session.add(snap)
            session.flush()

            # Insert 52 Features
            insert_features(session, snapshot_id=snap.id, features=features_dict)

            # Insert Cluster Assignment
            upsert_cluster_assignment(
                session,
                snapshot_id=snap.id,
                cluster_id=cluster_id,
                distance_to_centroid=dist_to_center
            )

            # Insert Archetype Prediction & SHAP
            insert_archetype_prediction(
                session,
                snapshot_id=snap.id,
                archetype_label=archetype,
                confidence=confidence,
                shap_top_features=shap_top_features
            )

            # Insert Quality Scores
            upsert_doc_score(
                session,
                snapshot_id=snap.id,
                score=features_dict.get("doc_score", 0.0),
                components=doc_comp
            )
            upsert_eng_maturity_score(
                session,
                snapshot_id=snap.id,
                score=features_dict.get("eng_maturity_score", 0.0),
                components=eng_comp
            )

            session.commit()

        # -------------------------------------------------------------------
        # Complete
        # -------------------------------------------------------------------
        job["status"] = "done"
        job["stage"] = "done"
        job["progress"] = f"Successfully collected and profiled candidate '{username}'!"
        job["percent"] = 100
        job["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"Job {job_id} for '{username}' successfully completed.")

    except Exception as e:
        logger.error(f"Error in job {job_id} for '{username}': {e}", exc_info=True)
        job = JOBS.get(job_id)
        if job:
            job["status"] = "failed"
            job["stage"] = "failed"
            job["error"] = str(e)
            job["progress"] = f"Pipeline failed: {str(e)}"
            job["completed_at"] = datetime.utcnow().isoformat()
