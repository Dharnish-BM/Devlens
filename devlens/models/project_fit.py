"""
DevLens Phase 8: Project-Fit Matchmaking & Recommendation Engine.

- ProjectRequirement schema with strict weight sum validation
- Explicit technical-adjacency archetype partial-credit table
- Multi-component fit scoring:
  1. Archetype component (partial credit technical matrix)
  2. Language component (cosine similarity over byte-distribution vector)
  3. Documentation & Engineering Maturity component (continuous normalized fit)
- Human-readable component contribution explanations
- Developer pool ranking
- Naive rule-based baseline comparator (GDERS keyword-matching benchmark)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from devlens.db.models import ArchetypePrediction, Developer, DocScore, EngMaturityScore, Snapshot
from devlens.db.repository import get_all_current_features
from devlens.db.session import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical valid archetypes that can be requested
VALID_REQUIRED_ARCHETYPES: Set[str] = {
    "Frontend Developer",
    "Backend Developer",
    "Full-Stack Developer",
    "ML Specialist",
    "DevOps Engineer",
    "Mobile Developer",
}

# Explicit Technical-Adjacency Archetype Partial-Credit Matrix
# (Developer Archetype, Required Archetype) -> Partial Credit [0.0 - 1.0]
ARCHETYPE_CREDIT_MATRIX: Dict[Tuple[str, str], float] = {
    # Exact Matches (1.0)
    ("Frontend Developer", "Frontend Developer"): 1.0,
    ("Backend Developer", "Backend Developer"): 1.0,
    ("Full-Stack Developer", "Full-Stack Developer"): 1.0,
    ("ML Specialist", "ML Specialist"): 1.0,
    ("DevOps Engineer", "DevOps Engineer"): 1.0,
    ("Mobile Developer", "Mobile Developer"): 1.0,

    # Full-Stack Developer matching specific domain requirements (0.6)
    ("Full-Stack Developer", "Frontend Developer"): 0.6,
    ("Full-Stack Developer", "Backend Developer"): 0.6,
    ("Full-Stack Developer", "Mobile Developer"): 0.6,
    ("Full-Stack Developer", "DevOps Engineer"): 0.6,
    ("Full-Stack Developer", "ML Specialist"): 0.6,

    # Domain Specialists matching a Full-Stack requirement (0.4 - 0.5)
    ("Frontend Developer", "Full-Stack Developer"): 0.5,
    ("Backend Developer", "Full-Stack Developer"): 0.5,
    ("Mobile Developer", "Full-Stack Developer"): 0.4,
    ("DevOps Engineer", "Full-Stack Developer"): 0.4,
    ("ML Specialist", "Full-Stack Developer"): 0.4,

    # Frontend & Mobile UI overlap (0.3)
    ("Frontend Developer", "Mobile Developer"): 0.3,
    ("Mobile Developer", "Frontend Developer"): 0.3,

    # DevOps & Backend Infrastructure Adjacency (0.4)
    ("DevOps Engineer", "Backend Developer"): 0.4,
    ("Backend Developer", "DevOps Engineer"): 0.4,

    # Backend & ML Specialist Data Pipeline / Python Overlap (0.3)
    ("Backend Developer", "ML Specialist"): 0.3,
    ("ML Specialist", "Backend Developer"): 0.3,
}


@dataclass
class ProjectRequirement:
    """Schema defining project requirements for developer matchmaking."""
    name: str
    required_archetypes: List[str]
    min_doc_score: float
    min_eng_maturity_score: float
    required_languages: List[str]
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "archetype_match": 0.40,
            "language_similarity": 0.40,
            "doc_maturity_fit": 0.20,
        }
    )

    def __post_init__(self):
        # Validate weights presence and sum
        required_keys = {"archetype_match", "language_similarity", "doc_maturity_fit"}
        if set(self.weights.keys()) != required_keys:
            raise ValueError(f"Weights dict must contain exactly keys: {required_keys}, got {set(self.weights.keys())}")

        w_sum = sum(self.weights.values())
        if not np.isclose(w_sum, 1.0, atol=1e-5):
            raise ValueError(f"Weights must sum to 1.0, got {w_sum:.6f} (weights: {self.weights})")

        # Validate required archetypes (Unclassified/Low Signal is excluded from requirements)
        if not self.required_archetypes:
            raise ValueError("required_archetypes cannot be empty.")

        for arch in self.required_archetypes:
            if arch not in VALID_REQUIRED_ARCHETYPES:
                raise ValueError(
                    f"Invalid required archetype '{arch}'. Must be one of: {sorted(list(VALID_REQUIRED_ARCHETYPES))}"
                )


@dataclass
class DeveloperProfile:
    """Rich profile representation of a developer for matchmaking."""
    username: str
    assigned_archetype: str
    primary_language: str
    language_bytes: Dict[str, int]
    doc_score: float
    eng_maturity_score: float


def get_archetype_partial_credit(dev_archetype: str, required_archetypes: List[str]) -> float:
    """Compute maximum partial-credit across requested archetypes."""
    if dev_archetype == "Unclassified / Low Signal":
        return 0.0

    best_credit = 0.0
    for req_arch in required_archetypes:
        credit = ARCHETYPE_CREDIT_MATRIX.get((dev_archetype, req_arch), 0.0)
        if credit > best_credit:
            best_credit = credit
    return float(best_credit)


def compute_language_similarity(dev_lang_bytes: Dict[str, int], required_languages: List[str]) -> float:
    """Compute Cosine Similarity between developer language byte distribution and requirement vector."""
    if not required_languages:
        return 0.0

    # Clean and normalize language names to lower-case for robust matching
    dev_bytes_norm: Dict[str, float] = {}
    for lang, b in dev_lang_bytes.items():
        dev_bytes_norm[lang.strip().lower()] = float(b)

    total_dev_bytes = sum(dev_bytes_norm.values())
    if total_dev_bytes <= 0:
        return 0.0

    # Build target vector (uniform weight across required languages)
    req_langs_norm = [l.strip().lower() for l in required_languages if l.strip()]
    if not req_langs_norm:
        return 0.0

    req_vec_norm: Dict[str, float] = {l: 1.0 / len(req_langs_norm) for l in req_langs_norm}

    # Proportional developer distribution
    dev_prop = {l: b / total_dev_bytes for l, b in dev_bytes_norm.items()}

    # Compute dot product and norms over union of languages
    all_langs = set(dev_prop.keys()) | set(req_vec_norm.keys())
    u = np.array([dev_prop.get(l, 0.0) for l in all_langs])
    v = np.array([req_vec_norm.get(l, 0.0) for l in all_langs])

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u <= 0 or norm_v <= 0:
        return 0.0

    cosine_sim = float(np.dot(u, v) / (norm_u * norm_v))
    return max(0.0, min(1.0, cosine_sim))


def compute_doc_maturity_fit(
    doc_score: float,
    eng_score: float,
    min_doc_score: float,
    min_eng_maturity_score: float
) -> float:
    """Compute normalized linear continuous fit score for documentation and engineering maturity."""
    doc_ratio = 1.0 if min_doc_score <= 0 else min(1.0, doc_score / min_doc_score)
    eng_ratio = 1.0 if min_eng_maturity_score <= 0 else min(1.0, eng_score / min_eng_maturity_score)
    return max(0.0, min(1.0, (doc_ratio + eng_ratio) / 2.0))


def generate_fit_explanation(
    archetype_score: float,
    lang_score: float,
    doc_eng_score: float,
    dev: DeveloperProfile,
    req: ProjectRequirement
) -> str:
    """Generate human-readable explanation of fit score breakdown."""
    parts = []

    # Archetype commentary
    if archetype_score >= 0.99:
        parts.append(f"Exact archetype match ({dev.assigned_archetype})")
    elif archetype_score >= 0.50:
        parts.append(f"Partial archetype credit ({archetype_score:.2f} as {dev.assigned_archetype})")
    else:
        parts.append(f"No archetype match ({dev.assigned_archetype})")

    # Language commentary
    if lang_score >= 0.70:
        parts.append(f"strong language match ({lang_score:.2f})")
    elif lang_score >= 0.30:
        parts.append(f"moderate language overlap ({lang_score:.2f})")
    else:
        parts.append(f"low language similarity ({lang_score:.2f})")

    # Doc & Maturity commentary
    doc_ok = dev.doc_score >= req.min_doc_score
    eng_ok = dev.eng_maturity_score >= req.min_eng_maturity_score
    if doc_ok and eng_ok:
        parts.append("exceeds doc/maturity targets")
    elif doc_ok:
        parts.append("meets doc target (eng below threshold)")
    elif eng_ok:
        parts.append("meets eng maturity target (doc below threshold)")
    else:
        parts.append(f"below doc/maturity targets (fit={doc_eng_score:.2f})")

    return "; ".join(parts)


def compute_fit_score(dev: DeveloperProfile, req: ProjectRequirement) -> Dict[str, Any]:
    """Compute overall weighted fit score and detailed component breakdown."""
    arch_score = get_archetype_partial_credit(dev.assigned_archetype, req.required_archetypes)
    lang_score = compute_language_similarity(dev.language_bytes, req.required_languages)
    doc_eng_score = compute_doc_maturity_fit(
        dev.doc_score, dev.eng_maturity_score, req.min_doc_score, req.min_eng_maturity_score
    )

    w = req.weights
    total_fit = (
        w["archetype_match"] * arch_score
        + w["language_similarity"] * lang_score
        + w["doc_maturity_fit"] * doc_eng_score
    )

    explanation = generate_fit_explanation(arch_score, lang_score, doc_eng_score, dev, req)

    return {
        "username": dev.username,
        "fit_score": float(total_fit),
        "archetype_score": float(arch_score),
        "language_score": float(lang_score),
        "doc_eng_score": float(doc_eng_score),
        "assigned_archetype": dev.assigned_archetype,
        "primary_language": dev.primary_language,
        "doc_score": dev.doc_score,
        "eng_maturity_score": dev.eng_maturity_score,
        "explanation": explanation,
    }


def rank_developers(req: ProjectRequirement, developer_pool: List[DeveloperProfile]) -> List[Dict[str, Any]]:
    """Rank developer pool by fit_score descending with explanations."""
    results = [compute_fit_score(dev, req) for dev in developer_pool]
    results.sort(key=lambda x: x["fit_score"], reverse=True)

    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return results


def rule_based_match(req: ProjectRequirement, developer_pool: List[DeveloperProfile]) -> List[Dict[str, Any]]:
    """Naive baseline comparator (simulating GDERS keyword/binary matching).
    
    Match Criteria:
    - Primary language string in req.required_languages (case-insensitive)
    - AND assigned archetype in req.required_archetypes
    """
    req_langs_norm = {l.strip().lower() for l in req.required_languages}
    matches = []

    for dev in developer_pool:
        primary_match = dev.primary_language.strip().lower() in req_langs_norm
        arch_match = dev.assigned_archetype in req.required_archetypes
        is_match = 1 if (primary_match and arch_match) else 0

        matches.append({
            "username": dev.username,
            "match": is_match,
            "assigned_archetype": dev.assigned_archetype,
            "primary_language": dev.primary_language,
            "doc_score": dev.doc_score,
            "eng_maturity_score": dev.eng_maturity_score,
        })

    # Sort matching developers first, then alphabetically
    matches.sort(key=lambda x: (-x["match"], x["username"]))
    for rank, m in enumerate(matches, start=1):
        m["rank"] = rank
    return matches


def load_developer_pool(data_raw_dir: str = "data/raw", source: Optional[str] = "consented_cohort") -> List[DeveloperProfile]:
    """Load full developer pool from database and raw JSON profiles."""
    raw_dir = Path(data_raw_dir)
    pool: List[DeveloperProfile] = []

    with get_session() as session:
        # Load developers, snapshots, doc scores, eng maturity scores, archetype predictions
        query = (
            session.query(
                Developer.username,
                Snapshot.id.label("snapshot_id"),
                DocScore.score.label("doc_score"),
                EngMaturityScore.score.label("eng_score"),
                ArchetypePrediction.archetype_label,
            )
            .join(Snapshot, Developer.id == Snapshot.developer_id)
            .outerjoin(DocScore, Snapshot.id == DocScore.snapshot_id)
            .outerjoin(EngMaturityScore, Snapshot.id == EngMaturityScore.snapshot_id)
            .outerjoin(ArchetypePrediction, Snapshot.id == ArchetypePrediction.snapshot_id)
        )
        if source is not None:
            query = query.filter(Developer.source == source)
        rows = query.all()

    # Deduplicate to latest snapshot per developer
    dev_data: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        uname = r.username
        if uname not in dev_data or r.snapshot_id > dev_data[uname]["snapshot_id"]:
            dev_data[uname] = {
                "snapshot_id": r.snapshot_id,
                "doc_score": r.doc_score or 0.0,
                "eng_score": r.eng_score or 0.0,
                "archetype_label": r.archetype_label or "Unclassified / Low Signal",
            }

    # Load raw JSONs for language byte distributions
    for uname, info in dev_data.items():
        json_path = raw_dir / f"{uname}.json"
        lang_bytes: Dict[str, int] = {}
        primary_lang = "Unknown"

        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                # Extract languages breakdown from graphql_data
                gql = raw_data.get("graphql_data") or {}
                lb = gql.get("languages_breakdown") or {}
                for repo_entries in lb.values():
                    if isinstance(repo_entries, list):
                        for entry in repo_entries:
                            if isinstance(entry, dict):
                                l_name = entry.get("language", "Unknown")
                                b_val = entry.get("size_bytes", 0) or 0
                                lang_bytes[l_name] = lang_bytes.get(l_name, 0) + b_val

                # Fallback to repo primary languages if gql breakdown is empty
                if not lang_bytes:
                    repos = raw_data.get("repositories") or []
                    for repo in repos:
                        if isinstance(repo, dict):
                            l_name = repo.get("language") or "Unknown"
                            lang_bytes[l_name] = lang_bytes.get(l_name, 0) + 1

                if lang_bytes:
                    primary_lang = max(lang_bytes, key=lambda k: lang_bytes[k])
            except Exception as e:
                logger.warning(f"Failed reading raw JSON for {uname}: {e}")

        profile = DeveloperProfile(
            username=uname,
            assigned_archetype=info["archetype_label"],
            primary_language=primary_lang,
            language_bytes=lang_bytes,
            doc_score=info["doc_score"],
            eng_maturity_score=info["eng_score"],
        )
        pool.append(profile)

    logger.info(f"Loaded {len(pool)} developer profiles for matchmaking.")
    return pool


def print_partial_credit_table() -> None:
    """Print the explicit technical-adjacency partial-credit matrix."""
    print("\n" + "=" * 80)
    print("  DEV LENS - ARCHETYPE TECHNICAL-ADJACENCY PARTIAL-CREDIT MATRIX")
    print("=" * 80)
    print(f"{'Developer Archetype':<26} {'Required Archetype':<26} {'Credit':<8} {'Rationale'}")
    print("-" * 80)

    # Sorted list of pairs
    for (dev_a, req_a), credit in sorted(ARCHETYPE_CREDIT_MATRIX.items(), key=lambda x: (-x[1], x[0][0])):
        if credit == 1.0:
            rat = "Exact domain match"
        elif dev_a == "Full-Stack Developer":
            rat = "Full-stack multi-domain breadth"
        elif req_a == "Full-Stack Developer":
            rat = "Domain specialist partial breadth"
        elif "Mobile" in dev_a and "Frontend" in req_a or "Frontend" in dev_a and "Mobile" in req_a:
            rat = "UI/client-side paradigm overlap"
        elif "DevOps" in dev_a and "Backend" in req_a or "Backend" in dev_a and "DevOps" in req_a:
            rat = "Infrastructure & server-side adjacency"
        elif "ML" in dev_a and "Backend" in req_a or "Backend" in dev_a and "ML" in req_a:
            rat = "Python / data engineering pipeline overlap"
        else:
            rat = "Adjacency credit"

        print(f"{dev_a:<26} {req_a:<26} {credit:<8.2f} {rat}")

    print("-" * 80)
    print("• Unclassified / Low Signal -> Any Requirement: 0.00 (Insufficient Signal)")
    print("• Any unlisted non-matching pair          -> 0.00 (Orthogonal Domains)")
    print("=" * 80 + "\n")


def run_benchmark_comparison():
    """Run matchmaking benchmark on 2 representative project requirements."""
    # 1. Print partial credit table
    print_partial_credit_table()

    # 2. Load full 188 developer pool
    pool = load_developer_pool()

    # 3. Define 2 Sample Project Requirements
    req1 = ProjectRequirement(
        name="Modern Web Client (Frontend Specialist)",
        required_archetypes=["Frontend Developer"],
        min_doc_score=0.40,
        min_eng_maturity_score=0.30,
        required_languages=["JavaScript", "TypeScript", "HTML", "CSS"],
        weights={"archetype_match": 0.40, "language_similarity": 0.40, "doc_maturity_fit": 0.20}
    )

    req2 = ProjectRequirement(
        name="Cloud Infrastructure & Automation (DevOps Specialist)",
        required_archetypes=["DevOps Engineer"],
        min_doc_score=0.30,
        min_eng_maturity_score=0.45,
        required_languages=["Dockerfile", "Shell", "Python", "Makefile", "HCL"],
        weights={"archetype_match": 0.35, "language_similarity": 0.35, "doc_maturity_fit": 0.30}
    )

    for req in [req1, req2]:
        print("=" * 85)
        print(f"  PROJECT REQUIREMENT BENCHMARK: {req.name.upper()}")
        print("=" * 85)
        print(f"Required Archetypes : {req.required_archetypes}")
        print(f"Required Languages  : {req.required_languages}")
        print(f"Quality Targets     : Doc Score >= {req.min_doc_score:.2f} | Eng Maturity >= {req.min_eng_maturity_score:.2f}")
        print(f"Component Weights   : {req.weights}\n")

        # Run DevLens Fit Score Ranking
        devlens_ranked = rank_developers(req, pool)

        # Run Naive Rule-Based Match (GDERS baseline)
        gders_matched = rule_based_match(req, pool)
        gders_match_count = sum(1 for m in gders_matched if m["match"] == 1)

        print(f"--- DevLens Continuous Matchmaking (Top 10 of {len(pool)}) ---")
        print(f"{'Rank':<5} {'Username':<22} {'Fit Score':<10} {'Arch':<6} {'Lang':<6} {'Doc/Eng':<8} {'Explanation'}")
        print("-" * 85)
        for r in devlens_ranked[:10]:
            print(
                f"#{r['rank']:<4} {r['username']:<22} {r['fit_score']:<10.4f} "
                f"{r['archetype_score']:<6.2f} {r['language_score']:<6.2f} {r['doc_eng_score']:<8.2f} "
                f"{r['explanation'][:38]}"
            )

        print(f"\n--- Naive Rule-Based Baseline (GDERS Keyword Match: {gders_match_count} Total Matches) ---")
        print(f"{'Rank':<5} {'Username':<22} {'Match?':<8} {'Primary Lang':<16} {'Assigned Archetype':<22}")
        print("-" * 85)
        for m in gders_matched[:10]:
            match_str = "MATCH (1)" if m["match"] == 1 else "NO (0)"
            print(f"#{m['rank']:<4} {m['username']:<22} {match_str:<8} {m['primary_language']:<16} {m['assigned_archetype']:<22}")

        print("\n" + "=" * 85 + "\n")


if __name__ == "__main__":
    run_benchmark_comparison()
