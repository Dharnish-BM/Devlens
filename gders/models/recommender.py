"""
GDERS Phase 7A Expert Developer Recommendation Engine.

Matches and ranks candidate developers from the frozen GDERS corpus based on
multi-category and natural-language expertise requirements:
- Deterministic, auditable query-to-taxonomy mapping
- Enforces Phase 6.5 recommendation-eligibility contract (recommendation_eligible == True)
- Strict exclusion of low-confidence and abstained predictions from recommendation scores
- Transparent multi-dimensional scoring formula with PR & Repository diversity adjustment
- Full comment-level explanation and traceability for every candidate recommendation
"""

import collections
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from gders.config import DEFAULT_CONFIG, GDERSConfig
from gders.data.annotation_manager import VALID_TAXONOMY_CATEGORIES
from gders.data.dataset_loader import GDERSDatasetLoader
from gders.models.profile_builder import DeveloperExpertiseProfile, ProfileBuilder

logger = logging.getLogger(__name__)

# Natural Language Mapping Vocabulary / Domain Synonyms
NATURAL_LANGUAGE_TAXONOMY_KEYWORDS: Dict[str, Set[str]] = {
    "ARCH_DESIGN": {
        "architecture", "design", "pattern", "decouple", "interface", "abstract",
        "hierarchy", "modularity", "encapsulate", "contract", "responsibility",
        "layer", "refactor", "lifecycle", "solid", "coupling", "microservice",
        "domain", "clean architecture"
    },
    "TESTING_QUALITY": {
        "test", "testing", "unit test", "integration test", "qa", "quality",
        "mock", "fixture", "assert", "assertion", "coverage", "regression",
        "spec", "flaky", "pytest", "junit", "e2e"
    },
    "PERF_OPTIMIZATION": {
        "performance", "optimize", "optimization", "memory", "leak", "alloc",
        "allocation", "latency", "throughput", "gc", "garbage collection",
        "benchmark", "lookup", "hashset", "cache", "caching", "speed",
        "overhead", "bottleneck", "profiling", "cpu"
    },
    "SECURITY_PRIVACY": {
        "security", "privacy", "vulnerability", "sanitize", "sanitization",
        "escape", "auth", "authentication", "authorization", "token",
        "permission", "csrf", "xss", "injection", "crypto", "cryptography",
        "tls", "certificate", "credential", "secret", "owasp", "encryption"
    },
    "DATA_MANAGEMENT": {
        "database", "sql", "query", "index", "indexing", "postgres", "mysql",
        "migration", "orm", "serialize", "serialization", "deserialization",
        "schema", "column", "entity", "table", "transaction", "record",
        "protobuf", "json storage", "data store", "nosql"
    },
    "INFRA_DEVOPS": {
        "infrastructure", "devops", "docker", "kubernetes", "k8s", "container",
        "helm", "ci", "cd", "pipeline", "workflow", "build", "gradle", "bazel",
        "cmake", "makefile", "monitoring", "telemetry", "prometheus", "deployment"
    },
    "FRONTEND_UI_UX": {
        "frontend", "ui", "ux", "widget", "render", "rendering", "css",
        "layout", "react", "component", "dom", "animation", "accessibility",
        "a11y", "client", "flutter", "web interface", "html"
    },
    "DOCUMENTATION": {
        "documentation", "doc", "docs", "docstring", "javadoc", "godoc",
        "readme", "comment", "explaining", "clarify", "user guide", "api reference"
    },
    "CODE_STYLE": {
        "style", "formatting", "lint", "linter", "naming", "convention",
        "casing", "unused import", "indent", "whitespace", "pep8", "clean code"
    },
    "BUG_LOGIC": {
        "bug", "fix", "error", "exception", "crash", "null", "nil", "panic",
        "loop", "race condition", "deadlock", "logic", "debugging", "fault",
        "unhandled", "throw", "breakage"
    },
}


@dataclass
class RecommendationRequest:
    """Task or project expertise requirement query for GDERS recommendations."""
    task_id: str = ""
    target_repository: Optional[str] = None
    required_categories: List[str] = field(default_factory=list)
    query: Optional[str] = None
    top_k: int = 5
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class DeveloperRecommendation:
    """Individual candidate recommendation result with score and justification."""
    developer_login: str
    score: float
    matched_categories: List[str]
    rank: int
    rationale: str


@dataclass
class RecommendationCandidate:
    """Individual candidate recommendation result with audit trail and explanation."""
    username: str
    recommendation_score: float
    matched_categories: List[str]
    category_evidence: Dict[str, Dict[str, Any]]
    recommendation_eligible_score: float
    evidence_tier: str
    identity_class: str = "human_candidate"  # 'human_candidate', 'bot_or_service_account', 'uncertain'
    developer_recommendation_eligible: bool = True
    gold_count: int = 0
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    distinct_prs: int = 0
    distinct_repositories: int = 0
    supporting_comment_ids: List[int] = field(default_factory=list)
    repositories: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationResponse:
    """Structured response container for GDERS recommendations."""
    query: str
    matched_categories: List[str]
    query_mapping_rationale: Dict[str, str]
    status: str  # 'ok', 'no_eligible_candidates', 'unmatched_query', 'invalid_query'
    total_candidates_found: int
    candidates: List[RecommendationCandidate] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["candidates"] = [c.to_dict() if hasattr(c, "to_dict") else c for c in self.candidates]
        return d


class GDERSRecommender:
    """GDERS Expert Developer Recommendation Engine."""

    def __init__(
        self,
        config: Optional[GDERSConfig] = None,
        profiles: Optional[List[DeveloperExpertiseProfile]] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_directories()
        self.loader = GDERSDatasetLoader(config=self.config)
        self.profiles: List[DeveloperExpertiseProfile] = profiles if profiles is not None else []
        if profiles is None:
            self._load_profiles()

    def _load_profiles(self) -> None:
        """Load developer expertise profiles from disk or compute them if absent."""
        prof_file = self.config.developer_expertise_profiles_file
        if prof_file and prof_file.exists():
            profiles = []
            with open(prof_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        profiles.append(DeveloperExpertiseProfile(**d))
            self.profiles = profiles
        else:
            builder = ProfileBuilder(config=self.config)
            self.profiles, _ = builder.build_all_developer_profiles()

    def map_query_to_categories(self, query: Any) -> Tuple[List[str], Dict[str, str], str]:
        """
        Map taxonomy category identifiers, lists, or natural-language strings
        deterministically to valid GDERS taxonomy categories.
        """
        if not query:
            return [], {}, "Empty query provided."

        matched_categories: Set[str] = set()
        rationales: Dict[str, str] = {}

        # 1. Handle direct taxonomy category list or string
        if isinstance(query, (list, tuple, set)):
            for item in query:
                item_str = str(item).strip().upper()
                if item_str in VALID_TAXONOMY_CATEGORIES:
                    matched_categories.add(item_str)
                    rationales[item_str] = f"Exact taxonomy category identifier '{item_str}'."
                else:
                    # Try natural language mapping on the item
                    sub_cats, sub_rat, _ = self._map_text_to_categories(str(item))
                    for sc in sub_cats:
                        matched_categories.add(sc)
                        rationales[sc] = sub_rat.get(sc, f"Matched via phrase '{item}'.")
            return sorted(matched_categories), rationales, "ok" if matched_categories else "No taxonomy categories matched."

        # 2. Handle string query
        query_str = str(query).strip()
        if not query_str:
            return [], {}, "Empty query string."

        # Direct category check
        if query_str.upper() in VALID_TAXONOMY_CATEGORIES:
            cat = query_str.upper()
            return [cat], {cat: f"Exact taxonomy category identifier '{cat}'."}, "ok"

        # Natural language mapping
        return self._map_text_to_categories(query_str)

    def _map_text_to_categories(self, text: str) -> Tuple[List[str], Dict[str, str], str]:
        """Map free-form natural language text to taxonomy categories."""
        text_lower = text.lower().strip()
        normalized_words = set(re.findall(r"\b[a-z0-9_-]+\b", text_lower))

        matched_categories: Set[str] = set()
        rationales: Dict[str, str] = {}

        for cat, keywords in NATURAL_LANGUAGE_TAXONOMY_KEYWORDS.items():
            matched_kws = []
            for kw in keywords:
                if " " in kw:
                    if kw in text_lower:
                        matched_kws.append(kw)
                else:
                    if kw in normalized_words:
                        matched_kws.append(kw)

            if matched_kws:
                matched_categories.add(cat)
                rationales[cat] = f"Matched natural-language keywords: {', '.join(sorted(matched_kws)[:4])}"

        if not matched_categories:
            return [], {}, f"Unable to map natural-language query '{text}' to validated GDERS taxonomy."

        return sorted(matched_categories), rationales, "ok"

    def compute_candidate_recommendation_score(
        self,
        profile: DeveloperExpertiseProfile,
        target_categories: List[str],
        require_human: bool = True,
    ) -> Optional[RecommendationCandidate]:
        """
        Compute candidate recommendation score for a developer matching target categories.
        Enforces candidate contract gates:
        - Must have identity_class == 'human_candidate' (unless require_human=False for audit)
        - Must have recommendation_eligible == True in at least one target category
        - Recommendation score:
            Score = Σ (category_eligible_score_i) * Category_Coverage_Multiplier
            Coverage Multiplier = 1.0 + 0.25 * (matched_target_categories - 1)
        """
        # Identity contract gate: bots and uncertain identities excluded from human developer recommendations
        if require_human and getattr(profile, "identity_class", "human_candidate") != "human_candidate":
            return None

        matched_eligible_cats: List[str] = []
        category_evidence_map: Dict[str, Dict[str, Any]] = {}

        total_eligible_score = 0.0
        total_gold = 0
        total_high = 0
        total_med = 0
        all_supporting_ids: Set[int] = set()
        all_repos: Set[str] = set()
        all_prs: Set[int] = set()
        highest_tier = "insufficient_evidence"

        tier_rank = {
            "insufficient_evidence": 0,
            "emerging_evidence": 1,
            "supported_evidence": 2,
            "strong_evidence": 3,
        }

        for cat in target_categories:
            cp_data = profile.category_profiles.get(cat)
            if not cp_data:
                continue

            is_eligible = cp_data.get("recommendation_eligible", False)
            el_score = cp_data.get("recommendation_eligible_score", 0.0)
            tier = cp_data.get("evidence_tier", "insufficient_evidence")

            # Collect evidence for reporting
            category_evidence_map[cat] = cp_data

            if is_eligible and el_score > 0:
                matched_eligible_cats.append(cat)
                total_eligible_score += el_score
                total_gold += cp_data.get("gold_comment_count", 0)
                total_high += cp_data.get("high_confidence_count", 0)
                total_med += cp_data.get("medium_confidence_count", 0)

                # Collect distinct comment IDs (ignoring low-confidence)
                el_cids = cp_data.get("recommendation_eligible_comment_ids", cp_data.get("supporting_comment_ids", []))
                all_supporting_ids.update(el_cids)

                if tier_rank.get(tier, 0) > tier_rank.get(highest_tier, 0):
                    highest_tier = tier

        # Candidate MUST be recommendation-eligible in at least one target category
        if not matched_eligible_cats or total_eligible_score <= 0:
            return None

        # Gather reviewer repositories & PRs associated with eligible evidence
        for r in profile.repositories_reviewed:
            all_repos.add(r)
        for p in profile.pull_requests_reviewed:
            all_prs.add(p)

        # Multi-category coverage multiplier
        coverage_mult = 1.0 + 0.25 * (len(matched_eligible_cats) - 1)
        final_recommendation_score = round(total_eligible_score * coverage_mult, 3)

        # Generate structured, transparent explanation
        cat_str = " & ".join(matched_eligible_cats)
        exp_parts = [
            f"Matched {cat_str} ({highest_tier})",
            f"supported by {total_gold} gold and {total_high + total_med} validated high/medium predictions",
            f"across {len(all_prs)} PRs in {len(all_repos)} repos.",
        ]
        explanation = " ".join(exp_parts)

        identity_cls = getattr(profile, "identity_class", "human_candidate")
        is_dev_eligible = (identity_cls == "human_candidate")

        return RecommendationCandidate(
            username=profile.developer_login,
            recommendation_score=final_recommendation_score,
            matched_categories=matched_eligible_cats,
            category_evidence=category_evidence_map,
            recommendation_eligible_score=round(total_eligible_score, 3),
            evidence_tier=highest_tier,
            identity_class=identity_cls,
            developer_recommendation_eligible=is_dev_eligible,
            gold_count=total_gold,
            high_confidence_count=total_high,
            medium_confidence_count=total_med,
            distinct_prs=len(all_prs),
            distinct_repositories=len(all_repos),
            supporting_comment_ids=sorted(all_supporting_ids),
            repositories=sorted(all_repos),
            explanation=explanation,
        )

    def recommend(
        self,
        expertise_query: Any,
        top_k: int = 5,
        require_human: bool = True,
    ) -> RecommendationResponse:
        """
        Rank and recommend developers matching target expertise requirements.
        - Supports taxonomy category list or natural-language query
        - Filters candidates by identity_class == 'human_candidate' when require_human=True
        - Returns structured RecommendationResponse with audit trail
        """
        matched_cats, rationales, map_status = self.map_query_to_categories(expertise_query)

        if not matched_cats:
            return RecommendationResponse(
                query=str(expertise_query),
                matched_categories=[],
                query_mapping_rationale=rationales,
                status="unmatched_query" if map_status != "Empty query provided." else "invalid_query",
                total_candidates_found=0,
                candidates=[],
            )

        candidates: List[RecommendationCandidate] = []

        for profile in self.profiles:
            cand = self.compute_candidate_recommendation_score(
                profile,
                matched_cats,
                require_human=require_human,
            )
            if cand is not None:
                candidates.append(cand)

        # Deterministic sorting:
        # 1. recommendation_score descending
        # 2. matched_categories count descending
        # 3. gold_count descending
        # 4. username ascending (tie-breaker)
        candidates.sort(
            key=lambda c: (
                -c.recommendation_score,
                -len(c.matched_categories),
                -c.gold_count,
                c.username,
            )
        )

        ranked_candidates = candidates[:max(1, top_k)]
        status = "ok" if candidates else "no_eligible_candidates"

        return RecommendationResponse(
            query=str(expertise_query),
            matched_categories=matched_cats,
            query_mapping_rationale=rationales,
            status=status,
            total_candidates_found=len(candidates),
            candidates=ranked_candidates,
        )

    def generate_recommendation_engine_report(self) -> Dict[str, Any]:
        """Generate machine-readable audit report for the recommendation engine."""
        categories = sorted(VALID_TAXONOMY_CATEGORIES)
        cat_recommendation_counts = collections.Counter()

        sample_queries = [
            "TESTING_QUALITY",
            ["BUG_LOGIC", "TESTING_QUALITY"],
            "Need a developer skilled in database schema design and SQL optimization",
            "Looking for security engineer to sanitize input tokens and audit auth vulnerabilities",
            "Frontend UI widget rendering and accessibility in Flutter React",
            "UnknownNonExistentCategoryQuery",
        ]

        sample_outputs = []
        for q in sample_queries:
            resp = self.recommend(q, top_k=3)
            sample_outputs.append({
                "query": resp.query,
                "status": resp.status,
                "matched_categories": resp.matched_categories,
                "candidates_returned": len(resp.candidates),
                "top_candidates": [
                    {
                        "username": c.username,
                        "score": c.recommendation_score,
                        "tier": c.evidence_tier,
                        "matched_categories": c.matched_categories,
                        "explanation": c.explanation,
                    }
                    for c in resp.candidates
                ]
            })

        for cat in categories:
            resp = self.recommend(cat, top_k=100)
            cat_recommendation_counts[cat] = len(resp.candidates)

        human_candidates = sum(1 for p in self.profiles if getattr(p, "identity_class", "human_candidate") == "human_candidate")
        bot_candidates = sum(1 for p in self.profiles if getattr(p, "identity_class", "human_candidate") == "bot_or_service_account")
        uncertain_candidates = sum(1 for p in self.profiles if getattr(p, "identity_class", "human_candidate") == "uncertain")
        human_eligible = sum(1 for p in self.profiles if getattr(p, "developer_recommendation_eligible", False))
        total_eligible = sum(1 for p in self.profiles if p.has_sufficient_evidence)

        report = {
            "phase": "Phase 7A — GDERS Expert Developer Recommendation Engine",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "candidate_pool_summary": {
                "total_reviewers_in_corpus": len(self.profiles),
                "human_candidates": human_candidates,
                "bots_or_service_accounts": bot_candidates,
                "uncertain_identities": uncertain_candidates,
                "expertise_eligible_reviewers_all": total_eligible,
                "human_developer_recommendation_eligible": human_eligible,
            },
            "eligibility_contract": {
                "identity_gate": "identity_class == 'human_candidate' (bots and uncertain accounts strictly excluded)",
                "expertise_gate": "recommendation_eligible == True (at least one target category in 'supported_evidence' or 'strong_evidence')",
                "low_confidence_policy": "Excluded from recommendation score and eligibility computation",
                "abstained_policy": "Excluded from recommendation score",
            },
            "scoring_formula": {
                "category_score": "recommendation_eligible_score (Gold * 1.0 + High * 0.7 + Med * 0.4) * PR_Multiplier * Repo_Multiplier",
                "multi_category_coverage_multiplier": "1.0 + 0.25 * (matched_categories - 1)",
                "tie_breakers": ["recommendation_score DESC", "matched_categories_count DESC", "gold_count DESC", "username ASC"],
            },
            "category_candidate_availability": dict(cat_recommendation_counts),
            "sample_recommendation_evaluations": sample_outputs,
        }

        report_file = self.config.recommendation_engine_report_file
        if report_file:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved recommendation engine report to {report_file}")

        return report
