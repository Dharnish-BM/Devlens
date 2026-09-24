"""
GDERS Expertise-Based Recommender Interface.

Matches and ranks candidate developers for target software development tasks,
pull requests, or global software development project requirements.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from gders.config import DEFAULT_CONFIG, GDERSConfig
from gders.models.profile_builder import DeveloperExpertiseProfile


@dataclass
class RecommendationRequest:
    """Task or project expertise requirement query for GDERS recommendations."""
    task_id: str
    target_repository: Optional[str] = None
    required_categories: List[str] = field(default_factory=list)
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


class GDERSRecommender:
    """Interface for GDERS expertise-based recommendation algorithm."""

    def __init__(self, config: Optional[GDERSConfig] = None):
        self.config = config or DEFAULT_CONFIG

    def recommend_developers(
        self,
        request: RecommendationRequest,
        candidate_profiles: List[DeveloperExpertiseProfile],
    ) -> List[DeveloperRecommendation]:
        """Rank and recommend candidate developers according to task expertise requirements."""
        raise NotImplementedError("Recommendation algorithm will be implemented in a future phase.")
