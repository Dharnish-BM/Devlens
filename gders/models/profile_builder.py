"""
GDERS Developer Profile Builder Interface.

Constructs multi-dimensional dynamic developer expertise profiles from
classified PR review comments, recency weighting, and repository context.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from gders.config import DEFAULT_CONFIG, GDERSConfig


@dataclass
class DeveloperExpertiseProfile:
    """Developer Expertise Profile representation in GDERS."""
    developer_login: str
    total_comments_analyzed: int
    category_distribution: Dict[str, float] = field(default_factory=dict)
    top_expertise_categories: List[str] = field(default_factory=list)
    contributed_repositories: List[str] = field(default_factory=list)
    recent_activity_score: float = 0.0


class ProfileBuilder:
    """Interface for synthesizing developer expertise profiles from review comments."""

    def __init__(self, config: Optional[GDERSConfig] = None):
        self.config = config or DEFAULT_CONFIG

    def build_profile(
        self,
        developer_login: str,
        classified_comments: List[Dict[str, Any]],
    ) -> DeveloperExpertiseProfile:
        """Construct an expertise profile for a specific developer from their classified review comments."""
        raise NotImplementedError("Profile building will be implemented in a future phase.")

    def save_profile(self, profile: DeveloperExpertiseProfile) -> None:
        """Persist an expertise profile into data/gders/expertise_profiles/."""
        raise NotImplementedError("Profile persistence will be implemented in a future phase.")
