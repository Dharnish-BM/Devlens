"""
GDERS Comment Classifier Interface.

Responsible for categorizing PR review comments into expertise taxonomy categories
(e.g., Architecture, Bug Fixing, Testing, Performance, Style).
"""

from typing import Any, Dict, List, Optional
from gders.config import DEFAULT_CONFIG, GDERSConfig


class CommentClassifier:
    """Interface for classifying PR review comments into expertise categories."""

    def __init__(self, config: Optional[GDERSConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.categories: List[str] = list(self.config.expertise_categories)
        self.is_trained: bool = False

    def train(self, comments: List[str], labels: List[str]) -> Dict[str, Any]:
        """Train the comment classifier on labeled review comments (to be implemented in future phase)."""
        raise NotImplementedError("Comment classification training will be implemented in a future phase.")

    def predict_category(self, comment_text: str) -> str:
        """Predict the primary expertise category for a given comment text."""
        raise NotImplementedError("Comment prediction will be implemented in a future phase.")

    def predict_proba(self, comment_text: str) -> Dict[str, float]:
        """Predict probability distribution across all expertise categories."""
        raise NotImplementedError("Probability prediction will be implemented in a future phase.")
