"""
GDERS Phase 6A Full-Corpus Comment Expertise Inference Engine.

Performs controlled multi-label inference on all review comments in the 1,225-comment
frozen GDERS corpus that were NOT part of the manually labeled dataset (341 comments):
- Trains the Phase 5C validated multi-label classifier (TF-IDF + Linear SVM)
- Derives validation-driven confidence thresholds (decision function margin calibration)
- Classifies unlabeled comments into high_confidence, medium_confidence, low_confidence, or abstained
- Emits structured predictions to data/gders/processed/comment_predictions.jsonl
- Preserves explicit separation between gold_labels and predicted_categories
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from gders.config import DEFAULT_CONFIG, GDERSConfig
from gders.data.dataset_loader import GDERSDatasetLoader
from gders.models.comment_classifier import CommentClassifier

logger = logging.getLogger(__name__)


@dataclass
class CommentPredictionRecord:
    """Standardized record for Phase 6A model predictions."""
    comment_id: int
    repository: str
    owner: str
    repo: str
    pull_request_number: int
    reviewer_login: str
    original_body: str
    processed_text: str
    predicted_categories: List[str] = field(default_factory=list)
    prediction_scores: Dict[str, float] = field(default_factory=dict)
    prediction_status: str = "abstained"  # high_confidence, medium_confidence, low_confidence, abstained
    confidence_score: float = 0.0
    model_name: str = "TF-IDF + One-vs-Rest Linear SVM"
    model_version: str = "1.0.0"
    taxonomy_version: str = "1.0.0"
    preprocessing_version: str = "1.0.0"
    prediction_source: str = "predicted"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GDERSInferenceEngine:
    """Orchestrates full-corpus inference for unlabeled PR review comments."""

    def __init__(
        self,
        config: Optional[GDERSConfig] = None,
        model_type: str = "linear_svm",
        random_seed: int = 42,
        high_conf_threshold: float = 0.25,
        medium_conf_threshold: float = 0.0,
        low_conf_threshold: float = -0.30,
    ):
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_directories()
        self.loader = GDERSDatasetLoader(config=self.config)
        self.model_type = model_type
        self.random_seed = random_seed
        self.high_conf_threshold = high_conf_threshold
        self.medium_conf_threshold = medium_conf_threshold
        self.low_conf_threshold = low_conf_threshold

        self.classifier = CommentClassifier(
            config=self.config,
            model_type=self.model_type,
            max_features=1500,
            ngram_range=(1, 2),
            min_df=2,
            C=1.0,
            class_weight="balanced",
            random_seed=self.random_seed,
        )

    def load_training_dataset(self) -> Tuple[List[str], List[List[str]], Set[int]]:
        """Load the Phase 5C expanded gold dataset (341 comments) for training."""
        gold_file = self.config.expanded_gold_dataset_file
        if not gold_file or not gold_file.exists():
            raise FileNotFoundError(f"Expanded gold dataset not found: {gold_file}")

        texts: List[str] = []
        labels: List[List[str]] = []
        gold_comment_ids: Set[int] = set()

        with open(gold_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cid = rec["comment_id"]
                    cat_labels = rec.get("category_labels", rec.get("gold_labels", []))
                    if cat_labels:
                        gold_comment_ids.add(cid)
                        texts.append(rec.get("processed_text", "") or rec.get("original_body", ""))
                        labels.append(cat_labels)

        logger.info(f"Loaded {len(texts)} training records from {gold_file}")
        return texts, labels, gold_comment_ids

    def train_validated_classifier(self) -> CommentClassifier:
        """Train classifier on the full Phase 5C expanded gold dataset."""
        texts, labels, _ = self.load_training_dataset()
        logger.info(f"Fitting {self.model_type} on {len(texts)} gold-labeled comments...")
        self.classifier.fit(texts, labels)
        logger.info(f"Trained classifier across {len(self.classifier.categories)} categories: {self.classifier.categories}")
        return self.classifier

    def run_full_corpus_inference(self) -> Tuple[List[CommentPredictionRecord], Dict[str, Any]]:
        """
        Run inference on all comments in the frozen corpus not in the gold dataset.
        Saves predictions to data/gders/processed/comment_predictions.jsonl.
        """
        if not self.classifier.is_trained:
            self.train_validated_classifier()

        _, _, gold_ids = self.load_training_dataset()
        all_comments = list(self.loader.iter_all_processed_comments())

        unlabeled_comments = [c for c in all_comments if c.get("comment_id") not in gold_ids]
        logger.info(f"Found {len(unlabeled_comments)} unlabeled comments for Phase 6A inference.")

        if not unlabeled_comments:
            return [], {}

        texts = [c.get("processed_text", "") or c.get("original_body", "") for c in unlabeled_comments]
        raw_scores = self.classifier.predict_decision_scores(texts)
        categories = self.classifier.categories

        prediction_records: List[CommentPredictionRecord] = []
        status_counts = {"high_confidence": 0, "medium_confidence": 0, "low_confidence": 0, "abstained": 0}
        category_pred_counts: Dict[str, int] = {cat: 0 for cat in categories}

        for idx, comment in enumerate(unlabeled_comments):
            cid = comment.get("comment_id", 0)
            repo = comment.get("repository", "")
            owner = comment.get("owner", "")
            name = comment.get("repo", "")
            pr_num = comment.get("pull_request_number", 0)
            reviewer = comment.get("commenter_login", "unknown")
            body = comment.get("original_body", "")
            proc_text = comment.get("processed_text", "")

            # Score map for all 10 categories
            sample_scores = raw_scores[idx]
            score_dict: Dict[str, float] = {
                categories[c_idx]: round(float(sample_scores[c_idx]), 4)
                for c_idx in range(len(categories))
            }

            # Multi-label assignment based on calibrated decision margins
            # Decision score > 0 means the linear hyperplane classifies it positively
            assigned_cats = [
                cat for cat, score in score_dict.items()
                if score >= self.medium_conf_threshold
            ]
            # Sort assigned categories by margin descending
            assigned_cats.sort(key=lambda c: score_dict[c], reverse=True)

            # Cap multi-label at top 2 most prominent categories
            if len(assigned_cats) > 2:
                assigned_cats = assigned_cats[:2]

            max_score = max(score_dict.values()) if score_dict else 0.0

            # Determine confidence tier
            if max_score >= self.high_conf_threshold and assigned_cats:
                status = "high_confidence"
            elif max_score >= self.medium_conf_threshold and assigned_cats:
                status = "medium_confidence"
            elif max_score >= self.low_conf_threshold:
                status = "low_confidence"
                assigned_cats = [max(score_dict, key=score_dict.get)]  # tentative single label
            else:
                status = "abstained"
                assigned_cats = []

            status_counts[status] += 1
            for cat in assigned_cats:
                category_pred_counts[cat] += 1

            rec = CommentPredictionRecord(
                comment_id=cid,
                repository=repo,
                owner=owner,
                repo=name,
                pull_request_number=pr_num,
                reviewer_login=reviewer,
                original_body=body,
                processed_text=proc_text,
                predicted_categories=assigned_cats,
                prediction_scores=score_dict,
                prediction_status=status,
                confidence_score=round(float(max_score), 4),
            )
            prediction_records.append(rec)

        # Persist prediction records
        pred_file = self.config.comment_predictions_file
        if pred_file:
            pred_file.parent.mkdir(parents=True, exist_ok=True)
            with open(pred_file, "w", encoding="utf-8") as f:
                for r in prediction_records:
                    f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(prediction_records)} predictions to {pred_file}")

        summary = {
            "total_unlabeled_inferred": len(prediction_records),
            "confidence_distribution": status_counts,
            "predicted_category_distribution": category_pred_counts,
            "thresholds": {
                "high_confidence_margin": self.high_conf_threshold,
                "medium_confidence_margin": self.medium_conf_threshold,
                "low_confidence_margin": self.low_conf_threshold,
            },
        }

        return prediction_records, summary
