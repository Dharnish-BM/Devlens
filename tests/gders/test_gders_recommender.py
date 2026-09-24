"""
Unit tests for GDERS model interfaces, evaluation interfaces, and CLI.
"""

from gders.config import GDERSConfig
from gders.models.comment_classifier import CommentClassifier
from gders.models.profile_builder import ProfileBuilder, DeveloperExpertiseProfile
from gders.models.recommender import GDERSRecommender, RecommendationRequest
from gders.evaluation.benchmark import GDERSBenchmark
from gders.cli import build_parser, main


def test_gders_models_instantiation():
    config = GDERSConfig()
    classifier = CommentClassifier(config=config)
    builder = ProfileBuilder(config=config)
    recommender = GDERSRecommender(config=config)
    benchmark = GDERSBenchmark(config=config)

    assert isinstance(classifier.categories, list)
    assert classifier.is_trained is False
    assert isinstance(builder, ProfileBuilder)
    assert isinstance(recommender, GDERSRecommender)
    assert isinstance(benchmark, GDERSBenchmark)


def test_gders_cli_parser_and_inspect(capsys):
    parser = build_parser()
    assert parser is not None

    exit_code = main(["inspect"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "GDERS" in captured.out
    assert "apache/spark" in captured.out
