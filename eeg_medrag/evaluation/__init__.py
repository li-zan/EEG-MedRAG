"""Offline aggregation and resumable, reference-aware RAG evaluation."""

from .protocol import EvaluationConfig, PROTOCOL
from .runner import evaluate_answers, evaluate_retrieval, prepare_rubric

__all__ = [
    "EvaluationConfig", "PROTOCOL", "prepare_rubric", "evaluate_answers",
    "evaluate_retrieval",
]
