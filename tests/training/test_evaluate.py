import pytest

from training.evaluate import compare_metrics


@pytest.mark.parametrize(
    "input_metrics,threshold_metrics,evaluation_passed",
    [
        (
            {"acc": 0.8},
            {"acc": 0.8, "precision": 0.5, "recall": 0.4},
            False
        ),
        (
            {"acc": 0.9, "eval_loss": 0.9},
            {"acc": 0.8},
            True
        ),
        (
            {"acc": 0.6},
            {"acc": 0.8},
            False
        ),
        (
            {"acc": 0.6},
            {"acc": 0.5},
            True
        ),
        (
            {"acc": 0.8, "precision": 0.4, "recall": 0.4},
            {"acc": 0.8, "precision": 0.5, "recall": 0.4},
            False
        ),
        (
            {"acc": 0.8, "precision": 0.5, "recall": 0.4},
            {"acc": 0.8, "precision": 0.5, "recall": 0.4},
            True
        ),
    ],
    ids=[
        "missing_metrics",
        "extra_training_metrics",
        "single_metric_unsatisfied",
        "single_metric_satisfied",
        "multiple_metrics_unsatisfied",
        "multiple_metrics_satisfied",
    ]
)
def test_evaluate_metric(input_metrics, threshold_metrics, evaluation_passed):
    eval = compare_metrics(input_metrics, threshold_metrics)
    assert eval["passed"] is evaluation_passed
