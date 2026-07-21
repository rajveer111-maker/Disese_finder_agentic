"""
Utilities for logging model performance metrics for monitoring.
Writes JSON lines to a metrics log file for later analysis.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def log_metrics(
    model_name: str,
    dataset_name: str,
    metrics: Dict[str, Any],
    confusion_matrix: Optional[List[List[int]]] = None,
    extra: Optional[Dict[str, Any]] = None,
    output_path: str = "monitoring/metrics_log.jsonl",
) -> None:
    """Append a metrics record to the monitoring log.

    Args:
        model_name: Human-readable model name or key
        dataset_name: Name of the dataset evaluated
        metrics: Dictionary of metric name to value (accuracy, precision, recall, f1, etc.)
        confusion_matrix: Optional confusion matrix as list of lists (rows=true labels)
        extra: Optional extra fields
        output_path: Path to the JSONL log file
    """
    _ensure_dir(output_path)
    record: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "dataset": dataset_name,
        "metrics": metrics,
    }
    if confusion_matrix is not None:
        record["confusion_matrix"] = confusion_matrix
    if extra:
        record.update(extra)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


