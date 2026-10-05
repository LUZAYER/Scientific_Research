"""
Evaluation Metrics for Hallucination Detection.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR, and latency efficiency.
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class EvaluationMetrics:
    """Computes comprehensive classification and efficiency metrics."""

    @staticmethod
    def compute_all(
        y_true: List[int],
        y_pred: List[int],
        y_scores: List[float],
        latencies: List[float] = None
    ) -> Dict[str, float]:
        """Calculates standard classification metrics and efficiency stats."""
        if not y_true or len(y_true) == 0:
            return {}

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        y_scores_arr = np.array(y_scores)

        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

        # ROC-AUC requires both positive and negative classes
        try:
            if len(np.unique(y_true_arr)) > 1:
                auc = float(roc_auc_score(y_true_arr, y_scores_arr))
            else:
                auc = 0.5
        except Exception:
            auc = 0.5

        # Confusion Matrix for FPR and FNR
        try:
            tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        except Exception:
            fpr, fnr = 0.0, 0.0

        avg_latency = float(np.mean(latencies)) if latencies else 0.0

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "avg_latency_sec": round(avg_latency, 4)
        }
