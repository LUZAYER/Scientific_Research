"""
Benchmark Runner Module.
Executes systematic comparisons between Baseline detectors (Confidence, Retrieval,
Semantic, Self-Verification) and the Proposed Hybrid Detector.
"""

import time
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

from src.data_loader import UnifiedRecord
from src.detector import HybridHallucinationDetector
from src.baselines import (
    ConfidenceOnlyBaseline,
    RetrievalOnlyBaseline,
    SemanticOnlyBaseline,
    SelfVerificationOnlyBaseline
)
from src.evaluation.metrics import EvaluationMetrics

class BenchmarkRunner:
    """Evaluates all detectors on benchmark datasets."""

    def __init__(self, use_online_wiki: bool = False):
        self.use_online_wiki = use_online_wiki
        self.models = {
            "Confidence-Only (B1)": ConfidenceOnlyBaseline(),
            "Retrieval-Only (B2)": RetrievalOnlyBaseline(use_online_wiki=use_online_wiki),
            "Semantic-Only (B3)": SemanticOnlyBaseline(),
            "SelfVerification-Only (B4)": SelfVerificationOnlyBaseline(),
            "Proposed Hybrid Framework": HybridHallucinationDetector(use_online_wiki=use_online_wiki)
        }

    def evaluate_model_on_dataset(
        self,
        model_name: str,
        model_instance: Any,
        records: List[UnifiedRecord],
        desc: str = ""
    ) -> Dict[str, Any]:
        """Evaluates a single detector model on a list of records."""
        y_true = []
        y_pred = []
        y_scores = []
        latencies = []

        for rec in tqdm(records, desc=f"{model_name} [{desc}]", leave=False):
            t0 = time.time()
            res = model_instance.detect(
                query=rec.query,
                response=rec.response,
                evidence=rec.grounding_evidence,
                samples=rec.stochastic_samples
            )
            elapsed = time.time() - t0
            
            ov = res["overall_verdict"]
            pred = 1 if ov["is_hallucination"] else 0
            score = ov["hallucination_score"]

            y_true.append(rec.label)
            y_pred.append(pred)
            y_scores.append(score)
            latencies.append(elapsed)

        metrics = EvaluationMetrics.compute_all(y_true, y_pred, y_scores, latencies)
        metrics["model"] = model_name
        metrics["num_samples"] = len(records)
        return metrics

    def run_benchmark(self, dataset_name: str, records: List[UnifiedRecord]) -> pd.DataFrame:
        """Runs all baseline models and the proposed hybrid detector on the dataset."""
        results = []
        for name, model in self.models.items():
            metrics = self.evaluate_model_on_dataset(name, model, records, desc=dataset_name)
            metrics["dataset"] = dataset_name
            results.append(metrics)

        df = pd.DataFrame(results)
        # Reorder columns for presentation
        cols = ["dataset", "model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "false_positive_rate", "false_negative_rate", "avg_latency_sec"]
        existing_cols = [c for c in cols if c in df.columns]
        return df[existing_cols]
