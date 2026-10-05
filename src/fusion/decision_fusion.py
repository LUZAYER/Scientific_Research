"""
Decision Fusion Layer for Multi-Signal Hallucination Detection.
Combines Confidence (C), Retrieval (R), Semantic Alignment (S), and Self-Verification (V).
Supports:
1. Linear Weighted Fusion with adaptive confidence-weighted renormalization.
2. Supervised Meta-Classifiers (Logistic Regression, Random Forest).
3. Risk Categorization (Low, Medium, High).
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.config import DetectorConfig
from src.signals.base import SignalResult

class DecisionFusion:
    """Combines complementary detection signals into a unified hallucination verdict."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        fusion_strategy: str = "linear_weighted", # "linear_weighted", "logistic_regression", "random_forest"
        low_threshold: float = DetectorConfig.LOW_RISK_THRESHOLD,
        high_threshold: float = DetectorConfig.HIGH_RISK_THRESHOLD
    ):
        self.weights = weights or dict(DetectorConfig.DEFAULT_WEIGHTS)
        self.fusion_strategy = fusion_strategy
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        # Meta-models for supervised fusion
        self.lr_model: Optional[LogisticRegression] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.is_fitted = False

    def train_meta_classifier(self, feature_matrix: np.ndarray, labels: np.ndarray, model_type: str = "logistic_regression"):
        """Trains a meta-classifier on signal features to learn optimal non-linear fusion."""
        if model_type == "logistic_regression":
            self.lr_model = LogisticRegression(class_weight="balanced", random_state=42)
            self.lr_model.fit(feature_matrix, labels)
            self.fusion_strategy = "logistic_regression"
            self.is_fitted = True
        elif model_type == "random_forest":
            self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
            self.rf_model.fit(feature_matrix, labels)
            self.fusion_strategy = "random_forest"
            self.is_fitted = True

    def _fuse_linear_weighted(self, signal_results: Dict[str, SignalResult]) -> Tuple[float, Dict[str, float]]:
        """
        Computes weighted fusion: H = sum(w_i * S_i)
        Dynamically renormalizes weights if a signal has low confidence or is absent.
        """
        active_weights = {}
        total_weight = 0.0

        for key, res in signal_results.items():
            base_w = self.weights.get(key, 0.25)
            # Modulate weight by signal confidence
            effective_w = base_w * res.confidence
            active_weights[key] = effective_w
            total_weight += effective_w

        if total_weight <= 0:
            total_weight = 1.0
            for key in signal_results:
                active_weights[key] = 1.0 / len(signal_results)

        # Normalize weights to sum to 1.0
        normalized_weights = {k: w / total_weight for k, w in active_weights.items()}

        overall_score = 0.0
        contributions = {}
        for key, res in signal_results.items():
            contrib = normalized_weights[key] * res.score
            contributions[key] = contrib
            overall_score += contrib

        return float(np.clip(overall_score, 0.0, 1.0)), contributions

    def fuse(self, signal_results: Dict[str, SignalResult]) -> Dict[str, Any]:
        """Fuses signal outputs and generates final hallucination risk assessment."""
        # Feature vector for ML models: [confidence, retrieval, semantic, self_verification]
        c_score = signal_results.get("confidence", SignalResult("confidence", 0.5, 0.5)).score
        r_score = signal_results.get("retrieval", SignalResult("retrieval", 0.5, 0.5)).score
        s_score = signal_results.get("semantic", SignalResult("semantic", 0.5, 0.5)).score
        v_score = signal_results.get("self_verification", SignalResult("self_verification", 0.5, 0.5)).score
        features = np.array([[c_score, r_score, s_score, v_score]])

        if self.fusion_strategy == "logistic_regression" and self.is_fitted and self.lr_model is not None:
            proba = self.lr_model.predict_proba(features)[0][1]
            hallucination_score = float(proba)
            contributions = {"model": "logistic_regression"}
        elif self.fusion_strategy == "random_forest" and self.is_fitted and self.rf_model is not None:
            proba = self.rf_model.predict_proba(features)[0][1]
            hallucination_score = float(proba)
            contributions = {"model": "random_forest"}
        else:
            # Default linear weighted with dynamic adaptation
            hallucination_score, contributions = self._fuse_linear_weighted(signal_results)

        # Categorize risk level
        if hallucination_score >= self.high_threshold:
            risk_level = "HIGH"
            status = "POTENTIALLY_HALLUCINATED"
            is_hallucination = True
        elif hallucination_score >= self.low_threshold:
            risk_level = "MEDIUM"
            status = "UNCERTAIN_REQUIRES_VERIFICATION"
            is_hallucination = True # Conservative binary classification
        else:
            risk_level = "LOW"
            status = "RELIABLE_FACTUAL"
            is_hallucination = False

        # Identify primary contributing signal to hallucination risk
        top_contributor = max(signal_results.items(), key=lambda item: item[1].score)

        return {
            "hallucination_score": round(hallucination_score, 4),
            "risk_level": risk_level,
            "status": status,
            "is_hallucination": is_hallucination,
            "fusion_strategy": self.fusion_strategy,
            "primary_signal_contributor": top_contributor[0],
            "signal_contributions": {k: round(v, 4) for k, v in contributions.items()} if isinstance(contributions, dict) else contributions
        }
