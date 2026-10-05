"""
Baseline Hallucination Detectors.
Implements the 4 individual baseline detectors specified in Section 16 of the research proposal:
1. Baseline 1: Confidence-Only Detector
2. Baseline 2: Retrieval-Only Detector
3. Baseline 3: Semantic / Entailment-Only Detector
4. Baseline 4: Self-Verification / Consistency-Only Detector
"""

from typing import List, Dict, Any, Optional
from src.detector import HybridHallucinationDetector

class ConfidenceOnlyBaseline:
    """Baseline 1: Uses only model confidence / uncertainty analysis."""
    def __init__(self):
        self.detector = HybridHallucinationDetector(
            weights={"confidence": 1.0, "retrieval": 0.0, "semantic": 0.0, "self_verification": 0.0},
            disabled_signals=["retrieval", "semantic", "self_verification"]
        )

    def detect(self, query: str, response: str, **kwargs) -> Dict[str, Any]:
        return self.detector.detect(query, response, **kwargs)


class RetrievalOnlyBaseline:
    """Baseline 2: Uses only external evidence retrieval and grounding coverage."""
    def __init__(self, use_online_wiki: bool = True):
        self.detector = HybridHallucinationDetector(
            weights={"confidence": 0.0, "retrieval": 1.0, "semantic": 0.0, "self_verification": 0.0},
            use_online_wiki=use_online_wiki,
            disabled_signals=["confidence", "semantic", "self_verification"]
        )

    def detect(self, query: str, response: str, **kwargs) -> Dict[str, Any]:
        return self.detector.detect(query, response, **kwargs)


class SemanticOnlyBaseline:
    """Baseline 3: Uses only semantic alignment and textual entailment."""
    def __init__(self):
        self.detector = HybridHallucinationDetector(
            weights={"confidence": 0.0, "retrieval": 0.0, "semantic": 1.0, "self_verification": 0.0},
            disabled_signals=["confidence", "retrieval", "self_verification"]
        )

    def detect(self, query: str, response: str, **kwargs) -> Dict[str, Any]:
        return self.detector.detect(query, response, **kwargs)


class SelfVerificationOnlyBaseline:
    """Baseline 4: Uses only self-verification and sampling consistency (FactSelfCheck paradigm)."""
    def __init__(self):
        self.detector = HybridHallucinationDetector(
            weights={"confidence": 0.0, "retrieval": 0.0, "semantic": 0.0, "self_verification": 1.0},
            disabled_signals=["confidence", "retrieval", "semantic"]
        )

    def detect(self, query: str, response: str, **kwargs) -> Dict[str, Any]:
        return self.detector.detect(query, response, **kwargs)
