"""
Base class and data contracts for verification signals.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class SignalResult:
    """Encapsulates the output of a single hallucination verification signal."""
    def __init__(
        self,
        signal_name: str,
        score: float, # Normalized hallucination risk score in [0.0, 1.0] (0 = factual, 1 = hallucinated)
        confidence: float, # Confidence in this signal's verdict [0.0, 1.0]
        details: Optional[Dict[str, Any]] = None,
        explanation: str = ""
    ):
        self.signal_name = signal_name
        self.score = max(0.0, min(1.0, float(score)))
        self.confidence = max(0.0, min(1.0, float(confidence)))
        self.details = details or {}
        self.explanation = explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "hallucination_score": round(self.score, 4),
            "signal_confidence": round(self.confidence, 4),
            "details": self.details,
            "explanation": self.explanation
        }


class BaseSignal(ABC):
    """Abstract base class for all detection signals."""
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(
        self,
        claim_text: str,
        query: str = "",
        evidence: str = "",
        samples: Optional[list] = None,
        raw_logprobs: Optional[list] = None,
        **kwargs
    ) -> SignalResult:
        """Evaluates a claim and returns a SignalResult."""
        pass
