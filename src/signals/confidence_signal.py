"""
Signal 1: Confidence Analysis and Calibrated Uncertainty.
Implements confidence scoring accounting for the finding (Trust Me, I'm Wrong, 2025)
that LLMs can hallucinate with extreme confidence. Evaluates token logprob entropy,
perplexity, and linguistic certainty markers.
"""

import math
import re
from typing import List, Optional, Dict, Any
from src.signals.base import BaseSignal, SignalResult

class ConfidenceSignal(BaseSignal):
    """Analyzes confidence and uncertainty signals for a generated claim."""

    def __init__(self):
        super().__init__(name="confidence_signal")
        
        # Linguistic markers of over-confidence (often compensating for hallucination)
        self.overconfidence_markers = [
            "definitely", "certainly", "undoubtedly", "without a doubt",
            "proven fact", "absolutely", "always", "guaranteed", "indisputable"
        ]
        # Linguistic markers of epistemic hesitation / hedging
        self.hedging_markers = [
            "perhaps", "maybe", "possibly", "might", "could be",
            "it is believed", "reportedly", "supposedly", "approximate"
        ]

    def _analyze_linguistic_confidence(self, text: str) -> Dict[str, Any]:
        """Analyzes text surface markers for verbalized confidence calibration."""
        text_lower = text.lower()
        
        overconf_hits = [m for m in self.overconfidence_markers if re.search(r'\b' + re.escape(m) + r'\b', text_lower)]
        hedge_hits = [m for m in self.hedging_markers if re.search(r'\b' + re.escape(m) + r'\b', text_lower)]
        
        # Overconfidence without external verification often correlates with fabrication
        linguistic_risk = 0.5 # Neutral baseline
        if len(overconf_hits) > 0:
            linguistic_risk += min(0.3, len(overconf_hits) * 0.15)
        if len(hedge_hits) > 0:
            linguistic_risk -= min(0.2, len(hedge_hits) * 0.1)
            
        return {
            "overconfidence_markers": overconf_hits,
            "hedging_markers": hedge_hits,
            "linguistic_risk": linguistic_risk
        }

    def _analyze_logprobs(self, raw_logprobs: List[float]) -> Dict[str, float]:
        """Computes perplexity, average negative log likelihood, and entropy from token logprobs."""
        if not raw_logprobs:
            return {}
            
        avg_nll = -sum(raw_logprobs) / len(raw_logprobs)
        perplexity = math.exp(min(avg_nll, 20.0)) # Clip to avoid overflow
        min_prob = math.exp(min(raw_logprobs))
        
        # Calculate approximate entropy
        probs = [math.exp(lp) for lp in raw_logprobs]
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs) / len(probs)
        
        # Uncertainty-based hallucination risk
        # High perplexity / high entropy indicates the model was uncertain during generation
        uncert_risk = 1.0 - math.exp(-avg_nll / 2.0)
        
        return {
            "avg_nll": avg_nll,
            "perplexity": perplexity,
            "min_token_prob": min_prob,
            "entropy": entropy,
            "uncertainty_risk": uncert_risk
        }

    def evaluate(
        self,
        claim_text: str,
        query: str = "",
        evidence: str = "",
        samples: Optional[list] = None,
        raw_logprobs: Optional[list] = None,
        confidence_input: Optional[float] = None,
        **kwargs
    ) -> SignalResult:
        """Evaluates model confidence and returns a calibrated hallucination risk score."""
        ling_info = self._analyze_linguistic_confidence(claim_text)
        
        if raw_logprobs and len(raw_logprobs) > 0:
            logprob_info = self._analyze_logprobs(raw_logprobs)
            # Combine token uncertainty with linguistic confidence
            risk_score = 0.7 * logprob_info["uncertainty_risk"] + 0.3 * ling_info["linguistic_risk"]
            confidence = 0.9 # High confidence in signal when logprobs are present
            details = {**ling_info, **logprob_info}
            explanation = f"Evaluated via token perplexity ({logprob_info['perplexity']:.2f}) and sequence entropy."
        elif confidence_input is not None:
            # External confidence score passed in (e.g. from an API or model)
            # Invert so that higher model confidence maps to lower baseline risk, but modulated by hedges
            raw_risk = 1.0 - max(0.0, min(1.0, float(confidence_input)))
            risk_score = 0.8 * raw_risk + 0.2 * ling_info["linguistic_risk"]
            confidence = 0.75
            details = {**ling_info, "external_confidence": confidence_input}
            explanation = f"Evaluated from model confidence score ({confidence_input:.2f}) with linguistic calibration."
        else:
            # Black-box fallback: linguistic calibration
            risk_score = ling_info["linguistic_risk"]
            confidence = 0.5 # Moderate confidence in surface-only signal
            details = ling_info
            explanation = "Evaluated via epistemic hedge and over-confidence linguistic calibration."

        return SignalResult(
            signal_name=self.name,
            score=risk_score,
            confidence=confidence,
            details=details,
            explanation=explanation
        )
