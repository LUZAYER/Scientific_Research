"""
Signal 4: Self-Verification and Sampling Consistency.
Implements black-box consistency analysis inspired by FactSelfCheck (2026 EACL)
and SelfCheckGPT. Evaluates the stability of factual claims across multiple
stochastic LLM generations.
"""

from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.signals.base import BaseSignal, SignalResult

class SelfVerificationSignal(BaseSignal):
    """Evaluates cross-sample consistency and self-verification stability."""

    def __init__(self, sample_agreement_threshold: float = 0.35):
        super().__init__(name="self_verification_signal")
        self.sample_agreement_threshold = sample_agreement_threshold

    def _evaluate_sample_agreement(self, claim_text: str, samples: List[str]) -> Dict[str, Any]:
        """Calculates how consistently the claim is affirmed across independent stochastic samples."""
        if not samples:
            return {"consistency_ratio": 0.5, "supporting_samples": 0, "total_samples": 0}

        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
            corpus = [claim_text] + samples
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Compare claim against each sample
            sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            supporting = [i for i, s in enumerate(sims) if s >= self.sample_agreement_threshold]
            weak_support = [i for i, s in enumerate(sims) if 0.15 <= s < self.sample_agreement_threshold]
            
            # Weighted consistency ratio
            consistency_ratio = (len(supporting) + 0.5 * len(weak_support)) / len(samples)
            
            return {
                "consistency_ratio": min(1.0, consistency_ratio),
                "supporting_samples": len(supporting),
                "weak_supporting_samples": len(weak_support),
                "total_samples": len(samples),
                "sample_similarities": [round(float(s), 3) for s in sims[:5]]
            }
        except Exception:
            return {"consistency_ratio": 0.5, "supporting_samples": 0, "total_samples": len(samples)}

    def evaluate(
        self,
        claim_text: str,
        query: str = "",
        evidence: str = "",
        samples: Optional[List[str]] = None,
        **kwargs
    ) -> SignalResult:
        """Computes self-verification hallucination risk based on cross-sample stability."""
        samples = samples or []
        
        if not samples or len(samples) == 0:
            # When no multi-samples are provided, fallback to self-contained consistency prior
            return SignalResult(
                signal_name=self.name,
                score=0.50, # Neutral uninformative score
                confidence=0.30,
                details={"status": "NO_STOCHASTIC_SAMPLES_AVAILABLE"},
                explanation="No stochastic samples provided for cross-generation consistency check."
            )

        info = self._evaluate_sample_agreement(claim_text, samples)
        consistency_ratio = info["consistency_ratio"]
        total_s = info["total_samples"]
        supp_s = info["supporting_samples"]

        # If claim is supported in most samples -> low hallucination risk
        # If claim is contradicted or absent across samples -> high hallucination risk
        hallucination_risk = 1.0 - consistency_ratio

        if consistency_ratio >= 0.70:
            explanation = f"High cross-sample stability: verified in {supp_s}/{total_s} independent generations."
        elif consistency_ratio >= 0.40:
            explanation = f"Moderate cross-sample stability: consistent in {supp_s}/{total_s} generations."
        else:
            explanation = f"Inconsistent across generations: factual claim absent or unstable ({supp_s}/{total_s} samples)."

        return SignalResult(
            signal_name=self.name,
            score=hallucination_risk,
            confidence=0.85 if total_s >= 5 else 0.65,
            details=info,
            explanation=explanation
        )
