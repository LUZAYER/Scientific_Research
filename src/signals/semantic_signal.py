"""
Signal 3: Semantic Alignment and Textual Entailment.
Evaluates the semantic relationship between the generated claim and evidence.
Classifies the relationship into SUPPORTED, CONTRADICTED, or UNSUPPORTED,
and computes a semantic alignment hallucination score.
"""

import re
from typing import Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.signals.base import BaseSignal, SignalResult

class SemanticSignal(BaseSignal):
    """Evaluates semantic consistency and textual entailment between claim and evidence."""

    def __init__(self):
        super().__init__(name="semantic_signal")
        self.negation_words = {"not", "never", "no", "neither", "nor", "none", "without", "hardly", "barely"}

    def _check_polarity_contradiction(self, claim_text: str, evidence_text: str) -> bool:
        """Detects explicit negation and polarity mismatch between claim and evidence."""
        claim_tokens = set(re.findall(r'\b\w+\b', claim_text.lower()))
        evid_tokens = set(re.findall(r'\b\w+\b', evidence_text.lower()))
        
        claim_has_neg = bool(claim_tokens.intersection(self.negation_words))
        evid_has_neg = bool(evid_tokens.intersection(self.negation_words))
        
        # If one has explicit negation while the other doesn't, but they share core content
        content_overlap = len(claim_tokens.intersection(evid_tokens) - self.negation_words)
        if (claim_has_neg != evid_has_neg) and content_overlap >= 3:
            return True
            
        return False

    def _detect_numeric_conflict(self, claim_text: str, evidence_text: str) -> Optional[Tuple[str, str]]:
        """Detects date or numerical conflict between claim and evidence (e.g. 1921 vs 1935)."""
        claim_nums = re.findall(r'\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?\b', claim_text)
        evid_nums = re.findall(r'\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?\b', evidence_text)
        
        if claim_nums and evid_nums:
            for cn in claim_nums:
                if cn not in evid_nums:
                    # Check if there is another number in evidence in similar context
                    for en in evid_nums:
                        if len(cn) == len(en) and cn != en:
                            return (cn, en)
        return None

    def _compute_semantic_similarity(self, claim_text: str, evidence_text: str) -> float:
        """Computes n-gram semantic similarity using TF-IDF."""
        if not evidence_text or not evidence_text.strip():
            return 0.0
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
            tfidf = vectorizer.fit_transform([claim_text, evidence_text])
            sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
            return sim
        except Exception:
            return 0.0

    def evaluate(
        self,
        claim_text: str,
        query: str = "",
        evidence: str = "",
        retrieved_snippet: str = "",
        **kwargs
    ) -> SignalResult:
        """Determines whether evidence semantically supports, contradicts, or fails to establish the claim."""
        eval_evidence = retrieved_snippet if retrieved_snippet else evidence

        if not eval_evidence or not eval_evidence.strip():
            return SignalResult(
                signal_name=self.name,
                score=0.75, # High risk due to absence of grounding evidence
                confidence=0.60,
                details={"verdict": "UNSUPPORTED", "semantic_similarity": 0.0},
                explanation="No evidence available to semantically verify this claim."
            )

        sim_score = self._compute_semantic_similarity(claim_text, eval_evidence)
        polarity_conflict = self._check_polarity_contradiction(claim_text, eval_evidence)
        num_conflict = self._detect_numeric_conflict(claim_text, eval_evidence)

        # Classification Logic
        if polarity_conflict:
            verdict = "CONTRADICTED"
            hallucination_risk = 0.95
            explanation = "Evidence directly contradicts the claim via polarity / negation mismatch."
        elif num_conflict is not None:
            verdict = "CONTRADICTED"
            hallucination_risk = 0.90
            explanation = f"Numerical/date contradiction: claim states '{num_conflict[0]}' but evidence mentions '{num_conflict[1]}'."
        elif sim_score >= 0.50:
            verdict = "SUPPORTED"
            hallucination_risk = max(0.05, 0.40 - sim_score * 0.5)
            explanation = f"Claim is semantically supported by evidence (alignment score: {sim_score:.2f})."
        elif sim_score >= 0.25:
            verdict = "PARTIALLY_SUPPORTED"
            hallucination_risk = 0.45
            explanation = f"Partial semantic alignment with evidence ({sim_score:.2f}); some nuances ungrounded."
        else:
            verdict = "UNSUPPORTED"
            hallucination_risk = 0.75
            explanation = f"Low semantic alignment ({sim_score:.2f}); evidence does not sufficiently establish claim."

        details = {
            "verdict": verdict,
            "semantic_similarity": round(sim_score, 4),
            "polarity_conflict": polarity_conflict,
            "numeric_conflict": num_conflict
        }

        return SignalResult(
            signal_name=self.name,
            score=hallucination_risk,
            confidence=0.85,
            details=details,
            explanation=explanation
        )
