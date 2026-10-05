"""
Master Hybrid Hallucination Detector Orchestrator.
Coordinates Claim Decomposition -> Multi-Signal Verification -> Decision Fusion -> Localization -> Explainability.
"""

from typing import List, Dict, Any, Optional
import time
from src.config import DetectorConfig
from src.claim_decomposition import ClaimDecomposer, DecomposedClaim
from src.signals.confidence_signal import ConfidenceSignal
from src.signals.retrieval_signal import RetrievalSignal
from src.signals.semantic_signal import SemanticSignal
from src.signals.self_verification_signal import SelfVerificationSignal
from src.fusion.decision_fusion import DecisionFusion
from src.localization.hallucination_localizer import HallucinationLocalizer
from src.explainability.explainer import HallucinationExplainer

class HybridHallucinationDetector:
    """
    End-to-End Hybrid Hallucination Detection Pipeline.
    Combines Confidence, Retrieval Grounding, Semantic Entailment, and Self-Verification.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        fusion_strategy: str = "linear_weighted",
        use_online_wiki: bool = True,
        disabled_signals: Optional[List[str]] = None
    ):
        self.config = DetectorConfig()
        self.decomposer = ClaimDecomposer()
        self.disabled_signals = set(disabled_signals or [])
        
        # Initialize verification signals
        self.signals = {}
        if "confidence" not in self.disabled_signals:
            self.signals["confidence"] = ConfidenceSignal()
        if "retrieval" not in self.disabled_signals:
            self.signals["retrieval"] = RetrievalSignal(use_online_wiki=use_online_wiki)
        if "semantic" not in self.disabled_signals:
            self.signals["semantic"] = SemanticSignal()
        if "self_verification" not in self.disabled_signals:
            self.signals["self_verification"] = SelfVerificationSignal()

        # Decision fusion
        self.fusion = DecisionFusion(
            weights=weights,
            fusion_strategy=fusion_strategy,
            low_threshold=self.config.LOW_RISK_THRESHOLD,
            high_threshold=self.config.HIGH_RISK_THRESHOLD
        )
        self.localizer = HallucinationLocalizer(risk_threshold=self.config.LOW_RISK_THRESHOLD)
        self.explainer = HallucinationExplainer()

    def detect(
        self,
        query: str,
        response: str,
        evidence: str = "",
        samples: Optional[List[str]] = None,
        raw_logprobs: Optional[List[float]] = None,
        confidence_input: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes complete verification pipeline for an LLM response.
        Returns comprehensive detection verdicts, localization, and explanations.
        """
        start_time = time.time()
        
        # 1. Claim Decomposition
        claims = self.decomposer.decompose(response)
        
        evaluated_claims = []
        claim_scores = []
        
        for claim in claims:
            # 2. Multi-Signal Verification
            claim_signal_results = {}
            
            # (a) Retrieval Signal first so its retrieved snippet can inform semantic alignment
            retrieved_snippet = ""
            if "retrieval" in self.signals:
                r_res = self.signals["retrieval"].evaluate(
                    claim_text=claim.text,
                    query=query,
                    evidence=evidence,
                    entities=claim.entities
                )
                claim_signal_results["retrieval"] = r_res
                retrieved_snippet = r_res.details.get("retrieved_evidence", "")

            # (b) Confidence Signal
            if "confidence" in self.signals:
                c_res = self.signals["confidence"].evaluate(
                    claim_text=claim.text,
                    raw_logprobs=raw_logprobs,
                    confidence_input=confidence_input
                )
                claim_signal_results["confidence"] = c_res

            # (c) Semantic Signal
            if "semantic" in self.signals:
                s_res = self.signals["semantic"].evaluate(
                    claim_text=claim.text,
                    query=query,
                    evidence=evidence,
                    retrieved_snippet=retrieved_snippet
                )
                claim_signal_results["semantic"] = s_res

            # (d) Self-Verification Signal
            if "self_verification" in self.signals:
                v_res = self.signals["self_verification"].evaluate(
                    claim_text=claim.text,
                    query=query,
                    samples=samples
                )
                claim_signal_results["self_verification"] = v_res

            # 3. Decision Fusion for this claim
            fusion_verdict = self.fusion.fuse(claim_signal_results)
            claim_scores.append(fusion_verdict["hallucination_score"])

            evaluated_claims.append({
                "claim": claim.to_dict(),
                "fusion_verdict": fusion_verdict,
                "signals": {k: v.to_dict() for k, v in claim_signal_results.items()}
            })

        # 4. Overall Response Aggregation
        # In factuality research, a response is hallucinated if any substantial claim is hallucinated (max aggregation),
        # smoothed with mean risk for calibrated uncertainty.
        if claim_scores:
            max_risk = max(claim_scores)
            mean_risk = sum(claim_scores) / len(claim_scores)
            overall_score = 0.7 * max_risk + 0.3 * mean_risk
        else:
            overall_score = 0.0

        if overall_score >= self.config.HIGH_RISK_THRESHOLD:
            overall_risk = "HIGH"
            overall_status = "POTENTIALLY_HALLUCINATED"
            is_hallu = True
        elif overall_score >= self.config.LOW_RISK_THRESHOLD:
            overall_risk = "MEDIUM"
            overall_status = "UNCERTAIN_REQUIRES_VERIFICATION"
            is_hallu = True
        else:
            overall_risk = "LOW"
            overall_status = "RELIABLE_FACTUAL"
            is_hallu = False

        # 5. Hallucination Localization
        localization = self.localizer.localize(response, evaluated_claims)
        
        elapsed_sec = time.time() - start_time

        result = {
            "query": query,
            "response_text": response,
            "overall_verdict": {
                "hallucination_score": round(overall_score, 4),
                "risk_level": overall_risk,
                "status": overall_status,
                "is_hallucination": is_hallu,
                "latency_seconds": round(elapsed_sec, 4)
            },
            "num_claims": len(claims),
            "num_hallucinated_claims": localization["num_claims_flagged"],
            "claims": evaluated_claims,
            "localization": localization
        }
        
        # 6. Generate Human-Readable Markdown Report
        result["explanation_markdown"] = self.explainer.generate_full_report(result)
        return result
