"""
================================================================================
  HARMONIQ: Hybrid Adaptive Reliability & Multi-signal Orchestration
            Network for Interpretable Quality verification
================================================================================
Core framework implementation based on the End-to-End System Architecture:
1. User Query + LLM Response
2. Claim & Fact Decomposition (atomic claims, spans, entity extraction)
3. Four Verification Signals:
   - C: Confidence Analysis & Calibrated Uncertainty
   - R: Retrieval Grounding & Entity Coverage
   - S: Semantic Entailment & Contradiction Detection
   - V: Self-Verification & Sampling Consistency
4. Decision Fusion Layer (Adaptive Weighted / Meta-Classifiers)
5. Claim-Level Hallucination Localization
6. Final Assessment & Interpretability Report
================================================================================
"""

import time
from typing import List, Dict, Any, Optional

from src.config import DetectorConfig
from src.claim_decomposition import ClaimDecomposer, DecomposedClaim
from src.signals.confidence_signal import ConfidenceSignal
from src.signals.retrieval_signal import RetrievalSignal
from src.signals.semantic_signal import SemanticSignal
from src.signals.self_verification_signal import SelfVerificationSignal
from src.fusion.decision_fusion import DecisionFusion
from src.localization.hallucination_localizer import HallucinationLocalizer
from src.explainability.explainer import HallucinationExplainer

class HarmoniqAssessment:
    """Encapsulates the structured final assessment from the HARMONIQ framework."""
    def __init__(
        self,
        query: str,
        response_text: str,
        hallucination_score: float,
        risk_level: str,
        status: str,
        is_hallucination: bool,
        latency_sec: float,
        claims: List[Dict[str, Any]],
        localization: Dict[str, Any],
        explanation_markdown: str
    ):
        self.query = query
        self.response_text = response_text
        self.hallucination_score = hallucination_score
        self.risk_level = risk_level
        self.status = status
        self.is_hallucination = is_hallucination
        self.latency_sec = latency_sec
        self.claims = claims
        self.localization = localization
        self.explanation_markdown = explanation_markdown

    def display(self):
        """Displays a formatted visual assessment to the terminal or notebook."""
        border = "=" * 78
        sub_border = "-" * 78

        print(f"\n{border}")
        print("                 HARMONIQ FRAMEWORK: FINAL ASSESSMENT                 ")
        print(f"{border}")
        print(f"User Query      : {self.query}")
        print(f"Generated Text  : {self.response_text}")
        print(f"{sub_border}")
        
        # Color coding for terminal output
        risk_badge = f"[{self.risk_level} RISK]"
        status_badge = f"{self.status}"
        
        print(f"Final Hallucination Score : {self.hallucination_score:.4f}  (Scale 0.00 = Factual, 1.00 = Hallucinated)")
        print(f"Risk Stratification       : {risk_badge}")
        print(f"Verification Status       : {status_badge}")
        print(f"Total Atomic Claims       : {len(self.claims)}")
        print(f"Claims Flagged            : {self.localization.get('num_claims_flagged', 0)}")
        print(f"Processing Latency        : {self.latency_sec:.4f} seconds")
        print(f"{sub_border}")

        print("\n--- CLAIM-LEVEL VERIFICATION BREAKDOWN ---")
        for idx, c_info in enumerate(self.claims, 1):
            claim_text = c_info["claim"]["claim_text"]
            score = c_info["fusion_verdict"]["hallucination_score"]
            risk = c_info["fusion_verdict"]["risk_level"]
            primary_driver = c_info["fusion_verdict"]["primary_signal_contributor"]
            signals = c_info["signals"]
            
            s_verdict = signals.get("semantic", {}).get("details", {}).get("verdict", "N/A")
            r_evid = signals.get("retrieval", {}).get("details", {}).get("retrieved_evidence", "None")[:80]
            if len(signals.get("retrieval", {}).get("details", {}).get("retrieved_evidence", "")) > 80:
                r_evid += "..."

            print(f"\n[Claim #{idx}] \"{claim_text}\"")
            print(f"  • Hallucination Risk : {score:.2f} ({risk})")
            print(f"  • Semantic Alignment : {s_verdict}")
            print(f"  • Retrieved Evidence : \"{r_evid}\"")
            print(f"  • Primary Driver     : {primary_driver}")

        print(f"\n{sub_border}")
        print("--- LOCALIZED HIGHLIGHTED OUTPUT ---")
        print(self.localization.get("highlighted_markdown", self.response_text))
        print(f"{border}\n")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response_text": self.response_text,
            "hallucination_score": self.hallucination_score,
            "risk_level": self.risk_level,
            "status": self.status,
            "is_hallucination": self.is_hallucination,
            "latency_seconds": self.latency_sec,
            "num_claims": len(self.claims),
            "num_flagged_claims": self.localization.get("num_claims_flagged", 0),
            "claims": self.claims,
            "localization": self.localization
        }


class HARMONIQ:
    """
    HARMONIQ: Hybrid Adaptive Reliability & Multi-signal Orchestration
              Network for Interpretable Quality verification.

    End-to-End System Pipeline:
    Query + Response -> Claim Decomposition -> 4 Signals (C, R, S, V) -> Fusion -> Assessment.
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
        
        # 1. Initialize Verification Signals
        self.signals = {}
        if "confidence" not in self.disabled_signals:
            self.signals["confidence"] = ConfidenceSignal()
        if "retrieval" not in self.disabled_signals:
            self.signals["retrieval"] = RetrievalSignal(use_online_wiki=use_online_wiki)
        if "semantic" not in self.disabled_signals:
            self.signals["semantic"] = SemanticSignal()
        if "self_verification" not in self.disabled_signals:
            self.signals["self_verification"] = SelfVerificationSignal()

        # 2. Decision Fusion Layer
        self.fusion = DecisionFusion(
            weights=weights,
            fusion_strategy=fusion_strategy,
            low_threshold=self.config.LOW_RISK_THRESHOLD,
            high_threshold=self.config.HIGH_RISK_THRESHOLD
        )

        # 3. Localization & Explainability
        self.localizer = HallucinationLocalizer(risk_threshold=self.config.LOW_RISK_THRESHOLD)
        self.explainer = HallucinationExplainer()

    def verify(
        self,
        query: str,
        response: str,
        evidence: str = "",
        samples: Optional[List[str]] = None,
        raw_logprobs: Optional[List[float]] = None,
        confidence_input: Optional[float] = None
    ) -> HarmoniqAssessment:
        """
        Executes the end-to-end HARMONIQ verification workflow.
        Returns a rich HarmoniqAssessment object.
        """
        start_time = time.time()

        # Step 1: Claim & Atomic Fact Decomposition
        decomposed_claims = self.decomposer.decompose(response)
        evaluated_claims = []
        claim_scores = []

        # Step 2: Multi-Signal Verification for each atomic claim
        for claim in decomposed_claims:
            claim_signals = {}
            retrieved_snippet = ""

            # Signal 2: Retrieval Grounding (R)
            if "retrieval" in self.signals:
                r_res = self.signals["retrieval"].evaluate(
                    claim_text=claim.text,
                    query=query,
                    evidence=evidence,
                    entities=claim.entities
                )
                claim_signals["retrieval"] = r_res
                retrieved_snippet = r_res.details.get("retrieved_evidence", "")

            # Signal 1: Confidence Analysis (C)
            if "confidence" in self.signals:
                c_res = self.signals["confidence"].evaluate(
                    claim_text=claim.text,
                    raw_logprobs=raw_logprobs,
                    confidence_input=confidence_input
                )
                claim_signals["confidence"] = c_res

            # Signal 3: Semantic Entailment & Contradiction (S)
            if "semantic" in self.signals:
                s_res = self.signals["semantic"].evaluate(
                    claim_text=claim.text,
                    query=query,
                    evidence=evidence,
                    retrieved_snippet=retrieved_snippet
                )
                claim_signals["semantic"] = s_res

            # Signal 4: Self-Verification Consistency (V)
            if "self_verification" in self.signals:
                v_res = self.signals["self_verification"].evaluate(
                    claim_text=claim.text,
                    query=query,
                    samples=samples
                )
                claim_signals["self_verification"] = v_res

            # Step 3: Decision Fusion for this claim
            fusion_out = self.fusion.fuse(claim_signals)
            claim_scores.append(fusion_out["hallucination_score"])

            evaluated_claims.append({
                "claim": claim.to_dict(),
                "fusion_verdict": fusion_out,
                "signals": {k: v.to_dict() for k, v in claim_signals.items()}
            })

        # Step 4: Aggregate overall response score
        if claim_scores:
            max_score = max(claim_scores)
            mean_score = sum(claim_scores) / len(claim_scores)
            overall_score = 0.7 * max_score + 0.3 * mean_score
        else:
            overall_score = 0.0

        # Risk Stratification
        if overall_score >= self.config.HIGH_RISK_THRESHOLD:
            risk_level = "HIGH"
            status = "POTENTIALLY_HALLUCINATED (WARNING)"
            is_hallu = True
        elif overall_score >= self.config.LOW_RISK_THRESHOLD:
            risk_level = "MEDIUM"
            status = "UNCERTAIN (REQUIRES_HUMAN_REVIEW)"
            is_hallu = True
        else:
            risk_level = "LOW"
            status = "RELIABLE_FACTUAL (VERIFIED)"
            is_hallu = False

        # Step 5: Claim-Level Localization
        localization = self.localizer.localize(response, evaluated_claims)
        elapsed_sec = time.time() - start_time

        # Step 6: Generate Full Assessment
        raw_result_dict = {
            "query": query,
            "response_text": response,
            "overall_verdict": {
                "hallucination_score": round(overall_score, 4),
                "risk_level": risk_level,
                "status": status,
                "is_hallucination": is_hallu,
                "latency_seconds": round(elapsed_sec, 4)
            },
            "num_claims": len(decomposed_claims),
            "num_hallucinated_claims": localization["num_claims_flagged"],
            "claims": evaluated_claims,
            "localization": localization
        }
        explanation_md = self.explainer.generate_full_report(raw_result_dict)

        return HarmoniqAssessment(
            query=query,
            response_text=response,
            hallucination_score=round(overall_score, 4),
            risk_level=risk_level,
            status=status,
            is_hallucination=is_hallu,
            latency_sec=round(elapsed_sec, 4),
            claims=evaluated_claims,
            localization=localization,
            explanation_markdown=explanation_md
        )

# Alias for ease of import
HarmoniqFramework = HARMONIQ
