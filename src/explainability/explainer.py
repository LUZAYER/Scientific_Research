"""
Hallucination Explainer Module.
Generates structured, human-interpretable explanations detailing:
1. Decomposed Claim
2. Retrieved Evidence & Source
3. Semantic Entailment Verdict (Supported / Contradicted / Unsupported)
4. Multi-sample Consistency
5. Calibrated Confidence
6. Final Hallucination Score and Plain-Language Reason.
"""

from typing import List, Dict, Any

class HallucinationExplainer:
    """Produces interpretable explainability reports for evaluated responses."""

    def format_claim_report(self, evaluated_claim: Dict[str, Any]) -> str:
        """Formats a single claim's evaluation into human-readable markdown."""
        claim_data = evaluated_claim.get("claim", {})
        verdict = evaluated_claim.get("fusion_verdict", {})
        signals = evaluated_claim.get("signals", {})
        
        c_sig = signals.get("confidence", {})
        r_sig = signals.get("retrieval", {})
        s_sig = signals.get("semantic", {})
        v_sig = signals.get("self_verification", {})

        md = []
        md.append(f"### Claim #{claim_data.get('claim_id')}: \"{claim_data.get('claim_text')}\"")
        md.append(f"- **Hallucination Risk Score**: `{verdict.get('hallucination_score', 0.0):.2f}` | **Risk Level**: **{verdict.get('risk_level')}** | **Status**: {verdict.get('status')}")
        md.append(f"- **Primary Driver**: `{verdict.get('primary_signal_contributor')}`")
        
        # Evidence & Grounding
        evid_snippet = r_sig.get("details", {}).get("retrieved_evidence", "None")
        md.append(f"- **Retrieved Grounding Evidence**: *\"{evid_snippet}\"*")
        
        # Signal Breakdown
        md.append(f"- **Signals Breakdown**:")
        md.append(f"  * **Semantic Alignment**: {s_sig.get('explanation', 'N/A')} (Risk: {s_sig.get('hallucination_score', 0.0):.2f})")
        md.append(f"  * **Retrieval Grounding**: {r_sig.get('explanation', 'N/A')} (Risk: {r_sig.get('hallucination_score', 0.0):.2f})")
        md.append(f"  * **Self-Verification**: {v_sig.get('explanation', 'N/A')} (Risk: {v_sig.get('hallucination_score', 0.0):.2f})")
        md.append(f"  * **Confidence Calibration**: {c_sig.get('explanation', 'N/A')} (Risk: {c_sig.get('hallucination_score', 0.0):.2f})")
        
        return "\n".join(md)

    def generate_full_report(self, detection_result: Dict[str, Any]) -> str:
        """Generates comprehensive markdown report for the entire LLM response."""
        report = []
        report.append("# LLM Hallucination Verification Report")
        report.append(f"**Query**: {detection_result.get('query', 'N/A')}\n")
        report.append(f"## Response Overview")
        report.append(f"> {detection_result.get('response_text', '')}\n")
        
        overall = detection_result.get("overall_verdict", {})
        report.append("## Overall Assessment")
        report.append(f"- **Overall Hallucination Score**: `{overall.get('hallucination_score', 0.0):.2f}`")
        report.append(f"- **System Risk Level**: **{overall.get('risk_level', 'UNKNOWN')}**")
        report.append(f"- **Status**: `{overall.get('status', 'UNKNOWN')}`")
        report.append(f"- **Claims Flagged**: {detection_result.get('num_hallucinated_claims', 0)} / {len(detection_result.get('claims', []))}\n")

        # Localization summary
        loc = detection_result.get("localization", {})
        if loc.get("hallucinated_spans"):
            report.append("## Flagged Hallucination Spans")
            for span in loc.get("hallucinated_spans", []):
                report.append(f"- `[{span['start']}:{span['end']}]`: **\"{span['claim_text']}\"** (Score: {span['score']:.2f})")
            report.append("")

        # Claim-level breakdown
        report.append("## Detailed Claim-Level Explanations")
        for claim in detection_result.get("claims", []):
            report.append(self.format_claim_report(claim))
            report.append("")

        return "\n".join(report)
