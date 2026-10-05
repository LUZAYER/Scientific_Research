"""
Hallucination Localization Module.
Pinpoints specific sentences, spans, and claims within an LLM response
that triggered hallucination warnings, generating highlighted visual outputs.
"""

from typing import List, Dict, Any

class HallucinationLocalizer:
    """Localizes and highlights hallucinated spans in generated text."""

    def __init__(self, risk_threshold: float = 0.50):
        self.risk_threshold = risk_threshold

    def localize(self, original_text: str, evaluated_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Maps hallucinated claim verdicts back to original text character offsets.
        Generates annotated Markdown and HTML with highlighted spans.
        """
        hallucinated_spans = []
        for claim_info in evaluated_claims:
            score = claim_info.get("fusion_verdict", {}).get("hallucination_score", 0.0)
            if score >= self.risk_threshold:
                span = claim_info.get("claim", {}).get("span", (0, 0))
                hallucinated_spans.append({
                    "claim_id": claim_info.get("claim", {}).get("claim_id"),
                    "claim_text": claim_info.get("claim", {}).get("claim_text"),
                    "start": span[0],
                    "end": span[1],
                    "score": score,
                    "risk_level": claim_info.get("fusion_verdict", {}).get("risk_level")
                })

        # Sort spans by starting index
        hallucinated_spans.sort(key=lambda s: s["start"])

        # Construct highlighted text
        highlighted_html = []
        highlighted_md = []
        last_idx = 0

        for s in hallucinated_spans:
            start, end = s["start"], s["end"]
            if start >= last_idx and start < len(original_text):
                # Unflagged prefix
                highlighted_html.append(original_text[last_idx:start])
                highlighted_md.append(original_text[last_idx:start])
                
                # Flagged segment
                segment = original_text[start:min(end, len(original_text))]
                color = "#ffcccc" if s["score"] >= 0.65 else "#fff0b3"
                highlighted_html.append(f'<span style="background-color: {color}; border-bottom: 2px solid red;" title="Risk: {s["score"]:.2f}">{segment}</span>')
                highlighted_md.append(f"**[HALLUCINATION RISK: {segment}]**")
                last_idx = end

        if last_idx < len(original_text):
            highlighted_html.append(original_text[last_idx:])
            highlighted_md.append(original_text[last_idx:])

        return {
            "num_claims_flagged": len(hallucinated_spans),
            "hallucinated_spans": hallucinated_spans,
            "highlighted_html": "".join(highlighted_html),
            "highlighted_markdown": "".join(highlighted_md)
        }
