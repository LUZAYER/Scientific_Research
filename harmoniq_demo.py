"""
================================================================================
  HARMONIQ: Framework Demonstration & Scientific Benchmark Comparison
================================================================================
Demonstrates:
1. End-to-End System Architecture on a generated response.
2. Claim-level multi-signal verification (C, R, S, V).
3. The Final Assessment output (Risk Level, Status, Localized Spans).
4. Academic Comparison Table comparing HARMONIQ with other frameworks.
================================================================================
"""

import sys
import os

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.harmoniq import HARMONIQ

def print_comparison_table():
    """Prints the comprehensive scientific comparison table."""
    table = """
==================================================================================================================================================
                                    COMPREHENSIVE SCIENTIFIC COMPARISON: HARMONIQ vs EXISTING FRAMEWORKS
==================================================================================================================================================
Framework       | Primary Venue / Year | Detection Granularity | Signals Exploited                | Decision Mechanism      | Explainability & Localization
----------------+----------------------+-----------------------+----------------------------------+-------------------------+-----------------------------
SelfCheckGPT    | EMNLP 2023           | Sentence-level        | Multi-sample Consistency only    | Heuristic / Avg score   | Sentence-level score only
HaluEval        | EMNLP 2023           | Passage / QA-level    | Static reference QA matching     | LLM-as-a-judge / direct | Binary (Hallucinated Yes/No)
REFIND          | arXiv 2025           | Sub-sentence span     | Retrieval Sensitivity (CSR)      | Sensitivity ratio cutoff| Span-level highlighting
FactBench/VERIFY| EMNLP 2024–2025      | Sentence / Fact-level | Web Search retrieval             | 3-way classifier (S/U/U)| Evidence citation (No multi-signal)
PROBE           | Findings of ACL 2026 | Process-stage level   | Process decomposition (4 steps)  | Pipeline judge LLM      | Process-step attribution
FactSelfCheck   | Findings of EACL 2026| Fact / Triple-level   | KG triples + Sampling consistency| Jaccard / KG match      | Fact-triple graph inconsistency
----------------+----------------------+-----------------------+----------------------------------+-------------------------+-----------------------------
HARMONIQ (Ours) | Proposed 2026        | Claim & Atomic Fact   | 4 Complementary Signals:         | Adaptive Multi-Signal   | Complete Claim-by-Claim
                |                      | (with character span) | 1. Calibrated Confidence (C)     | Decision Fusion         | Attribution + Localized Spans
                |                      |                       | 2. Evidence Retrieval (R)        | (Linear Weighted /      | + Risk Stratification
                |                      |                       | 3. Semantic Entailment (S)       | Logistic Regression /   | (LOW / MEDIUM / HIGH)
                |                      |                       | 4. Self-Verification (V)         | Random Forest)          |
==================================================================================================================================================
"""
    print(table)


def run_demo():
    print("====================================================================")
    print("            INITIALIZING HARMONIQ VERIFICATION PIPELINE             ")
    print("====================================================================")

    # Initialize the HARMONIQ framework
    harmoniq = HARMONIQ(use_online_wiki=False)

    # Example query and generated response with mixed factual and hallucinated assertions
    query = "Where was Albert Einstein born and which Nobel Prize did he receive?"
    
    # Generated response containing 1 factual claim and 1 hallucinated claim
    generated_response = (
        "Albert Einstein was born in Ulm, Germany, and he won the Nobel Prize in Literature in 1921."
    )
    
    # External Grounding Evidence (from Wikipedia / RAG retrieval)
    grounding_evidence = (
        "Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879. "
        "In 1922, he was awarded the 1921 Nobel Prize in Physics for his services to Theoretical Physics, "
        "and especially for his discovery of the law of the photoelectric effect."
    )

    # Stochastic samples from independent LLM generations (FactSelfCheck / SelfCheckGPT paradigm)
    stochastic_samples = [
        "Einstein was born in Ulm, Germany and won the Nobel Prize in Physics.",
        "Albert Einstein, born in Ulm, received the Nobel Prize in Physics in 1921 for the photoelectric effect.",
        "Albert Einstein was a German physicist born in Ulm who was awarded the Nobel Prize in Physics."
    ]

    print("\nRunning HARMONIQ end-to-end multi-signal verification...\n")
    assessment = harmoniq.verify(
        query=query,
        response=generated_response,
        evidence=grounding_evidence,
        samples=stochastic_samples
    )

    # Display the Final Assessment
    assessment.display()

    # Print the Academic Comparison Table
    print_comparison_table()

if __name__ == "__main__":
    run_demo()
