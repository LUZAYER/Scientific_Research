"""
Interactive / Single-Response Demonstration of the Hybrid Detector.
Runs claim extraction, multi-signal verification, decision fusion,
localization, and explainability on any user query and response.
"""

import os
import sys
import argparse
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import HybridHallucinationDetector

def parse_args():
    parser = argparse.ArgumentParser(description="Demo Inference for Hybrid Hallucination Detector")
    parser.add_argument("--query", type=str, default="Where was Albert Einstein born and what prize did he win?", help="User query")
    parser.add_argument(
        "--response",
        type=str,
        default="Albert Einstein was born in Munich, Germany, and he won the Nobel Prize in Literature in 1921.",
        help="Generated LLM response to evaluate"
    )
    parser.add_argument(
        "--evidence",
        type=str,
        default="Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire. He was awarded the 1921 Nobel Prize in Physics.",
        help="Grounding evidence or reference passage"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print("=================================================================")
    print("        HYBRID HALLUCINATION DETECTOR: SINGLE RUN DEMO           ")
    print("=================================================================")
    print(f"User Query:\n  {args.query}\n")
    print(f"Generated LLM Response:\n  {args.response}\n")
    print(f"Grounding Evidence:\n  {args.evidence}\n")
    print("Running verification pipeline...\n")

    detector = HybridHallucinationDetector(use_online_wiki=True)
    result = detector.detect(
        query=args.query,
        response=args.response,
        evidence=args.evidence
    )

    print("=================================================================")
    print("                     EVALUATION REPORT                           ")
    print("=================================================================")
    print(result["explanation_markdown"])
    print("\n=================================================================")
    print("                  LOCALIZED HIGHLIGHTED SPANS                    ")
    print("=================================================================")
    print(result["localization"]["highlighted_markdown"])
    print("=================================================================")

if __name__ == "__main__":
    main()
