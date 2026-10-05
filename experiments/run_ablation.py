"""
Experiment 2: Systematic Ablation Study.
Implements Section 25 of the research proposal:
Compares:
  - Full Hybrid (Confidence + Retrieval + Semantic + Self-Verification)
  against:
  - Hybrid – Confidence
  - Hybrid – Retrieval
  - Hybrid – Semantic
  - Hybrid – Self-Verification

Answers Research Question RQ3: "Which verification signal contributes most to performance?"
"""

import os
import sys
import argparse
import time
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import BenchmarkDataLoader
from src.detector import HybridHallucinationDetector
from src.evaluation.metrics import EvaluationMetrics
from src.config import OUTPUT_DIR

def parse_args():
    parser = argparse.ArgumentParser(description="Run Ablation Study")
    parser.add_argument("--samples", type=int, default=100, help="Number of records to test (default: 100)")
    parser.add_argument("--dataset", type=str, default="halueval", choices=["halueval", "truthfulqa", "wikibio", "favamultisamples"])
    return parser.parse_args()

def main():
    args = parse_args()
    print("=================================================================")
    print("                  SYSTEMATIC ABLATION STUDY                      ")
    print("=================================================================")
    print(f"Dataset: {args.dataset} | Sample count: {args.samples}")
    print("-----------------------------------------------------------------")

    # Load dataset
    if args.dataset == "halueval":
        records = BenchmarkDataLoader.load_halueval_qa(limit=args.samples)
    elif args.dataset == "truthfulqa":
        records = BenchmarkDataLoader.load_truthfulqa(limit=args.samples)
    elif args.dataset == "wikibio":
        records = BenchmarkDataLoader.load_wikibio_gpt3(limit=args.samples)
    else:
        records = BenchmarkDataLoader.load_favamultisamples(limit=args.samples)

    # Define Ablation Configurations
    configurations = {
        "Full Hybrid (C + R + S + V)": HybridHallucinationDetector(disabled_signals=[]),
        "Hybrid – Confidence": HybridHallucinationDetector(disabled_signals=["confidence"]),
        "Hybrid – Retrieval": HybridHallucinationDetector(disabled_signals=["retrieval"]),
        "Hybrid – Semantic": HybridHallucinationDetector(disabled_signals=["semantic"]),
        "Hybrid – Self-Verification": HybridHallucinationDetector(disabled_signals=["self_verification"]),
    }

    results = []

    for name, detector in configurations.items():
        y_true, y_pred, y_scores, latencies = [], [], [], []

        for rec in tqdm(records, desc=f"Ablation: {name}", leave=False):
            t0 = time.time()
            res = detector.detect(
                query=rec.query,
                response=rec.response,
                evidence=rec.grounding_evidence,
                samples=rec.stochastic_samples
            )
            elapsed = time.time() - t0
            
            ov = res["overall_verdict"]
            y_true.append(rec.label)
            y_pred.append(1 if ov["is_hallucination"] else 0)
            y_scores.append(ov["hallucination_score"])
            latencies.append(elapsed)

        metrics = EvaluationMetrics.compute_all(y_true, y_pred, y_scores, latencies)
        metrics["configuration"] = name
        metrics["dataset"] = args.dataset
        results.append(metrics)

    df_ablation = pd.DataFrame(results)
    cols = ["configuration", "f1_score", "precision", "recall", "accuracy", "roc_auc", "avg_latency_sec"]
    df_clean = df_ablation[[c for c in cols if c in df_ablation.columns]]

    print("\n--- Ablation Study Results Table ---")
    print(df_clean.to_string(index=False))

    # Save to file
    csv_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    md_path = os.path.join(OUTPUT_DIR, "ablation_results.md")
    df_clean.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Systematic Ablation Study Results\n\n")
        try:
            f.write(df_clean.to_markdown(index=False))
        except Exception:
            f.write(df_clean.to_string(index=False))

    print(f"\nAblation results saved to:\n  {csv_path}\n  {md_path}")

if __name__ == "__main__":
    main()
