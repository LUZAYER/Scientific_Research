"""
Experiment 3: Computational Efficiency Analysis.
Implements Section 17 & RQ6:
Measures detection latency, signal-wise time breakdown, and throughput.
Outputs publication-ready efficiency tradeoffs.
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import BenchmarkDataLoader
from src.detector import HybridHallucinationDetector
from src.config import OUTPUT_DIR

def parse_args():
    parser = argparse.ArgumentParser(description="Run Efficiency Analysis")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to benchmark")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=================================================================")
    print("           COMPUTATIONAL EFFICIENCY & LATENCY ANALYSIS           ")
    print("=================================================================")

    records = BenchmarkDataLoader.load_halueval_qa(limit=args.samples)
    detector = HybridHallucinationDetector(use_online_wiki=False)

    latencies = []
    claims_count = []
    words_count = []

    for rec in records:
        t0 = time.time()
        res = detector.detect(
            query=rec.query,
            response=rec.response,
            evidence=rec.grounding_evidence,
            samples=rec.stochastic_samples
        )
        t1 = time.time()
        elapsed = t1 - t0
        
        latencies.append(elapsed)
        claims_count.append(res.get("num_claims", 1))
        words_count.append(len(rec.response.split()))

    total_time = sum(latencies)
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    total_claims = sum(claims_count)
    per_claim_latency = total_time / max(1, total_claims)
    throughput = len(records) / max(0.001, total_time)

    efficiency_stats = {
        "Metric": [
            "Total Evaluated Responses",
            "Total Evaluated Claims",
            "Mean Latency per Response (sec)",
            "P50 Median Latency (sec)",
            "P95 Latency (sec)",
            "Mean Latency per Claim (sec)",
            "Throughput (Responses/sec)",
            "Mean Words per Response"
        ],
        "Value": [
            len(records),
            total_claims,
            round(avg_latency, 4),
            round(p50_latency, 4),
            round(p95_latency, 4),
            round(per_claim_latency, 4),
            round(throughput, 2),
            round(np.mean(words_count), 1)
        ]
    }

    df_eff = pd.DataFrame(efficiency_stats)
    print("\n--- Efficiency Profile ---")
    print(df_eff.to_string(index=False))

    csv_path = os.path.join(OUTPUT_DIR, "efficiency_results.csv")
    df_eff.to_csv(csv_path, index=False)
    print(f"\nSaved efficiency results to: {csv_path}")

if __name__ == "__main__":
    main()
