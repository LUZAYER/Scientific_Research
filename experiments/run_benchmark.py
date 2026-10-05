"""
Experiment 1: Benchmark Evaluation.
Compares Proposed Hybrid Framework against Baselines 1-4 across benchmark datasets:
- HaluEval QA
- TruthfulQA
- WikiBio-GPT3
- FavaMultiSamples (FactSelfCheck)

Outputs publication-ready tables answering Research Questions RQ1, RQ2, and RQ4.
"""

import os
import sys
import argparse
import pandas as pd

# Add repo root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import BenchmarkDataLoader
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.config import OUTPUT_DIR

def parse_args():
    parser = argparse.ArgumentParser(description="Run Benchmark Evaluation")
    parser.add_argument("--samples-per-dataset", type=int, default=50, help="Number of records to evaluate per dataset (default: 50 for quick test)")
    parser.add_argument("--use-online-wiki", action="store_true", help="Enable online Wikipedia API queries")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=================================================================")
    print("   RUNNING BENCHMARK EVALUATION: HYBRID DETECTOR VS BASELINES   ")
    print("=================================================================")
    print(f"Sample limit per dataset: {args.samples_per_dataset}")
    print(f"Online Wikipedia Search: {args.use_online_wiki}")
    print("-----------------------------------------------------------------")

    runner = BenchmarkRunner(use_online_wiki=args.use_online_wiki)
    all_results = []

    # 1. HaluEval QA
    try:
        print("\nLoading HaluEval QA...")
        halu_records = BenchmarkDataLoader.load_halueval_qa(limit=args.samples_per_dataset)
        df_halu = runner.run_benchmark("HaluEval-QA", halu_records)
        all_results.append(df_halu)
        print("\n--- HaluEval-QA Results ---")
        print(df_halu.to_string(index=False))
    except Exception as e:
        print(f"Skipping HaluEval QA: {e}")

    # 2. TruthfulQA
    try:
        print("\nLoading TruthfulQA...")
        tqa_records = BenchmarkDataLoader.load_truthfulqa(limit=args.samples_per_dataset)
        df_tqa = runner.run_benchmark("TruthfulQA", tqa_records)
        all_results.append(df_tqa)
        print("\n--- TruthfulQA Results ---")
        print(df_tqa.to_string(index=False))
    except Exception as e:
        print(f"Skipping TruthfulQA: {e}")

    # 3. WikiBio-GPT3
    try:
        print("\nLoading WikiBio-GPT3...")
        wb_records = BenchmarkDataLoader.load_wikibio_gpt3(limit=args.samples_per_dataset)
        df_wb = runner.run_benchmark("WikiBio-GPT3", wb_records)
        all_results.append(df_wb)
        print("\n--- WikiBio-GPT3 Results ---")
        print(df_wb.to_string(index=False))
    except Exception as e:
        print(f"Skipping WikiBio-GPT3: {e}")

    # 4. FavaMultiSamples
    try:
        print("\nLoading FavaMultiSamples...")
        fava_records = BenchmarkDataLoader.load_favamultisamples(limit=args.samples_per_dataset)
        df_fava = runner.run_benchmark("FavaMultiSamples", fava_records)
        all_results.append(df_fava)
        print("\n--- FavaMultiSamples Results ---")
        print(df_fava.to_string(index=False))
    except Exception as e:
        print(f"Skipping FavaMultiSamples: {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        csv_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
        md_path = os.path.join(OUTPUT_DIR, "benchmark_results.md")

        final_df.to_csv(csv_path, index=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Benchmark Comparison Results\n\n")
            try:
                f.write(final_df.to_markdown(index=False))
            except Exception:
                f.write(final_df.to_string(index=False))

        print(f"\n=================================================================")
        print(f"Benchmark results saved successfully:")
        print(f"  CSV: {csv_path}")
        print(f"  Markdown: {md_path}")
        print("=================================================================")

if __name__ == "__main__":
    main()
