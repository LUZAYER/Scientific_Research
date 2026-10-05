"""
Google Colab Master Execution Script.
One-click script to run the entire research evaluation pipeline in Google Colab:
1. Environment setup & dependency validation
2. Automatic benchmark dataset acquisition
3. Interactive single-inference demonstration
4. Full Benchmark evaluation (Baselines 1-4 vs Proposed Hybrid Framework)
5. Systematic Ablation Study (Full Hybrid vs Signal dropouts)
6. Computational Efficiency Analysis
"""

import os
import sys
import subprocess

def main():
    print("====================================================================")
    print("   HYBRID HALLUCINATION DETECTION FRAMEWORK — GOOGLE COLAB RUNNER   ")
    print("====================================================================")

    # Step 1: Download datasets if needed
    print("\n[Step 1/5] Verifying and Downloading Research Datasets...")
    try:
        from download_datasets import main as run_download
        run_download()
    except Exception as e:
        print(f"Error during dataset verification: {e}")

    # Step 2: Interactive Demonstration
    print("\n[Step 2/5] Running Interactive Verification Demonstration...")
    try:
        from experiments.demo_inference import main as run_demo
        run_demo()
    except Exception as e:
        print(f"Error during demo inference: {e}")

    # Step 3: Comparative Benchmark
    print("\n[Step 3/5] Running Comparative Benchmark (Baselines vs Hybrid)...")
    try:
        from experiments.run_benchmark import main as run_benchmark
        run_benchmark()
    except Exception as e:
        print(f"Error during benchmark: {e}")

    # Step 4: Systematic Ablation Study
    print("\n[Step 4/5] Running Systematic Ablation Study...")
    try:
        from experiments.run_ablation import main as run_ablation
        run_ablation()
    except Exception as e:
        print(f"Error during ablation: {e}")

    # Step 5: Efficiency Analysis
    print("\n[Step 5/5] Running Computational Efficiency Analysis...")
    try:
        from experiments.run_efficiency_analysis import main as run_eff
        run_eff()
    except Exception as e:
        print(f"Error during efficiency analysis: {e}")

    print("\n====================================================================")
    print("               RESEARCH PIPELINE EXECUTION COMPLETE!                ")
    print(" All results and publication tables have been saved to ./results/")
    print("====================================================================")

if __name__ == "__main__":
    main()
