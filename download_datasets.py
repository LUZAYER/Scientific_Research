"""
Dataset Downloader for LLM Hallucination Detection Framework
Downloads and verifies benchmarks:
- HaluEval (QA & Dialogue)
- TruthfulQA (CSV)
- REFIND (Sample & Multi-lingual benchmark)
- FactSelfCheck (FavaMultiSamples parquet)
"""

import os
import sys
import json
import requests
import pandas as pd
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

DATASET_URLS = {
    "truthfulqa": {
        "dir": os.path.join(DATA_DIR, "truthfulqa"),
        "files": {
            "TruthfulQA.csv": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
        }
    },
    "halueval": {
        "dir": os.path.join(DATA_DIR, "halueval"),
        "files": {
            "qa_data.json": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
            "dialogue_data.json": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/dialogue_data.json"
        }
    },
    "refind": {
        "dir": os.path.join(DATA_DIR, "refind"),
        "files": {
            "sample_set.v1.json": "https://raw.githubusercontent.com/oneonlee/REFIND/main/data/sample/sample_set.v1.json",
            "sample_structured.json": "https://raw.githubusercontent.com/oneonlee/REFIND/main/data/sample/sample_set.v1-structured.json"
        }
    },
    "favamultisamples": {
        "dir": os.path.join(DATA_DIR, "favamultisamples"),
        "files": {
            "fava_multisamples.parquet": "https://huggingface.co/datasets/graphml-lab-pwr/FavaMultiSamples/resolve/main/data/test-00000-of-00001.parquet"
        }
    }
}

def download_file(url: str, output_path: str):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        print(f"File already exists: {output_path} ({os.path.getsize(output_path):,} bytes)")
        return
    print(f"Downloading {url} -> {output_path}...")
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, "wb") as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024 * 64):
            size = f.write(data)
            bar.update(size)

def verify_datasets():
    print("\n--- Verifying Downloaded Datasets ---")
    
    # 1. TruthfulQA
    tqa_path = os.path.join(DATA_DIR, "truthfulqa", "TruthfulQA.csv")
    if os.path.exists(tqa_path):
        df_tqa = pd.read_csv(tqa_path)
        print(f"✓ TruthfulQA: Loaded {len(df_tqa)} questions. Columns: {list(df_tqa.columns[:5])}")
    
    # 2. HaluEval QA
    he_qa_path = os.path.join(DATA_DIR, "halueval", "qa_data.json")
    if os.path.exists(he_qa_path):
        count = 0
        with open(he_qa_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        print(f"✓ HaluEval QA: Loaded {count:,} paired QA entries.")

    # 3. HaluEval Dialogue
    he_dia_path = os.path.join(DATA_DIR, "halueval", "dialogue_data.json")
    if os.path.exists(he_dia_path):
        count = 0
        with open(he_dia_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        print(f"✓ HaluEval Dialogue: Loaded {count:,} dialogue entries.")

    # 4. REFIND
    refind_path = os.path.join(DATA_DIR, "refind", "sample_set.v1.json")
    if os.path.exists(refind_path):
        count = 0
        with open(refind_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        print(f"✓ REFIND Sample Set: Loaded {count:,} hallucination span instances.")

    # 5. FavaMultiSamples (FactSelfCheck)
    fava_path = os.path.join(DATA_DIR, "favamultisamples", "fava_multisamples.parquet")
    if os.path.exists(fava_path):
        df_fava = pd.read_parquet(fava_path)
        print(f"✓ FavaMultiSamples (FactSelfCheck): Loaded {len(df_fava):,} sampled texts with KG/consistency targets.")
        print(f"  Columns: {list(df_fava.columns)}")

def main():
    print("==================================================")
    print(" Downloading Research Datasets for Hybrid Detector")
    print("==================================================")
    for group, info in DATASET_URLS.items():
        os.makedirs(info["dir"], exist_ok=True)
        for fname, url in info["files"].items():
            dest = os.path.join(info["dir"], fname)
            try:
                download_file(url, dest)
            except Exception as e:
                print(f"Error downloading {url}: {e}")

    verify_datasets()
    print("\nDataset download and verification completed successfully!")

if __name__ == "__main__":
    main()
