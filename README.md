# Hallucination Detection in Large Language Models: A Hybrid Framework for Reliable AI Responses

An academic research framework proposing a **multi-signal, claim-level hybrid verification architecture** to detect hallucinations in Large Language Models (LLMs).

---

## 1. Research Overview

Large Language Models (ChatGPT, Claude, Gemini, Llama, Qwen) frequently generate fluent and confident responses that contain fabricated, unsupported, or contradictory claims (**LLM hallucinations**). Because high confidence does not guarantee factual correctness, single-signal detectors (e.g., confidence-only or surface-similarity) often fail.

This project introduces a **Hybrid Hallucination Detection Framework** that:
1. Decomposes LLM responses into verifiable **atomic claims/facts**.
2. Evaluates each claim through **four complementary verification signals**:
   - **$C$ — Confidence Analysis & Calibrated Uncertainty**: Perplexity, entropy, and epistemic hedge calibration (*Trust Me, I'm Wrong*, 2025).
   - **$R$ — Retrieval-Based Verification**: External knowledge grounding via document retrieval and entity coverage (*REFIND*, 2025; *FactBench*, 2024–2025).
   - **$S$ — Semantic Consistency & Entailment**: NLI classification into `SUPPORTED`, `UNSUPPORTED`, or `CONTRADICTED`.
   - **$V$ — Self-Verification & Sampling Consistency**: Cross-generation stability across stochastic LLM samples (*FactSelfCheck*, Findings of EACL 2026; *SelfCheckGPT*).
3. Synthesizes signals through a **Decision-Fusion Layer** ($H = w_1 C + w_2 R + w_3 S + w_4 V$ and trained meta-classifiers).
4. Produces **claim-level localization** (highlighted spans) and **interpretable explainability reports**.

---

## 2. End-to-End System Architecture

```text
                  User Query + Generated LLM Response
                                  │
                                  ▼
                    Claim & Fact Decomposition
           (Sentence splitting, atomic clauses, entity extraction)
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
         ▼                                                 ▼
   Claim #1: "..."                                   Claim #2: "..."
         │                                                 │
 ┌───────┴───────────────────────────────┐                 │
 │       Four Verification Signals       │                 │
 │                                       │                 │
 │ 1. Confidence Analysis (C)            │                 │
 │ 2. Retrieval Grounding (R)            │                 │
 │ 3. Semantic Entailment (S)            │                 │
 │ 4. Self-Verification Consistency (V)  │                 │
 └───────┬───────────────────────────────┘                 │
         │                                                 │
         ▼                                                 ▼
   Decision Fusion Layer ◄─────────────────────────────────┘
   (Adaptive weighted fusion / Logistic Regression / Random Forest)
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │                   Final Assessment                     │
 │  - Hallucination Score (0.0 to 1.0)                    │
 │  - Risk Level: LOW / MEDIUM / HIGH                     │
 │  - Status: RELIABLE vs POTENTIALLY HALLUCINATED        │
 │  - Claim-Level Localization (Highlighted Spans)        │
 │  - Explainability Report (Claim, Evidence, Verdict)    │
 └────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```text
.
├── README.md                      # Complete research documentation & Colab guide
├── requirements.txt               # Lightweight dependencies (Colab & local)
├── download_datasets.py           # Automated downloader for benchmark datasets
├── colab_runner.py                # One-click runner for Google Colab
├── data/
│   └── raw/                       # Downloaded benchmark datasets
│       ├── halueval/              # HaluEval QA & Dialogue datasets
│       ├── truthfulqa/            # TruthfulQA benchmark CSV
│       ├── favamultisamples/      # FavaMultiSamples (FactSelfCheck 2026)
│       ├── selfcheckgpt/          # WikiBio GPT-3 hallucination benchmark
│       └── refind/                # REFIND sample dataset & annotations
├── src/
│   ├── config.py                  # Global hyperparameters & default weights
│   ├── data_loader.py             # Unified loader for all 4 benchmark datasets
│   ├── claim_decomposition.py     # Sentence & atomic claim extraction module
│   ├── signals/                   # The 4 verification signals
│   │   ├── base.py                # Abstract base class & SignalResult
│   │   ├── confidence_signal.py   # Signal 1: Calibrated uncertainty
│   │   ├── retrieval_signal.py    # Signal 2: Evidence grounding & retrieval
│   │   ├── semantic_signal.py     # Signal 3: NLI alignment & contradiction
│   │   └── self_verification_signal.py # Signal 4: Cross-sample consistency
│   ├── fusion/
│   │   └── decision_fusion.py     # Multi-signal decision fusion layer
│   ├── localization/
│   │   └── hallucination_localizer.py # Span localization & highlighting
│   ├── explainability/
│   │   └── explainer.py           # Interpretable explanation generator
│   ├── detector.py                # Master HybridHallucinationDetector orchestrator
│   ├── baselines.py               # Baselines 1-4 (Confidence, Retrieval, Semantic, SelfCheck)
│   └── evaluation/
│       ├── metrics.py             # Accuracy, Precision, Recall, F1, ROC-AUC, Latency
│       └── benchmark_runner.py    # Batch evaluation runner
├── experiments/
│   ├── demo_inference.py          # Interactive single-response demonstration
│   ├── run_benchmark.py           # Benchmark comparison (Baselines vs Hybrid)
│   ├── run_ablation.py            # Systematic ablation study (Signal dropouts)
│   └── run_efficiency_analysis.py # Latency and throughput analysis
└── results/                       # Generated CSV & Markdown publication tables
```

---

## 4. Benchmark Datasets Included

The framework incorporates real-world datasets from top 2024–2026 hallucination research:

1. **HaluEval** (*Li et al.*): 10,000 QA and Dialogue samples with paired factual reference texts, ground truth answers, and hallucinated counter-factuals.
2. **TruthfulQA** (*Lin et al.*): 790 questions designed to elicit human false beliefs and misconceptions.
3. **FavaMultiSamples** (*Sawczyn et al., Findings of EACL 2026*): 460 passages from the *FactSelfCheck* project with 20 stochastic samples generated per prompt at temperature 1.0 for consistency analysis.
4. **WikiBio-GPT3** (*SelfCheckGPT*, *Manakul et al.*): 238 biographies generated by GPT-3 with sentence-level human annotations (`accurate`, `minor_inaccurate`, `major_inaccurate`) and multi-sample generations.
5. **REFIND** (*Lee et al., 2025*): Retrieval-augmented factuality benchmark with span-level hallucination labels.

---

## 5. How to Run in Google Colab (Step-by-Step)

### Option A: One-Click Runner (Fastest)

1. Upload this project folder to Google Colab or clone your GitHub repository:
   ```bash
   git clone <YOUR_REPO_URL>
   cd <REPO_NAME>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the master Colab runner:
   ```bash
   python colab_runner.py
   ```
   This will automatically execute the dataset verification, single-query demo, full benchmark evaluation, ablation study, and efficiency analysis, saving all publication tables to `results/`.

---

### Option B: Interactive Notebook Cells

You can run individual experiments directly in notebook cells:

#### 1. Download Datasets
```python
from download_datasets import main as download_all
download_all()
```

#### 2. Test a Single LLM Response
```python
from src.detector import HybridHallucinationDetector

detector = HybridHallucinationDetector(use_online_wiki=True)

query = "Where was Albert Einstein born and what prize did he win?"
response = "Albert Einstein was born in Munich, Germany, and won the Nobel Prize in Literature in 1921."
evidence = "Albert Einstein was born in Ulm, Germany. He won the 1921 Nobel Prize in Physics."

result = detector.detect(query=query, response=response, evidence=evidence)

print(result["explanation_markdown"])
```

#### 3. Run Benchmark Comparison (Baselines 1-4 vs Hybrid)
```python
from src.data_loader import BenchmarkDataLoader
from src.evaluation.benchmark_runner import BenchmarkRunner

runner = BenchmarkRunner(use_online_wiki=False)
records = BenchmarkDataLoader.load_halueval_qa(limit=50)

df_results = runner.run_benchmark("HaluEval-QA", records)
print(df_results.to_markdown(index=False))
```

#### 4. Run Ablation Study (Section 25 of Proposal)
```python
!python experiments/run_ablation.py --dataset halueval --samples 100
```

#### 5. Run Efficiency Analysis
```python
!python experiments/run_efficiency_analysis.py --samples 50
```

---

## 6. How to Run Locally

If running on your local machine:
```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Download datasets
python download_datasets.py

# 3. Run demo
python experiments/demo_inference.py

# 4. Run benchmark evaluation
python experiments/run_benchmark.py --samples-per-dataset 50

# 5. Run ablation study
python experiments/run_ablation.py --dataset truthfulqa --samples 100
```

---

## 7. Mapping to Research Questions (RQs)

| Research Question | Code Implementation | Output File |
| :--- | :--- | :--- |
| **RQ1**: How accurately can individual signals detect hallucinations? | `src/baselines.py` (Baselines 1–4) | `results/benchmark_results.csv` |
| **RQ2**: Does multi-signal fusion outperform individual baselines? | `src/detector.py` vs `src/baselines.py` | `results/benchmark_results.csv` |
| **RQ3**: Which verification signal contributes most? | `experiments/run_ablation.py` (Dropouts) | `results/ablation_results.csv` |
| **RQ4**: Does the hybrid framework reduce FPR & FNR? | `src/evaluation/metrics.py` (Confusion matrix) | `results/benchmark_results.csv` |
| **RQ5**: Can the framework provide claim-level explanations? | `src/localization/` & `src/explainability/` | Console & markdown reports |
| **RQ6**: What is the performance vs latency trade-off? | `experiments/run_efficiency_analysis.py` | `results/efficiency_results.csv` |

---

## 8. Sample Explainability Output

```markdown
### Claim #1: "Albert Einstein was born in Munich, Germany"
- **Hallucination Risk Score**: `0.88` | **Risk Level**: **HIGH** | **Status**: POTENTIALLY_HALLUCINATED
- **Primary Driver**: `semantic_signal`
- **Retrieved Grounding Evidence**: *"Albert Einstein was born in Ulm, Germany..."*
- **Signals Breakdown**:
  * **Semantic Alignment**: Evidence directly contradicts claim (Ulm vs Munich). (Risk: 0.92)
  * **Retrieval Grounding**: Evidence retrieved from Wikipedia/reference context. (Risk: 0.20)
  * **Self-Verification**: Inconsistent across stochastic samples. (Risk: 0.70)
  * **Confidence Calibration**: Expressed without epistemic hedges. (Risk: 0.50)
```
