"""
Configuration settings for Hybrid Hallucination Detection Framework.
Defines hyper-parameters, signal weights, threshold cutoffs, and dataset paths.
"""

import os

# Project Root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class DetectorConfig:
    """Hyperparameters and signal weights for the Hybrid Detector."""
    
    # Fusion Formulation: H = w_c * C + w_r * R + w_s * S + w_v * V
    # Weights must sum to 1.0
    DEFAULT_WEIGHTS = {
        "confidence": 0.15,      # C: Calibrated uncertainty signal
        "retrieval": 0.35,       # R: Retrieval grounding signal (evidence availability/coverage)
        "semantic": 0.30,        # S: Semantic entailment / alignment signal
        "self_verification": 0.20 # V: Self-verification / multi-sample consistency signal
    }
    
    # Risk Thresholds for Hallucination Score H in [0, 1]
    # H < LOW_THRESHOLD: Low Risk (Factual / Reliable)
    # LOW_THRESHOLD <= H < HIGH_THRESHOLD: Medium Risk (Uncertain / Requires Review)
    # H >= HIGH_THRESHOLD: High Risk (Hallucination Warning)
    LOW_RISK_THRESHOLD = 0.35
    HIGH_RISK_THRESHOLD = 0.65
    
    # NLI Entailment Thresholds
    ENTAILMENT_THRESHOLD = 0.65
    CONTRADICTION_THRESHOLD = 0.50
    
    # Retrieval Config
    RETRIEVAL_TOP_K = 3
    MIN_EVIDENCE_LEN = 20
    USE_ONLINE_WIKIPEDIA = True  # Can fall back to offline/provided context
    
    # Sampling / Consistency Config (FactSelfCheck / SelfCheckGPT)
    SAMPLE_CONSISTENCY_N = 5     # Number of stochastic samples to inspect if provided
    SIMILARITY_METRIC = "tfidf_cosine" # 'tfidf_cosine' or 'dense_embeddings'
    
    # Random Seed for Reproducibility
    RANDOM_SEED = 42
