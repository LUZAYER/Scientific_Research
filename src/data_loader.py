"""
Unified Dataset Loader for Hallucination Benchmarks:
- HaluEval (QA & Dialogue)
- TruthfulQA
- FavaMultiSamples (FactSelfCheck 2026)
- WikiBioGPT3 (SelfCheckGPT benchmark)
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from src.config import RAW_DATA_DIR

class UnifiedRecord:
    """Standardized evaluation record for hallucination detection."""
    def __init__(
        self,
        record_id: str,
        query: str,
        response: str,
        label: int, # 0 = Factual / Reliable, 1 = Hallucinated
        dataset_name: str,
        grounding_evidence: Optional[str] = None,
        stochastic_samples: Optional[List[str]] = None,
        claim_annotations: Optional[List[Dict[str, Any]]] = None,
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.record_id = record_id
        self.query = query
        self.response = response
        self.label = label
        self.dataset_name = dataset_name
        self.grounding_evidence = grounding_evidence or ""
        self.stochastic_samples = stochastic_samples or []
        self.claim_annotations = claim_annotations or []
        self.confidence_score = confidence_score
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "query": self.query,
            "response": self.response,
            "label": self.label,
            "dataset_name": self.dataset_name,
            "grounding_evidence": self.grounding_evidence,
            "num_samples": len(self.stochastic_samples),
            "claim_annotations": self.claim_annotations
        }


class BenchmarkDataLoader:
    """Loads and standardizes hallucination benchmark datasets."""
    
    @staticmethod
    def load_halueval_qa(limit: Optional[int] = None) -> List[UnifiedRecord]:
        """Loads HaluEval QA dataset with positive (factual) and negative (hallucinated) pairs."""
        path = os.path.join(RAW_DATA_DIR, "halueval", "qa_data.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"HaluEval QA file not found at {path}. Run download_datasets.py first.")
            
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if limit and len(records) >= limit * 2:
                    break
                if not line.strip():
                    continue
                item = json.loads(line)
                q = item.get("question", "")
                k = item.get("knowledge", "")
                right_ans = item.get("right_answer", "")
                hallu_ans = item.get("hallucinated_answer", "")
                
                # Factual instance (label = 0)
                if right_ans:
                    records.append(UnifiedRecord(
                        record_id=f"halueval_qa_{idx}_factual",
                        query=q,
                        response=right_ans,
                        label=0,
                        dataset_name="HaluEval-QA",
                        grounding_evidence=k,
                        metadata={"type": "factual"}
                    ))
                # Hallucinated instance (label = 1)
                if hallu_ans:
                    records.append(UnifiedRecord(
                        record_id=f"halueval_qa_{idx}_hallu",
                        query=q,
                        response=hallu_ans,
                        label=1,
                        dataset_name="HaluEval-QA",
                        grounding_evidence=k,
                        metadata={"type": "hallucination"}
                    ))
                    
        return records[:limit] if limit else records

    @staticmethod
    def load_truthfulqa(limit: Optional[int] = None) -> List[UnifiedRecord]:
        """Loads TruthfulQA dataset."""
        path = os.path.join(RAW_DATA_DIR, "truthfulqa", "TruthfulQA.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"TruthfulQA file not found at {path}.")
            
        df = pd.read_csv(path)
        records = []
        for idx, row in df.iterrows():
            if limit and len(records) >= limit * 2:
                break
            q = str(row.get("Question", ""))
            best_ans = str(row.get("Best Answer", ""))
            incorrect_ans = str(row.get("Best Incorrect Answer", ""))
            src = str(row.get("Source", ""))
            cat = str(row.get("Category", ""))
            
            # Factual instance
            if best_ans and best_ans != "nan":
                records.append(UnifiedRecord(
                    record_id=f"truthfulqa_{idx}_factual",
                    query=q,
                    response=best_ans,
                    label=0,
                    dataset_name="TruthfulQA",
                    grounding_evidence=src,
                    metadata={"category": cat}
                ))
            # Hallucinated instance
            if incorrect_ans and incorrect_ans != "nan":
                records.append(UnifiedRecord(
                    record_id=f"truthfulqa_{idx}_hallu",
                    query=q,
                    response=incorrect_ans,
                    label=1,
                    dataset_name="TruthfulQA",
                    grounding_evidence=src,
                    metadata={"category": cat}
                ))
        return records[:limit] if limit else records

    @staticmethod
    def load_favamultisamples(limit: Optional[int] = None) -> List[UnifiedRecord]:
        """Loads FavaMultiSamples (from FactSelfCheck 2026 EACL) with multi-sample stochastic generations."""
        path = os.path.join(RAW_DATA_DIR, "favamultisamples", "fava_multisamples.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"FavaMultiSamples file not found at {path}.")
            
        df = pd.read_parquet(path)
        records = []
        for idx, row in df.iterrows():
            if limit and len(records) >= limit:
                break
            prompt = str(row.get("prompt", ""))
            output = str(row.get("output", ""))
            sentences = row.get("sentences", [])
            bin_annots = row.get("sentences_binary_annotations", [])
            
            # Text samples generated by LLM for sampling-based verification
            text_samples = row.get("text_samples", [])
            if hasattr(text_samples, "tolist"):
                text_samples = text_samples.tolist()
            elif not isinstance(text_samples, list):
                text_samples = []
                
            # If any sentence in output is hallucinated, overall label = 1
            has_hallu = 1 if any(b == 1 for b in bin_annots) else 0
            
            claim_annots = []
            if len(sentences) == len(bin_annots):
                for s, b in zip(sentences, bin_annots):
                    claim_annots.append({"claim": s, "is_hallucination": int(b)})
                    
            records.append(UnifiedRecord(
                record_id=f"favamultisamples_{idx}",
                query=prompt,
                response=output,
                label=has_hallu,
                dataset_name="FavaMultiSamples",
                stochastic_samples=text_samples,
                claim_annotations=claim_annots,
                metadata={"dataset_origin": row.get("dataset", "FAVA")}
            ))
        return records[:limit] if limit else records

    @staticmethod
    def load_wikibio_gpt3(limit: Optional[int] = None) -> List[UnifiedRecord]:
        """Loads WikiBio GPT-3 hallucination benchmark (SelfCheckGPT benchmark)."""
        path = os.path.join(RAW_DATA_DIR, "selfcheckgpt", "wiki_bio_gpt3_hallucination.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"WikiBio file not found at {path}.")
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        records = []
        for idx, item in enumerate(data):
            if limit and len(records) >= limit:
                break
            gpt3_text = item.get("gpt3_text", "")
            wiki_text = item.get("wiki_bio_text", "")
            sentences = item.get("gpt3_sentences", [])
            annotations = item.get("annotation", [])
            samples = item.get("gpt3_text_samples", [])
            
            # Sentence-level annotations: accurate, minor_inaccurate, major_inaccurate
            has_hallu = 1 if any("inaccurate" in str(a) for a in annotations) else 0
            
            claim_annots = []
            for s, a in zip(sentences, annotations):
                claim_annots.append({
                    "claim": s,
                    "is_hallucination": 1 if "inaccurate" in str(a) else 0,
                    "raw_annotation": str(a)
                })
                
            records.append(UnifiedRecord(
                record_id=f"wikibio_{idx}",
                query=f"Biography for article #{item.get('wiki_bio_test_idx', idx)}",
                response=gpt3_text,
                label=has_hallu,
                dataset_name="WikiBio-GPT3",
                grounding_evidence=wiki_text,
                stochastic_samples=samples,
                claim_annotations=claim_annots,
                metadata={"test_idx": item.get("wiki_bio_test_idx")}
            ))
        return records[:limit] if limit else records
