"""
Signal 2: Retrieval-Based Evidence Verification.
Grounds claims against external knowledge sources (provided reference documents,
RAG context passages, or live Wikipedia search API as inspired by REFIND and FactBench).
Computes entity overlap and evidence coverage.
"""

import re
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.signals.base import BaseSignal, SignalResult

class RetrievalSignal(BaseSignal):
    """Retrieves relevant evidence and verifies claim grounding."""

    def __init__(self, use_online_wiki: bool = True):
        super().__init__(name="retrieval_signal")
        self.use_online_wiki = use_online_wiki
        self.wiki_session = requests.Session()
        self.wiki_session.headers.update({
            "User-Agent": "HybridHallucinationDetector/1.0 (academic research)"
        })

    def _retrieve_from_local_context(self, claim_text: str, context_text: str, top_k: int = 2) -> Tuple[str, float]:
        """Finds most relevant sentences from provided evidence context using TF-IDF."""
        if not context_text or not context_text.strip():
            return "", 0.0

        # Split context into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context_text) if len(s.strip()) > 10]
        if not sentences:
            return context_text[:300], 0.2

        try:
            corpus = [claim_text] + sentences
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            top_indices = similarities.argsort()[::-1][:top_k]
            best_sim = float(similarities[top_indices[0]]) if len(top_indices) > 0 else 0.0
            
            best_passages = [sentences[idx] for idx in top_indices if similarities[idx] > 0.05]
            snippet = " ".join(best_passages) if best_passages else sentences[0]
            return snippet, best_sim
        except Exception:
            # Fallback simple substring / overlap
            return context_text[:250], 0.2

    def _query_wikipedia_summary(self, query: str) -> Optional[str]:
        """Fetches page summary from Wikipedia API for given entity or query."""
        if not self.use_online_wiki or not query or len(query.strip()) < 3:
            return None
            
        try:
            # First search for most relevant page title
            search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(query)}&limit=1&namespace=0&format=json"
            res = self.wiki_session.get(search_url, timeout=3.0)
            if res.status_code == 200:
                data = res.json()
                if len(data) >= 2 and len(data[1]) > 0:
                    page_title = data[1][0]
                    # Fetch summary of that page
                    page_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(page_title)}"
                    summary_res = self.wiki_session.get(page_url, timeout=3.0)
                    if summary_res.status_code == 200:
                        return summary_res.json().get("extract", "")
        except Exception:
            return None
        return None

    def evaluate(
        self,
        claim_text: str,
        query: str = "",
        evidence: str = "",
        entities: Optional[List[str]] = None,
        **kwargs
    ) -> SignalResult:
        """Evaluates whether retrieved evidence grounds the claim."""
        entities = entities or []
        retrieved_evidence = ""
        relevance_score = 0.0

        # Step 1: Check provided ground-truth or RAG evidence
        if evidence and len(evidence.strip()) > 10:
            retrieved_evidence, relevance_score = self._retrieve_from_local_context(claim_text, evidence)
        
        # Step 2: If no evidence or weak relevance, optionally query Wikipedia for key entities
        if (not retrieved_evidence or relevance_score < 0.15) and entities and self.use_online_wiki:
            primary_entity = entities[0]
            wiki_text = self._query_wikipedia_summary(primary_entity)
            if wiki_text:
                retrieved_evidence, relevance_score = self._retrieve_from_local_context(claim_text, wiki_text)

        # Step 3: Entity Grounding Verification
        # Check if the extracted entities are present in retrieved evidence
        evidence_lower = retrieved_evidence.lower() if retrieved_evidence else ""
        entities_found = []
        entities_missing = []

        for ent in entities:
            if ent.lower() in evidence_lower:
                entities_found.append(ent)
            else:
                entities_missing.append(ent)

        total_entities = len(entities)
        entity_coverage = (len(entities_found) / total_entities) if total_entities > 0 else 0.5

        # Step 4: Compute Retrieval Grounding Risk
        # High relevance & high entity coverage -> low risk (0.0)
        # No evidence or missing entities -> high risk (1.0)
        if not retrieved_evidence or relevance_score < 0.05:
            # No evidence could be retrieved to support this claim
            grounding_score = 0.15
            explanation = "No authoritative evidence retrieved to ground this claim."
        else:
            grounding_score = 0.6 * relevance_score + 0.4 * entity_coverage
            if entity_coverage >= 0.8 and relevance_score > 0.3:
                explanation = f"Supported by retrieved evidence with {len(entities_found)} entities grounded."
            else:
                missing_str = ", ".join(entities_missing[:3])
                explanation = f"Partial evidence found, but missing key entities: [{missing_str}]."

        hallucination_risk = 1.0 - max(0.0, min(1.0, grounding_score))

        details = {
            "retrieved_evidence": retrieved_evidence[:300] + ("..." if len(retrieved_evidence) > 300 else ""),
            "relevance_score": round(relevance_score, 4),
            "entity_coverage": round(entity_coverage, 4),
            "entities_found": entities_found,
            "entities_missing": entities_missing
        }

        return SignalResult(
            signal_name=self.name,
            score=hallucination_risk,
            confidence=0.85 if retrieved_evidence else 0.60,
            details=details,
            explanation=explanation
        )
