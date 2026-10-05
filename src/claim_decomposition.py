"""
Claim and Atomic Fact Decomposition Module.
Segments raw LLM responses into sentences and atomic factual claims,
identifying key entities and spans for claim-level hallucination localization.
"""

import re
from typing import List, Dict, Any, Tuple

class DecomposedClaim:
    """Represents a single atomic factual claim extracted from LLM text."""
    def __init__(
        self,
        claim_id: int,
        text: str,
        sentence_text: str,
        start_char: int,
        end_char: int,
        entities: List[str]
    ):
        self.claim_id = claim_id
        self.text = text.strip()
        self.sentence_text = sentence_text.strip()
        self.start_char = start_char
        self.end_char = end_char
        self.entities = entities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.text,
            "sentence_text": self.sentence_text,
            "span": (self.start_char, self.end_char),
            "entities": self.entities
        }


class ClaimDecomposer:
    """Decomposes an LLM response into atomic verifiable claims."""

    def __init__(self):
        # Conjunctions and punctuation used to decompose compound sentences into atomic claims
        self.conjunction_pattern = re.compile(
            r'(?:,\s*and\s+|,\s*but\s+|;\s*|\s+whereas\s+|\s+while\s+)',
            re.IGNORECASE
        )
        # Regex for capitalized entity / noun phrase extraction
        self.entity_pattern = re.compile(
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[0-9]{4}\b|\b[0-9]+(?:\.[0-9]+)?\b'
        )

    def split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Splits text into sentences along with their character start/end offsets."""
        sentences = []
        # Fallback regex-based sentence splitter if NLTK is not downloaded
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
        
        last_idx = 0
        for match in sentence_endings.finditer(text):
            sent = text[last_idx:match.start()].strip()
            if sent:
                sentences.append((sent, last_idx, last_idx + len(sent)))
            last_idx = match.end()
            
        remaining = text[last_idx:].strip()
        if remaining:
            sentences.append((remaining, last_idx, last_idx + len(remaining)))
            
        return sentences

    def extract_entities(self, text: str) -> List[str]:
        """Extracts candidate entities, proper nouns, and numbers for retrieval queries."""
        matches = self.entity_pattern.findall(text)
        # Deduplicate while preserving order
        seen = set()
        entities = []
        for m in matches:
            m_clean = m.strip()
            if len(m_clean) > 1 and m_clean.lower() not in seen:
                seen.add(m_clean.lower())
                entities.append(m_clean)
        return entities

    def decompose(self, response_text: str) -> List[DecomposedClaim]:
        """Decomposes response into atomic claims with offsets and entities."""
        if not response_text or not response_text.strip():
            return []

        sentences = self.split_sentences(response_text)
        claims = []
        claim_counter = 1

        for sent_text, sent_start, sent_end in sentences:
            # Check if sentence is compound and can be broken into atomic clauses
            parts = self.conjunction_pattern.split(sent_text)
            
            # If compound sentence with 2+ informative clauses
            if len(parts) > 1 and all(len(p.strip().split()) >= 3 for p in parts):
                offset = 0
                for part in parts:
                    part_str = part.strip()
                    part_start = sent_text.find(part_str, offset)
                    part_end = part_start + len(part_str)
                    offset = part_end
                    
                    entities = self.extract_entities(part_str)
                    claims.append(DecomposedClaim(
                        claim_id=claim_counter,
                        text=part_str,
                        sentence_text=sent_text,
                        start_char=sent_start + part_start,
                        end_char=sent_start + part_end,
                        entities=entities
                    ))
                    claim_counter += 1
            else:
                entities = self.extract_entities(sent_text)
                claims.append(DecomposedClaim(
                    claim_id=claim_counter,
                    text=sent_text,
                    sentence_text=sent_text,
                    start_char=sent_start,
                    end_char=sent_end,
                    entities=entities
                ))
                claim_counter += 1

        return claims
