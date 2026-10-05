"""
HARMONIQ: Hybrid Adaptive Reliability & Multi-signal Orchestration
          Network for Interpretable Quality verification
"""

from src.harmoniq import HARMONIQ, HarmoniqFramework, HarmoniqAssessment
from src.detector import HybridHallucinationDetector

__version__ = "1.0.0"
__all__ = ["HARMONIQ", "HarmoniqFramework", "HarmoniqAssessment", "HybridHallucinationDetector"]
