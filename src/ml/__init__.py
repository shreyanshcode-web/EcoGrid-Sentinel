"""
ML Module for Vegetation Risk Intelligence

Provides trained ML models for vegetation risk prediction,
replacing/augmenting the heuristic risk scorer.
"""

from .data_prep import prepare_training_data

__all__ = ["prepare_training_data"]
