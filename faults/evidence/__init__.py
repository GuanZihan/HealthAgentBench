"""Validated evidence-degradation recipes for HealthAgentBench faults."""

from .recipes import (
    EVIDENCE_SCENARIOS,
    EvidenceRecipeError,
    apply_evidence_recipe,
    load_evidence_recipe,
)

__all__ = [
    "EVIDENCE_SCENARIOS",
    "EvidenceRecipeError",
    "apply_evidence_recipe",
    "load_evidence_recipe",
]
