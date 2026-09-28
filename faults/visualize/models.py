"""Structured models shared by the analyzer and Streamlit UI."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class BehaviorPhase(str, Enum):
    """Neutral activity phases used to organize a rollout."""

    orientation = "orientation"
    inspection = "inspection"
    analysis = "analysis"
    implementation = "implementation"
    validation = "validation"
    finalization = "finalization"
    other = "other"


class BehaviorBlock(BaseModel):
    """A contiguous, semantically coherent section of an agent trajectory."""

    block_id: str = Field(description="Short stable identifier such as B01")
    title: str = Field(description="Concise activity title")
    phase: BehaviorPhase
    start_step: int
    end_step: int
    intent: str = Field(description="What the agent was trying to accomplish")
    summary: str = Field(description="What the agent did in this block")
    outcome: str = Field(description="Observable result at the end of the block")
    tools: list[str] = Field(description="Distinct tools used in this block")
    evidence_step_ids: list[int] = Field(
        description="Most informative trajectory step IDs supporting the summary"
    )


class RolloutNarrative(BaseModel):
    """Model-produced neutral summary of one rollout."""

    headline: str = Field(description="One-line description of the rollout")
    overview: str = Field(description="Short factual overview of the approach")
    strategy: str = Field(description="The overall strategy and how it evolved")
    final_outcome: str = Field(description="What was produced and how the run ended")
    blocks: list[BehaviorBlock]


class AnalysisDocument(BaseModel):
    """Persisted analysis with provenance and source invalidation metadata."""

    schema_version: Literal["1"] = "1"
    generated_at: datetime
    generator: Literal["openai", "local"]
    model: str | None
    source_fingerprint: str
    response_id: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    analysis: RolloutNarrative
