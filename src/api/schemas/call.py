

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class CallEvalRequest(BaseModel):
    """Request body for call evaluation endpoint."""
    
    call_id: str = Field(..., description="Unique identifier for the call")
    lead_id: Optional[str] = Field(None, description="Associated lead ID if available")
    transcript: str = Field(..., min_length=10, description="Call transcript text")
    duration_seconds: Optional[int] = Field(None, ge=0, description="Call duration in seconds")


class CallLabels(BaseModel):
    """Quality labels for different aspects of the call."""
    
    rapport_building: float = Field(..., ge=0, le=1, description="Score for rapport building")
    need_discovery: float = Field(..., ge=0, le=1, description="Score for need discovery")
    closing_attempt: float = Field(..., ge=0, le=1, description="Score for closing attempt")
    compliance_risk: float = Field(..., ge=0, le=1, description="Score for compliance risk (lower is better)")


class ModelMetadata(BaseModel):
    """Metadata about the LLM model used."""
    
    model_name: str = Field(..., description="Name of the model used")
    latency_ms: int = Field(..., ge=0, description="Latency in milliseconds")
    input_tokens: Optional[int] = Field(None, description="Number of input tokens")
    output_tokens: Optional[int] = Field(None, description="Number of output tokens")


class CallEvalResponse(BaseModel):
    """Response body for call evaluation endpoint."""
    
    call_id: str = Field(..., description="Call identifier")
    quality_score: float = Field(..., ge=0, le=1, description="Overall quality score")
    labels: CallLabels = Field(..., description="Detailed quality labels")
    summary: str = Field(..., description="Summary of the call")
    next_actions: List[str] = Field(..., description="Recommended next actions")
    model_metadata: ModelMetadata = Field(..., description="Model metadata")
    
    
    is_good_call: bool = Field(..., description="Whether the call is classified as good")
    key_points: List[str] = Field(default_factory=list, description="Key points from the conversation")
