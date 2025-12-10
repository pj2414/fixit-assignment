

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class LeadInput(BaseModel):
    """Input schema for a single lead."""
    
    lead_id: str = Field(..., description="Unique identifier for the lead")
    source: str = Field(..., description="Lead source: portal, walk-in, website, referral, etc.")
    budget: float = Field(..., description="Budget in INR", gt=0)
    city: str = Field(..., description="City of interest")
    property_type: str = Field(..., description="Type of property: 2BHK, 3BHK, villa, etc.")
    last_activity_minutes_ago: int = Field(..., description="Minutes since last activity", ge=0)
    past_interactions: int = Field(..., description="Number of past interactions", ge=0)
    notes: str = Field("", description="Free text notes about the lead")
    status: Literal["new", "contacted", "follow_up"] = Field(..., description="Lead status")


class LeadPriorityRequest(BaseModel):
    """Request body for lead priority endpoint."""
    
    leads: List[LeadInput] = Field(..., min_length=1, description="List of leads to prioritize")
    max_results: int = Field(10, ge=1, le=100, description="Maximum number of results to return")


class LeadPriorityScore(BaseModel):
    """Scoring details for a single lead."""
    
    lead_id: str = Field(..., description="Lead identifier")
    priority_score: float = Field(..., ge=0, le=1, description="Priority score between 0 and 1")
    priority_bucket: Literal["hot", "warm", "cold"] = Field(..., description="Priority bucket classification")
    reasons: List[str] = Field(..., description="List of reasons explaining the score")
    
    # Detailed scoring breakdown
    recency_score: float = Field(..., description="Score based on recency of activity")
    engagement_score: float = Field(..., description="Score based on past interactions")
    source_score: float = Field(..., description="Score based on lead source quality")
    budget_score: float = Field(..., description="Score based on budget")
    notes_score: float = Field(..., description="Score based on LLM analysis of notes")


class LeadPriorityResponse(BaseModel):
    """Response body for lead priority endpoint."""
    
    ranked_leads: List[LeadPriorityScore] = Field(..., description="Leads ranked by priority")
    total_processed: int = Field(..., description="Total number of leads processed")
    model_metadata: dict = Field(default_factory=dict, description="LLM model information")
