

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query

from src.api.schemas.lead import LeadPriorityRequest, LeadPriorityResponse
from src.services.lead_scorer import LeadScorer
from src.config import get_settings
from models.llm_client import LLMClient, LLMClientError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Lead Priority"])


_llm_client: Optional[LLMClient] = None
_lead_scorer: Optional[LeadScorer] = None


def get_lead_scorer() -> LeadScorer:
    """Get or create the lead scorer with optional LLM."""
    global _llm_client, _lead_scorer
    
    if _lead_scorer is None:
        try:
            _llm_client = LLMClient()
            _lead_scorer = LeadScorer(llm_client=_llm_client)
            logger.info("Lead scorer initialized with LLM client")
        except Exception as e:
            logger.warning(f"Could not initialize LLM client: {e}. Using deterministic scoring only.")
            _lead_scorer = LeadScorer(llm_client=None)
    
    return _lead_scorer


@router.post("/lead-priority", response_model=LeadPriorityResponse)
async def prioritize_leads(
    request: LeadPriorityRequest,
    use_llm: bool = Query(True, description="Whether to use LLM for notes analysis")
):
    """
    Prioritize leads based on multiple factors.
    
    This endpoint analyzes a list of leads and returns them ranked by priority,
    using a hybrid approach of deterministic rules and LLM-based text analysis.
    
    **Scoring Factors:**
    - Recency of activity (25%)
    - Engagement level (20%)
    - Source quality (15%)
    - Budget (20%)
    - Notes/signals (20%)
    
    **Priority Buckets:**
    - Hot: score >= 0.7
    - Warm: 0.4 <= score < 0.7
    - Cold: score < 0.4
    """
    try:
        scorer = get_lead_scorer()
        settings = get_settings()
        
        
        ranked_leads = await scorer.prioritize_leads(
            leads=request.leads,
            max_results=request.max_results,
            use_llm=use_llm and scorer.llm_client is not None
        )
        
        
        model_metadata = {
            "model_used": scorer.llm_client.model_name if scorer.llm_client and use_llm else "deterministic",
            "llm_enabled": use_llm and scorer.llm_client is not None,
            "scoring_weights": {
                "recency": 0.25,
                "engagement": 0.20,
                "source": 0.15,
                "budget": 0.20,
                "notes": 0.20
            },
            "thresholds": {
                "hot": settings.hot_threshold,
                "warm": settings.warm_threshold
            }
        }
        
        return LeadPriorityResponse(
            ranked_leads=ranked_leads,
            total_processed=len(request.leads),
            model_metadata=model_metadata
        )
        
    except LLMClientError as e:
        logger.error(f"LLM error during lead prioritization: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM service error: {str(e)}. Try with use_llm=false"
        )
    except Exception as e:
        logger.exception(f"Error prioritizing leads: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@router.get("/lead-priority/health")
async def lead_priority_health():
    """Check health of lead priority service."""
    scorer = get_lead_scorer()
    
    llm_available = False
    if scorer.llm_client:
        llm_available = await scorer.llm_client.health_check()
    
    return {
        "status": "healthy",
        "llm_available": llm_available,
        "scoring_mode": "hybrid" if llm_available else "deterministic"
    }
