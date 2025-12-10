

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.api.schemas.call import CallEvalRequest, CallEvalResponse
from src.services.call_analyzer import CallAnalyzer, get_call_analyzer
from models.llm_client import LLMClientError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Call Evaluation"])


@router.post("/call-eval", response_model=CallEvalResponse)
async def evaluate_call(request: CallEvalRequest):
    """
    Evaluate call quality using LangGraph workflow.
    
    This endpoint analyzes a call transcript and returns quality scores
    across multiple dimensions using a stateful LangGraph workflow.
    
    **Evaluation Dimensions:**
    - Rapport Building (25%): Greeting, empathy, personalization
    - Need Discovery (30%): Questions asked, requirements captured
    - Closing Attempt (30%): Clear next steps, commitment sought
    - Compliance Risk (15%): False promises, pressure tactics (inverted)
    
    **Workflow Steps:**
    1. Parse Transcript - Validate and preprocess
    2. Analyze Dimensions - LLM-based multi-dimensional analysis
    3. Calculate Score - Weighted quality score
    4. Generate Output - Summary and recommendations
    
    **Quality Classification:**
    - Good Call: quality_score >= 0.6
    - Bad Call: quality_score < 0.6
    """
    try:
        analyzer = get_call_analyzer()
        
        
        result = await analyzer.analyze(request)
        
        logger.info(
            f"Call {request.call_id} evaluated: "
            f"score={result.quality_score}, "
            f"good_call={result.is_good_call}, "
            f"latency={result.model_metadata.latency_ms}ms"
        )
        
        return result
        
    except LLMClientError as e:
        logger.error(f"LLM error during call evaluation: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM service error: {str(e)}"
        )
    except Exception as e:
        logger.exception(f"Error evaluating call: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@router.get("/call-eval/health")
async def call_eval_health():
    """Check health of call evaluation service."""
    try:
        analyzer = get_call_analyzer()
        llm_available = await analyzer.llm_client.health_check()
        
        return {
            "status": "healthy",
            "llm_available": llm_available,
            "workflow": "langgraph"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "llm_available": False,
            "error": str(e)
        }
