
import logging
import time
from typing import TypedDict, List, Optional, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.api.schemas.call import CallEvalRequest, CallEvalResponse, CallLabels, ModelMetadata
from src.config import get_settings
from models.prompts import CALL_QUALITY_ANALYSIS_PROMPT, CALL_EVALUATION_SYSTEM_PROMPT
from models.llm_client import LLMClient

logger = logging.getLogger(__name__)
settings = get_settings()


class CallAnalysisState(TypedDict):
    """State for the call analysis workflow."""
    
   
    call_id: str
    lead_id: Optional[str]
    transcript: str
    duration_seconds: Optional[int]
    
    
    is_parsed: bool
    parse_error: Optional[str]
    
   
    rapport_building: float
    need_discovery: float
    closing_attempt: float
    compliance_risk: float
    summary: str
    key_points: List[str]
    next_actions: List[str]
    
    
    quality_score: float
    is_good_call: bool
    
    
    model_name: str
    latency_ms: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    errors: Annotated[List[str], add]


class CallAnalyzer:
    """
    LangGraph-based call analyzer with stateful workflow.
    
    Workflow:
    1. Parse Transcript - Validate and preprocess
    2. Analyze Dimensions - Use LLM to score each dimension
    3. Calculate Score - Compute overall quality score
    4. Generate Summary - Create actionable summary
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the call analyzer with LLM client."""
        self.llm_client = llm_client or LLMClient()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        workflow = StateGraph(CallAnalysisState)
        
       
        workflow.add_node("parse_transcript", self._parse_transcript)
        workflow.add_node("analyze_dimensions", self._analyze_dimensions)
        workflow.add_node("calculate_score", self._calculate_score)
        workflow.add_node("generate_output", self._generate_output)
        
        
        workflow.set_entry_point("parse_transcript")
        workflow.add_conditional_edges(
            "parse_transcript",
            self._should_continue_after_parse,
            {
                "continue": "analyze_dimensions",
                "error": "generate_output"
            }
        )
        workflow.add_edge("analyze_dimensions", "calculate_score")
        workflow.add_edge("calculate_score", "generate_output")
        workflow.add_edge("generate_output", END)
        
        return workflow.compile()
    
    def _parse_transcript(self, state: CallAnalysisState) -> CallAnalysisState:
        """Parse and validate the transcript."""
        logger.info(f"Parsing transcript for call {state['call_id']}")
        
        transcript = state["transcript"]
        
        
        if not transcript or len(transcript.strip()) < 20:
            return {
                **state,
                "is_parsed": False,
                "parse_error": "Transcript too short or empty",
                "errors": ["Transcript validation failed: too short"]
            }
        
        
        cleaned = transcript.strip()
        
       
        has_dialogue = any(marker in cleaned for marker in [":", "Agent:", "Customer:", "A:", "C:"])
        
        if not has_dialogue:
            logger.warning(f"Call {state['call_id']}: Transcript may not be a dialogue")
        
        return {
            **state,
            "transcript": cleaned,
            "is_parsed": True,
            "parse_error": None
        }
    
    def _should_continue_after_parse(self, state: CallAnalysisState) -> str:
        """Decide whether to continue analysis or skip to output."""
        if state.get("is_parsed", False):
            return "continue"
        return "error"
    
    async def _analyze_dimensions(self, state: CallAnalysisState) -> CallAnalysisState:
        """Analyze call using LLM across all dimensions."""
        logger.info(f"Analyzing dimensions for call {state['call_id']}")
        
        start_time = time.time()
        
        try:
            result = await self.llm_client.analyze_call_transcript(state["transcript"])
            
            labels = result["labels"]
            
            return {
                **state,
                "rapport_building": labels["rapport_building"],
                "need_discovery": labels["need_discovery"],
                "closing_attempt": labels["closing_attempt"],
                "compliance_risk": labels["compliance_risk"],
                "summary": result["summary"],
                "key_points": result["key_points"],
                "next_actions": result["next_actions"],
                "model_name": result["model"],
                "latency_ms": result["latency_ms"]
            }
            
        except Exception as e:
            logger.error(f"Dimension analysis failed for call {state['call_id']}: {e}")
            return {
                **state,
                "rapport_building": 0.5,
                "need_discovery": 0.5,
                "closing_attempt": 0.5,
                "compliance_risk": 0.5,
                "summary": "Analysis failed - using default scores",
                "key_points": [],
                "next_actions": ["Manual review required"],
                "model_name": "fallback",
                "latency_ms": int((time.time() - start_time) * 1000),
                "errors": [f"LLM analysis failed: {str(e)}"]
            }
    
    def _calculate_score(self, state: CallAnalysisState) -> CallAnalysisState:
        """Calculate overall quality score from dimensions."""
        logger.info(f"Calculating score for call {state['call_id']}")
        
       
        quality_score = (
            state["rapport_building"] * 0.25 +
            state["need_discovery"] * 0.30 +
            state["closing_attempt"] * 0.30 +
            (1 - state["compliance_risk"]) * 0.15
        )
        
        quality_score = round(quality_score, 3)
        is_good_call = quality_score >= settings.good_call_threshold
        
        return {
            **state,
            "quality_score": quality_score,
            "is_good_call": is_good_call
        }
    
    def _generate_output(self, state: CallAnalysisState) -> CallAnalysisState:
        """Finalize the output state."""
        logger.info(f"Generating output for call {state['call_id']}")
        
        
        if not state.get("summary"):
            state["summary"] = "Call analysis completed."
        if not state.get("next_actions"):
            state["next_actions"] = []
        if not state.get("key_points"):
            state["key_points"] = []
        
        return state
    
    async def analyze(self, request: CallEvalRequest) -> CallEvalResponse:
        """
        Run the complete call analysis workflow.
        
        Args:
            request: Call evaluation request with transcript
            
        Returns:
            CallEvalResponse with scores, labels, and recommendations
        """
        
        initial_state: CallAnalysisState = {
            "call_id": request.call_id,
            "lead_id": request.lead_id,
            "transcript": request.transcript,
            "duration_seconds": request.duration_seconds,
            "is_parsed": False,
            "parse_error": None,
            "rapport_building": 0.0,
            "need_discovery": 0.0,
            "closing_attempt": 0.0,
            "compliance_risk": 0.0,
            "summary": "",
            "key_points": [],
            "next_actions": [],
            "quality_score": 0.0,
            "is_good_call": False,
            "model_name": settings.ollama_model,
            "latency_ms": 0,
            "input_tokens": None,
            "output_tokens": None,
            "errors": []
        }
        
        
        final_state = await self.graph.ainvoke(initial_state)
        
        
        return CallEvalResponse(
            call_id=final_state["call_id"],
            quality_score=final_state["quality_score"],
            labels=CallLabels(
                rapport_building=final_state["rapport_building"],
                need_discovery=final_state["need_discovery"],
                closing_attempt=final_state["closing_attempt"],
                compliance_risk=final_state["compliance_risk"]
            ),
            summary=final_state["summary"],
            next_actions=final_state["next_actions"],
            model_metadata=ModelMetadata(
                model_name=final_state["model_name"],
                latency_ms=final_state["latency_ms"],
                input_tokens=final_state.get("input_tokens"),
                output_tokens=final_state.get("output_tokens")
            ),
            is_good_call=final_state["is_good_call"],
            key_points=final_state["key_points"]
        )



_analyzer_instance: Optional[CallAnalyzer] = None


def get_call_analyzer() -> CallAnalyzer:
    """Get or create the call analyzer singleton."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = CallAnalyzer()
    return _analyzer_instance
