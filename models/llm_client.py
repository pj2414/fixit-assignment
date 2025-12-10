

import json
import logging
import time
from typing import Dict, Any, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import get_settings
from models.prompts import (
    LEAD_NOTES_ANALYSIS_PROMPT,
    CALL_QUALITY_ANALYSIS_PROMPT,
    CALL_EVALUATION_SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMClientError(Exception):
    """Custom exception for LLM client errors."""
    pass


class LLMClient:
    """LangChain-based LLM client for Fixit operations."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        """Initialize the LLM client with Ollama."""
        self.model_name = model_name or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.timeout = timeout or settings.llm_timeout
        
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.1,  
            format="json"  
        )
        
        self.json_parser = JsonOutputParser()
        
        logger.info(f"LLM Client initialized with model: {self.model_name}")
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common issues."""
        try:
           
            return json.loads(content)
        except json.JSONDecodeError:
          
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                return json.loads(json_match.group(1))
            
            
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group(0))
            
            raise LLMClientError(f"Could not parse JSON from response: {content[:200]}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def analyze_lead_notes(self, notes: str) -> Dict[str, Any]:
        """
        Analyze lead notes using LLM.
        
        Returns:
            Dict with 'score' (float) and 'reasons' (list of strings)
        """
        start_time = time.time()
        
        try:
            prompt = LEAD_NOTES_ANALYSIS_PROMPT.format(notes=notes)
            messages = [HumanMessage(content=prompt)]
            
            response = await self.llm.ainvoke(messages)
            latency_ms = int((time.time() - start_time) * 1000)
            
            result = self._parse_json_response(response.content)
            
            
            score = float(result.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            
            reasons = result.get("reasons", [])
            if result.get("red_flags"):
                reasons.append(f"Red flags: {', '.join(result['red_flags'])}")
            
            logger.info(f"Lead notes analysis completed in {latency_ms}ms")
            
            return {
                "score": score,
                "reasons": reasons,
                "latency_ms": latency_ms,
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Lead notes analysis failed: {e}")
            raise LLMClientError(f"Failed to analyze lead notes: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def analyze_call_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze call transcript using LLM.
        
        Returns:
            Dict with quality scores, summary, and recommendations
        """
        start_time = time.time()
        
        try:
            prompt = CALL_QUALITY_ANALYSIS_PROMPT.format(transcript=transcript)
            messages = [
                SystemMessage(content=CALL_EVALUATION_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            latency_ms = int((time.time() - start_time) * 1000)
            
            result = self._parse_json_response(response.content)
            
            
            labels = {
                "rapport_building": max(0.0, min(1.0, float(result.get("rapport_building", 0.5)))),
                "need_discovery": max(0.0, min(1.0, float(result.get("need_discovery", 0.5)))),
                "closing_attempt": max(0.0, min(1.0, float(result.get("closing_attempt", 0.5)))),
                "compliance_risk": max(0.0, min(1.0, float(result.get("compliance_risk", 0.5))))
            }
            
            
            quality_score = (
                labels["rapport_building"] * 0.25 +
                labels["need_discovery"] * 0.30 +
                labels["closing_attempt"] * 0.30 +
                (1 - labels["compliance_risk"]) * 0.15
            )
            
            logger.info(f"Call transcript analysis completed in {latency_ms}ms")
            
            return {
                "labels": labels,
                "quality_score": round(quality_score, 3),
                "summary": result.get("summary", "Call analysis completed."),
                "key_points": result.get("key_points", []),
                "next_actions": result.get("next_actions", []),
                "latency_ms": latency_ms,
                "model": self.model_name,
                "input_length": len(transcript)
            }
            
        except Exception as e:
            logger.error(f"Call transcript analysis failed: {e}")
            raise LLMClientError(f"Failed to analyze call transcript: {e}")
    
    async def health_check(self) -> bool:
        """Check if the LLM service is available."""
        try:
            response = await self.llm.ainvoke([HumanMessage(content='{"status": "ok"}')])
            return True
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False
