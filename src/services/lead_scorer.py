

import re
import logging
from typing import List, Dict, Tuple

from src.api.schemas.lead import LeadInput, LeadPriorityScore
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LeadScorer:
    """Service for scoring and prioritizing leads using hybrid approach."""
    
    
    SOURCE_SCORES = {
        "referral": 1.0,
        "walk-in": 0.9,
        "portal": 0.75,
        "magicbricks": 0.75,
        "99acres": 0.75,
        "housing.com": 0.7,
        "website": 0.6,
        "social_media": 0.4,
    }
    
    
    URGENCY_KEYWORDS = [
        "urgent", "asap", "immediately", "priority", "today", "tomorrow",
        "this week", "this weekend", "ready to book", "ready to buy",
        "booking amount ready", "cash ready", "!!", "PRIORITY", "VIP"
    ]
    
    TIMELINE_KEYWORDS = [
        "march", "april", "diwali", "pongal", "next month", "by end of",
        "within", "before", "possession", "shifting", "relocating"
    ]
    
    POSITIVE_KEYWORDS = [
        "interested", "likes", "loved", "genuine", "serious", "confirmed",
        "scheduled", "ready", "approved", "flexible", "cash buyer"
    ]
    
    NEGATIVE_KEYWORDS = [
        "not serious", "fake", "spam", "wrong number", "window shopping",
        "not picking", "not interested", "unrealistic", "just browsing"
    ]
    
    def __init__(self, llm_client=None):
        """Initialize scorer with optional LLM client."""
        self.llm_client = llm_client
    
    def calculate_recency_score(self, minutes_ago: int) -> Tuple[float, str]:
        """
        Calculate score based on recency of last activity.
        Returns (score, reason)
        """
        if minutes_ago < 30:
            return 1.0, "Very recent activity (< 30 mins)"
        elif minutes_ago < 60:
            return 0.85, "Recent activity (< 1 hour)"
        elif minutes_ago < 240:
            return 0.70, "Activity within 4 hours"
        elif minutes_ago < 1440:
            return 0.50, "Activity within 24 hours"
        elif minutes_ago < 10080:
            return 0.25, "Activity within 7 days"
        else:
            return 0.10, "Old lead (> 7 days since activity)"
    
    def calculate_engagement_score(self, past_interactions: int, status: str) -> Tuple[float, str]:
        """
        Calculate score based on engagement level.
        Returns (score, reason)
        """
       
        interaction_score = min(past_interactions / 10, 1.0)
        
        
        status_modifier = {
            "contacted": 0.1,
            "follow_up": 0.15,
            "new": 0.0
        }
        
        base_score = interaction_score + status_modifier.get(status, 0)
        final_score = min(base_score, 1.0)
        
        if past_interactions >= 5:
            reason = f"Highly engaged ({past_interactions} interactions)"
        elif past_interactions >= 2:
            reason = f"Moderate engagement ({past_interactions} interactions)"
        else:
            reason = f"Low engagement ({past_interactions} interactions)"
        
        return final_score, reason
    
    def calculate_source_score(self, source: str) -> Tuple[float, str]:
        """
        Calculate score based on lead source quality.
        Returns (score, reason)
        """
        source_lower = source.lower()
        score = self.SOURCE_SCORES.get(source_lower, 0.5)
        
        if score >= 0.9:
            reason = f"High-quality source ({source})"
        elif score >= 0.7:
            reason = f"Good source ({source})"
        else:
            reason = f"Standard source ({source})"
        
        return score, reason
    
    def calculate_budget_score(self, budget: float) -> Tuple[float, str]:
        """
        Calculate score based on budget (higher budget = more valuable lead).
        Returns (score, reason)
        """
        
        budget_cr = budget / 10000000
        
        if budget_cr >= 5:
            return 1.0, f"Premium budget (₹{budget_cr:.1f}Cr)"
        elif budget_cr >= 2:
            return 0.85, f"High budget (₹{budget_cr:.1f}Cr)"
        elif budget_cr >= 1:
            return 0.70, f"Good budget (₹{budget_cr:.1f}Cr)"
        elif budget_cr >= 0.5:
            return 0.55, f"Moderate budget (₹{budget/100000:.0f}L)"
        else:
            return 0.40, f"Lower budget segment (₹{budget/100000:.0f}L)"
    
    def analyze_notes_deterministic(self, notes: str) -> Tuple[float, List[str]]:
        """
        Analyze notes using keyword matching (deterministic fallback).
        Returns (score, reasons)
        """
        if not notes:
            return 0.5, ["No notes available"]
        
        notes_lower = notes.lower()
        score = 0.5
        reasons = []
        
        
        urgency_matches = [kw for kw in self.URGENCY_KEYWORDS if kw.lower() in notes_lower]
        if urgency_matches:
            score += 0.2
            reasons.append(f"Urgency signals detected: {', '.join(urgency_matches[:2])}")
        
        
        timeline_matches = [kw for kw in self.TIMELINE_KEYWORDS if kw.lower() in notes_lower]
        if timeline_matches:
            score += 0.15
            reasons.append(f"Timeline mentioned: {timeline_matches[0]}")
        
        
        positive_matches = [kw for kw in self.POSITIVE_KEYWORDS if kw.lower() in notes_lower]
        if positive_matches:
            score += 0.15
            reasons.append(f"Positive signals: {', '.join(positive_matches[:2])}")
        
        
        negative_matches = [kw for kw in self.NEGATIVE_KEYWORDS if kw.lower() in notes_lower]
        if negative_matches:
            score -= 0.3
            reasons.append(f"Negative signals: {', '.join(negative_matches[:2])}")
        
        
        score = max(0.0, min(1.0, score))
        
        if not reasons:
            reasons.append("Neutral notes content")
        
        return score, reasons
    
    async def analyze_notes_with_llm(self, notes: str) -> Tuple[float, List[str]]:
        """
        Analyze notes using LLM for deeper understanding.
        Returns (score, reasons)
        """
        if not self.llm_client or not notes:
            return self.analyze_notes_deterministic(notes)
        
        try:
            result = await self.llm_client.analyze_lead_notes(notes)
            return result["score"], result["reasons"]
        except Exception as e:
            logger.warning(f"LLM analysis failed, falling back to deterministic: {e}")
            return self.analyze_notes_deterministic(notes)
    
    async def score_lead(self, lead: LeadInput, use_llm: bool = True) -> LeadPriorityScore:
        """
        Calculate comprehensive priority score for a lead.
        """
        reasons = []
        
        
        recency_score, recency_reason = self.calculate_recency_score(lead.last_activity_minutes_ago)
        reasons.append(recency_reason)
        
        engagement_score, engagement_reason = self.calculate_engagement_score(
            lead.past_interactions, lead.status
        )
        reasons.append(engagement_reason)
        
        source_score, source_reason = self.calculate_source_score(lead.source)
        reasons.append(source_reason)
        
        budget_score, budget_reason = self.calculate_budget_score(lead.budget)
        reasons.append(budget_reason)
        
        
        if use_llm:
            notes_score, notes_reasons = await self.analyze_notes_with_llm(lead.notes)
        else:
            notes_score, notes_reasons = self.analyze_notes_deterministic(lead.notes)
        reasons.extend(notes_reasons)
        
        
        priority_score = (
            recency_score * 0.25 +
            engagement_score * 0.20 +
            source_score * 0.15 +
            budget_score * 0.20 +
            notes_score * 0.20
        )
        
        
        if priority_score >= settings.hot_threshold:
            bucket = "hot"
        elif priority_score >= settings.warm_threshold:
            bucket = "warm"
        else:
            bucket = "cold"
        
        return LeadPriorityScore(
            lead_id=lead.lead_id,
            priority_score=round(priority_score, 3),
            priority_bucket=bucket,
            reasons=reasons,
            recency_score=round(recency_score, 3),
            engagement_score=round(engagement_score, 3),
            source_score=round(source_score, 3),
            budget_score=round(budget_score, 3),
            notes_score=round(notes_score, 3)
        )
    
    async def prioritize_leads(
        self, 
        leads: List[LeadInput], 
        max_results: int = 10,
        use_llm: bool = True
    ) -> List[LeadPriorityScore]:
        """
        Score and rank multiple leads.
        """
        scored_leads = []
        
        for lead in leads:
            scored = await self.score_lead(lead, use_llm=use_llm)
            scored_leads.append(scored)
        
        
        scored_leads.sort(key=lambda x: x.priority_score, reverse=True)
        
        return scored_leads[:max_results]
