

import pytest
import asyncio
from src.services.lead_scorer import LeadScorer
from src.api.schemas.lead import LeadInput


class TestRecencyScore:
    """Tests for recency score calculation."""
    
    def setup_method(self):
        self.scorer = LeadScorer()
    
    def test_very_recent_activity(self):
        """Activity within 30 minutes should get max score."""
        score, reason = self.scorer.calculate_recency_score(15)
        assert score == 1.0
        assert "< 30 mins" in reason
    
    def test_recent_activity(self):
        """Activity within 1 hour should get high score."""
        score, reason = self.scorer.calculate_recency_score(45)
        assert score == 0.85
        assert "< 1 hour" in reason
    
    def test_within_4_hours(self):
        """Activity within 4 hours."""
        score, reason = self.scorer.calculate_recency_score(180)
        assert score == 0.70
    
    def test_within_24_hours(self):
        """Activity within 24 hours."""
        score, reason = self.scorer.calculate_recency_score(720)
        assert score == 0.50
    
    def test_within_week(self):
        """Activity within 7 days."""
        score, reason = self.scorer.calculate_recency_score(5000)
        assert score == 0.25
    
    def test_old_lead(self):
        """Activity older than 7 days should get lowest score."""
        score, reason = self.scorer.calculate_recency_score(15000)
        assert score == 0.10
        assert "Old lead" in reason


class TestEngagementScore:
    """Tests for engagement score calculation."""
    
    def setup_method(self):
        self.scorer = LeadScorer()
    
    def test_high_engagement(self):
        """Multiple interactions should give high score."""
        score, reason = self.scorer.calculate_engagement_score(8, "contacted")
        assert score >= 0.8
        assert "Highly engaged" in reason
    
    def test_moderate_engagement(self):
        """Few interactions with follow_up status."""
        score, reason = self.scorer.calculate_engagement_score(3, "follow_up")
        assert 0.4 <= score <= 0.6
    
    def test_low_engagement(self):
        """No interactions, new lead."""
        score, reason = self.scorer.calculate_engagement_score(0, "new")
        assert score == 0.0
        assert "Low engagement" in reason
    
    def test_score_capped_at_1(self):
        """Score should not exceed 1.0."""
        score, _ = self.scorer.calculate_engagement_score(20, "follow_up")
        assert score <= 1.0


class TestSourceScore:
    """Tests for source quality score."""
    
    def setup_method(self):
        self.scorer = LeadScorer()
    
    def test_referral_highest(self):
        """Referral should be highest quality source."""
        score, reason = self.scorer.calculate_source_score("referral")
        assert score == 1.0
        assert "High-quality" in reason
    
    def test_walk_in_high(self):
        """Walk-in is high quality source."""
        score, _ = self.scorer.calculate_source_score("walk-in")
        assert score == 0.9
    
    def test_portal_good(self):
        """Portal sources are good."""
        score, _ = self.scorer.calculate_source_score("magicbricks")
        assert score == 0.75
    
    def test_social_media_lower(self):
        """Social media is lower quality."""
        score, _ = self.scorer.calculate_source_score("social_media")
        assert score == 0.4
    
    def test_unknown_source(self):
        """Unknown source gets default score."""
        score, _ = self.scorer.calculate_source_score("random_source")
        assert score == 0.5


class TestBudgetScore:
    """Tests for budget score calculation."""
    
    def setup_method(self):
        self.scorer = LeadScorer()
    
    def test_premium_budget(self):
        """Budget above 5 Cr is premium."""
        score, reason = self.scorer.calculate_budget_score(60000000)
        assert score == 1.0
        assert "Premium" in reason
    
    def test_high_budget(self):
        """Budget 2-5 Cr is high."""
        score, reason = self.scorer.calculate_budget_score(30000000)
        assert score == 0.85
        assert "High" in reason
    
    def test_good_budget(self):
        """Budget 1-2 Cr is good."""
        score, _ = self.scorer.calculate_budget_score(15000000)
        assert score == 0.70
    
    def test_lower_budget(self):
        """Budget below 50L."""
        score, _ = self.scorer.calculate_budget_score(3000000)
        assert score == 0.40


class TestNotesAnalysis:
    """Tests for deterministic notes analysis."""
    
    def setup_method(self):
        self.scorer = LeadScorer()
    
    def test_urgency_keywords(self):
        """Notes with urgency signals should score higher."""
        score, reasons = self.scorer.analyze_notes_deterministic(
            "Client is very interested. URGENT!! Wants to book ASAP."
        )
        assert score > 0.5
        assert any("Urgency" in r for r in reasons)
    
    def test_timeline_keywords(self):
        """Notes with timeline mentions."""
        score, reasons = self.scorer.analyze_notes_deterministic(
            "Looking for possession by March. Relocating from Pune."
        )
        assert score > 0.5
        assert any("Timeline" in r for r in reasons)
    
    def test_positive_keywords(self):
        """Notes with positive signals."""
        score, reasons = self.scorer.analyze_notes_deterministic(
            "Genuine buyer, very interested. Cash buyer ready."
        )
        assert score > 0.5
    
    def test_negative_keywords(self):
        """Notes with negative signals should score lower."""
        score, reasons = self.scorer.analyze_notes_deterministic(
            "Not serious, just window shopping. Wrong number."
        )
        assert score < 0.5
        assert any("Negative" in r for r in reasons)
    
    def test_empty_notes(self):
        """Empty notes should return neutral score."""
        score, reasons = self.scorer.analyze_notes_deterministic("")
        assert score == 0.5
        assert "No notes" in reasons[0]


class TestLeadScoring:
    """Integration tests for lead scoring."""
    
    def setup_method(self):
        self.scorer = LeadScorer()
    
    @pytest.mark.asyncio
    async def test_hot_lead_scoring(self, sample_lead):
        """Hot lead should get high score."""
        lead = LeadInput(**sample_lead)
        result = await self.scorer.score_lead(lead, use_llm=False)
        
        assert result.priority_score >= 0.7
        assert result.priority_bucket == "hot"
        assert len(result.reasons) > 0
    
    @pytest.mark.asyncio
    async def test_cold_lead_scoring(self, sample_cold_lead):
        """Cold lead should get low score."""
        lead = LeadInput(**sample_cold_lead)
        result = await self.scorer.score_lead(lead, use_llm=False)
        
        assert result.priority_score < 0.4
        assert result.priority_bucket == "cold"
    
    @pytest.mark.asyncio
    async def test_lead_prioritization(self, sample_lead, sample_cold_lead):
        """Multiple leads should be properly ranked."""
        leads = [
            LeadInput(**sample_cold_lead),
            LeadInput(**sample_lead)
        ]
        
        results = await self.scorer.prioritize_leads(leads, max_results=10, use_llm=False)
        
        # Hot lead should come first
        assert results[0].lead_id == "TEST001"
        assert results[1].lead_id == "TEST002"
    
    @pytest.mark.asyncio
    async def test_max_results_limit(self, sample_lead):
        """Max results should limit output."""
        leads = [LeadInput(**{**sample_lead, "lead_id": f"LEAD{i}"}) for i in range(10)]
        
        results = await self.scorer.prioritize_leads(leads, max_results=5, use_llm=False)
        
        assert len(results) == 5
