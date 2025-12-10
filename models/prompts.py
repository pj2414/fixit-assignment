

LEAD_NOTES_ANALYSIS_PROMPT = """You are a real estate sales analyst. Analyze the following lead notes and provide a score from 0.0 to 1.0 indicating how likely this lead is to convert, along with specific reasons.

Lead Notes:
{notes}

Consider the following factors:
1. Urgency signals (words like "urgent", "asap", "immediately", timeline mentions)
2. Buyer intent (serious vs casual, ready to buy vs just browsing)
3. Financial readiness (budget flexibility, loan approval, cash buyer)
4. Engagement level (scheduled visits, confirmation, follow-ups)
5. Red flags (not picking calls, unrealistic expectations, wrong contact)

Respond in this exact JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasons": ["reason1", "reason2", "reason3"],
    "urgency_level": "<high/medium/low>",
    "buyer_intent": "<strong/moderate/weak>",
    "red_flags": ["flag1", "flag2"] or []
}}
"""

CALL_QUALITY_ANALYSIS_PROMPT = """You are a call quality analyst for a real estate company. Analyze the following sales call transcript and evaluate the agent's performance.

Call Transcript:
{transcript}

Evaluate these dimensions on a scale of 0.0 to 1.0:

1. **Rapport Building (rapport_building)**: Did the agent greet properly, show empathy, personalize the conversation?
2. **Need Discovery (need_discovery)**: Did the agent ask relevant questions to understand customer requirements?
3. **Closing Attempt (closing_attempt)**: Did the agent attempt to close with clear next steps, commitment, or booking?
4. **Compliance Risk (compliance_risk)**: Any false promises, pressure tactics, or unprofessional behavior? (Lower is better)

Also provide:
- A brief summary of the call (2-3 sentences)
- Key points discussed
- Recommended next actions (if deal not closed)

Respond in this exact JSON format:
{{
    "rapport_building": <float 0.0-1.0>,
    "need_discovery": <float 0.0-1.0>,
    "closing_attempt": <float 0.0-1.0>,
    "compliance_risk": <float 0.0-1.0>,
    "summary": "<brief summary>",
    "key_points": ["point1", "point2", "point3"],
    "next_actions": ["action1", "action2"]
}}
"""

CALL_EVALUATION_SYSTEM_PROMPT = """You are an expert call quality analyst for a real estate company. Your job is to evaluate sales call transcripts and provide structured feedback on agent performance.

Be objective and fair in your assessment. Consider:
- The agent's communication skills
- How well they understood customer needs
- Their ability to address objections
- Whether they moved the conversation toward a positive outcome
- Any compliance or ethical concerns

Always respond with valid JSON matching the requested format."""
