# Fixit - GenAI Backend & Lead Prioritization System

A production-ready real estate sales analytics platform that helps teams prioritize leads and evaluate call quality using a hybrid approach of deterministic rules and Large Language Models.

##  Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) (for LLM features)
- Docker (optional)

### Local Setup

```bash
# Clone the repository
cd "fixit project"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start Ollama (in separate terminal)
ollama serve
ollama pull llama3.2:3b

# Run the application
uvicorn src.main:app --reload --port 8000
```

### Docker Setup

```bash
# Build and run with Docker
docker build -t fixit-genai-assignment .
docker run -p 8000:8000 fixit-genai-assignment

# Or use Docker Compose (includes Ollama)
docker-compose up --build
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Lead Priority Endpoint

**POST** `/api/v1/lead-priority`

Prioritizes leads using hybrid scoring (deterministic rules + LLM analysis).

```bash
curl -X POST http://localhost:8000/api/v1/lead-priority \
  -H "Content-Type: application/json" \
  -d '{
    "leads": [
      {
        "lead_id": "LEAD001",
        "source": "referral",
        "budget": 15000000,
        "city": "Mumbai",
        "property_type": "3BHK",
        "last_activity_minutes_ago": 30,
        "past_interactions": 5,
        "notes": "Very interested, wants to visit this weekend!",
        "status": "contacted"
      }
    ],
    "max_results": 10
  }'
```

**Response:**
```json
{
  "ranked_leads": [
    {
      "lead_id": "LEAD001",
      "priority_score": 0.815,
      "priority_bucket": "hot",
      "reasons": [
        "Recent activity (< 1 hour)",
        "Highly engaged (5 interactions)",
        "High-quality source (referral)",
        "Good budget (₹1.5Cr)",
        "Urgency signals detected: interested, weekend"
      ],
      "recency_score": 0.85,
      "engagement_score": 0.6,
      "source_score": 1.0,
      "budget_score": 0.7,
      "notes_score": 0.85
    }
  ],
  "total_processed": 1,
  "model_metadata": {...}
}
```

### Call Evaluation Endpoint

**POST** `/api/v1/call-eval`

Evaluates call quality using LangGraph workflow.

```bash
curl -X POST http://localhost:8000/api/v1/call-eval \
  -H "Content-Type: application/json" \
  -d '{
    "call_id": "CALL001",
    "transcript": "Agent: Good morning! Am I speaking with Mr. Sharma?\nCustomer: Yes, speaking.\nAgent: This is Rahul from Premium Properties...",
    "duration_seconds": 180
  }'
```

**Response:**
```json
{
  "call_id": "CALL001",
  "quality_score": 0.78,
  "labels": {
    "rapport_building": 0.85,
    "need_discovery": 0.72,
    "closing_attempt": 0.80,
    "compliance_risk": 0.1
  },
  "summary": "Agent demonstrated excellent rapport building and professional handling...",
  "next_actions": ["Schedule site visit", "Send property brochure"],
  "model_metadata": {
    "model_name": "llama3.2:3b",
    "latency_ms": 450
  },
  "is_good_call": true
}
```





## Scoring Methodology

### Lead Priority Scoring

Weighted combination of 5 dimensions:

| Dimension | Weight | Calculation |
|-----------|--------|-------------|
| Recency | 25% | Time since last activity (< 30min = 1.0) |
| Engagement | 20% | Past interactions + status modifier |
| Source | 15% | referral=1.0, walk-in=0.9, portal=0.75, social=0.4 |
| Budget | 20% | Scaled by amount (>5Cr = 1.0, <50L = 0.4) |
| Notes | 20% | LLM/keyword analysis for signals |

### Call Quality Scoring

LangGraph workflow with 4 dimensions:

| Dimension | Weight | Good Signs |
|-----------|--------|------------|
| Rapport Building | 25% | Greeting, empathy, personalization |
| Need Discovery | 30% | Questions asked, requirements captured |
| Closing Attempt | 30% | Next steps, commitment sought |
| Compliance Risk | 15% | (inverted) No false promises, no pressure |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_lead_scoring.py -v
```

## Evaluation Harness

### Run Evaluations

```bash
# Lead prioritization evaluation
python evaluation/evaluate_leads.py

# Call quality evaluation
python evaluation/evaluate_calls.py
```

### Evaluation Results

See generated reports:
- `evaluation/evaluation_leads.md`
- `evaluation/evaluation_calls.md`

## Project Structure

```
fixit project/
├── src/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── api/
│   │   ├── routes/
│   │   │   ├── lead_priority.py
│   │   │   └── call_eval.py
│   │   └── schemas/
│   │       ├── lead.py
│   │       └── call.py
│   └── services/
│       ├── lead_scorer.py   # Hybrid scoring logic
│       └── call_analyzer.py # LangGraph workflow
├── models/
│   ├── llm_client.py        # LangChain + Ollama
│   └── prompts.py           # Prompt templates
├── data/
│   ├── leads.csv            # 160 synthetic leads
│   ├── calls.json           # 30 call transcripts
│   ├── leads_ground_truth.json
│   └── calls_ground_truth.json
├── tests/
│   ├── test_lead_scoring.py
│   └── test_call_eval_api.py
├── evaluation/
│   ├── evaluate_leads.py
│   └── evaluate_calls.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Trade-offs Made

| Decision | Trade-off | Rationale |
|----------|-----------|-----------|
| Ollama local LLM | Requires local setup | Privacy, no cost, no API dependency |
| Hybrid scoring | More complex | Deterministic base + LLM enhancement = reliable yet nuanced |
| LangGraph for calls | Added complexity | Better state management, debugging, observability |
| Pydantic v2 | Stricter validation | Better type safety, performance |
| No database | Stateless | Simplified deployment, evaluation focus |
| Heuristic fallback | Less accurate | Works when Ollama unavailable |

## Configuration

Environment variables (`.env`):

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
LLM_TIMEOUT=60
HOT_THRESHOLD=0.7
WARM_THRESHOLD=0.4
GOOD_CALL_THRESHOLD=0.6
```


## Assignment Compliance

| Requirement | Status | Location |
|-------------|--------|----------|
| leads.csv (≥150) | ✅ | `data/leads.csv` (160 leads) |
| calls.json (≥25) | ✅ | `data/calls.json` (30 transcripts) |
| Lead Priority API | ✅ | `POST /api/v1/lead-priority` |
| Call Eval API | ✅ | `POST /api/v1/call-eval` |
| LLM Integration | ✅ | LangChain + Ollama |
| LangGraph | ✅ | `src/services/call_analyzer.py` |
| Evaluation (20 leads) | ✅ | `evaluation/evaluate_leads.py` |
| Evaluation (10 calls) | ✅ | `evaluation/evaluate_calls.py` |
| Unit Tests | ✅ | `tests/` |
| Dockerfile | ✅ | `Dockerfile` |
| README | ✅ | This file |

---

