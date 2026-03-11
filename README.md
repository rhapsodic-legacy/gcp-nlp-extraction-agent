# NLP Extraction & Agentic Workflows -- GCP

Entity extraction, abstractive summarization, and hierarchical multi-agent analytical queries. All backed by GCP services.

## What It Does

Processes heterogeneous unstructured text (product reviews, support tickets, news articles):

1. **Extracts** structured entities -- ensemble pipeline: SpaCy + Presidio + Gemini with fuzzy deduplication and confidence scoring
2. **Summarizes** documents -- Gemini 2.5 Flash with recursive map-reduce for long documents, benchmarked via ROUGE + BERTScore
3. **Orchestrates** multi-step queries -- hierarchical planner decomposes queries into a DAG of sub-tasks, executed in parallel by specialist ReAct agents, with actor-critic validation and human-in-the-loop escalation

Dual-backend: local tools (SpaCy, ROUGE) for baselines, GCP-native (Gemini, BigQuery, Firestore) for production.

## Architecture

```
User Query
    |
    v
+-----------------------------------------+
|  PLANNER (Gemini)                       |
|  Decomposes query into DAG of sub-tasks |
+-----------------------------------------+
    |           |           |
    v           v           v        <-- parallel execution
+----------+----------+----------+
| Agent 1  | Agent 2  | Agent 3  |   <-- specialist ReAct agents
| (ReAct)  | (ReAct)  | (ReAct)  |
+----------+----------+----------+
    |           |           |
    v           v           v
+-----------------------------------------+
|  SYNTHESIZER                            |
|  Merges sub-results into final answer   |
+-----------------------------------------+
    |
    v
+-----------------------------------------+
|  CRITIC (actor-critic validation)       |
|  Scores: completeness, grounding,       |
|  coherence (1-5). Self-correction loop  |
|  feeds critique back to agent.          |
+-----------------------------------------+
    |                          |
    v                          v
  ANSWER              ESCALATE (if low
                      confidence or
                      sensitive topic)

Each specialist agent has access to:
  SEARCH | EXTRACT | SENTIMENT | SUMMARIZE
  (BigQuery/   (Gemini/   (Gemini)  (Gemini +
   DataFrame)   SpaCy)              map-reduce)

Session memory: LocalMemory (dev) / Firestore (prod)
Token/cost tracking: per-call accumulator with model-level breakdown
```

## Results

### Summarization

| Metric | Extractive Baseline | Gemini 2.5 Flash | Delta |
|--------|---------------------|------------------|-------|
| ROUGE-1 F1 | 0.298 | **0.327** | +9.7% |
| BERTScore F1 | 0.8631 | **0.8780** | +1.7% |

### Extraction -- Needle-in-a-Haystack

| Extractor | Avg Recall | Strengths |
|-----------|------------|-----------|
| SpaCy | 90.3% | General NER: ORG, DATE, GPE, MONEY |
| Presidio | 44.4% | PII: PERSON 100%, DATE 100%, GPE 100% |
| Gemini | **90.3%** | All types, precise entity boundaries |
| **Ensemble** | **Best of all three** | Fuzzy dedup, confidence scoring, type normalization |

## GCP Services

| Service | Role |
|---------|------|
| Gemini 2.5 Flash | Summarization, extraction, sentiment, agent reasoning, planning |
| Cloud Storage | Raw data landing |
| BigQuery | Document store + search |
| Firestore | Agent session memory |

## Quick Start

```bash
cd nlp_parsing_gcp
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export GOOGLE_API_KEY="your-key"
```

**Notebooks** (run in order):
1. `notebooks/01_data_exploration.ipynb` -- EDA across all 3 datasets
2. `notebooks/02_extraction_and_eval.ipynb` -- extraction benchmarks
3. `notebooks/03_agent_demo.ipynb` -- live agent queries

**Streaming API:**
```bash
uvicorn src.api.app:app --reload --port 8000
# Open http://localhost:8000 for the built-in frontend
# API docs at http://localhost:8000/docs
```

**Evaluation Dashboard:**
```bash
streamlit run src/dashboard/app.py
```

**Docker (both services):**
```bash
docker-compose up --build
# API on :8080, Dashboard on :8501
```

**Tests** (161 tests, all mocked, no API key needed):
```bash
python -m pytest tests/ -v
```

**Programmatic usage:**
```python
# Single-agent ReAct loop
from src.agent import CustomerInsightAgent

agent = CustomerInsightAgent(
    api_key="your-key",
    documents_df=df,
    enable_critic=True,
)
response = agent.query("Top complaints in support tickets?")
print(response.answer)

# Hierarchical planner with parallel execution
from src.agent import HierarchicalPlanner

planner = HierarchicalPlanner(
    api_key="your-key",
    documents_df=df,
    enable_escalation=True,
)
response = planner.query("Compare sentiment across reviews and tickets, flag escalations")
print(response.answer)
print(f"Escalated: {response.escalated}")
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/query/stream` | Stream single-agent ReAct reasoning via SSE |
| POST | `/api/planner/stream` | Stream hierarchical planner execution via SSE |
| POST | `/api/load-data` | Load documents for the search tool |
| GET | `/api/health` | Health check (API key, data status) |
| GET | `/api/usage` | Token usage and estimated cost |
| POST | `/api/usage/reset` | Reset usage tracker |
| GET | `/` | Built-in frontend |

## Project Layout

```
src/
  api/              Streaming FastAPI + SSE, built-in frontend
  api_utils.py      Gemini retry/fallback wrapper, token/cost tracking
  agent/
    planner.py      Hierarchical planner, DAG executor, escalation
    agent.py        ReAct orchestrator with multi-turn + self-correction
    critic.py       Actor-critic quality gate (completeness/grounding/coherence)
    tools.py        Tool wrappers (search, extract, sentiment, summarize)
    memory.py       LocalMemory / Firestore session backends
  dashboard/        Streamlit evaluation dashboard (4 tabs)
  data/             Loading + preprocessing
  extraction/
    ensemble.py     Ensemble pipeline: SpaCy + Presidio + Gemini
    spacy_baseline.py, presidio_extract.py, vertex_extract.py
  summarization/    Gemini abstractive + recursive map-reduce for long docs
  evaluation/       ROUGE, BERTScore, needle-in-a-haystack
tests/              161 unit + integration tests (12 test files)
notebooks/          EDA, benchmarks, agent demo
scripts/            Eval runners, data loaders, PDF generator
config/             Model versions, project settings
Dockerfile          Python 3.12-slim, Cloud Run ready
docker-compose.yml  API + Dashboard services
.github/workflows/  CI: pytest (3.11-3.13), ruff lint, syntax check
```

## Data Sources

| Dataset | Size | Characteristics |
|---------|------|----------------|
| Amazon Product Reviews | ~74k | Short, opinionated, entity-sparse |
| Customer Support Tickets | ~29.8k | Template-heavy, action-oriented |
| CNN/DailyMail News | ~11.5k | Long-form, entity-rich, gold-standard summaries |

See [DEV_NOTES.md](DEV_NOTES.md) for setup details and data placement.
