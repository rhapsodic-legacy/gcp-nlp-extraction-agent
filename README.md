# NLP Extraction & Agentic Workflows — GCP

Entity extraction, abstractive summarization, and multi-step analytical queries via a ReAct agent. All backed by GCP services.

## What It Does

Processes heterogeneous unstructured text (product reviews, support tickets, news articles):

1. **Extracts** structured entities — Gemini, SpaCy, Presidio
2. **Summarizes** documents — Gemini 2.5 Flash, benchmarked via ROUGE + BERTScore
3. **Orchestrates** multi-step queries — ReAct agent with actor-critic validation

Dual-backend: local tools (SpaCy, ROUGE) for baselines, GCP-native (Gemini, BigQuery, Firestore) for production.

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────┐
│  AGENT (Gemini reasoning loop)   │
│                                  │
│  THINK → ACT → OBSERVE → repeat │
│  ANSWER → CRITIC validates       │
└──────────────────────────────────┘
    │           │           │           │
    ▼           ▼           ▼           ▼
 SEARCH    EXTRACT     SENTIMENT   SUMMARIZE
(BigQuery/  (Gemini/    (Gemini)   (Gemini)
 DataFrame)  SpaCy)
    │           │           │           │
    ▼           ▼           ▼           ▼
┌──────────────────────────────────────────┐
│  MEMORY (LocalMemory / Firestore)        │
└──────────────────────────────────────────┘
```

## Results

### Summarization

| Metric | Extractive Baseline | Gemini 2.5 Flash | Delta |
|--------|---------------------|------------------|-------|
| ROUGE-1 F1 | 0.298 | **0.327** | +9.7% |
| BERTScore F1 | 0.8631 | **0.8780** | +1.7% |

### Extraction — Needle-in-a-Haystack

| Extractor | Avg Recall | Strengths |
|-----------|------------|-----------|
| SpaCy | 90.3% | General NER: ORG, DATE, GPE, MONEY |
| Presidio | 44.4% | PII: PERSON 100%, DATE 100%, GPE 100% |
| Gemini | **90.3%** | All types, precise entity boundaries |

## GCP Services

| Service | Role |
|---------|------|
| Gemini 2.5 Flash | Summarization, extraction, sentiment, agent reasoning |
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
1. `notebooks/01_data_exploration.ipynb` — EDA across all 3 datasets
2. `notebooks/02_extraction_and_eval.ipynb` — extraction benchmarks
3. `notebooks/03_agent_demo.ipynb` — live agent queries

**Tests** (57 tests, mocked, no API key needed):
```bash
python -m pytest tests/ -v
```

**Programmatic usage:**
```python
from src.agent import CustomerInsightAgent

agent = CustomerInsightAgent(
    api_key="your-key",
    documents_df=df,
    enable_critic=True,
)
response = agent.query("Top complaints in support tickets?")
print(response.answer)
```

## Project Layout

```
src/
  data/           Loading + preprocessing
  extraction/     SpaCy, Presidio, Gemini NER
  summarization/  Extractive baseline + Gemini abstractive
  evaluation/     ROUGE, BERTScore, needle-in-a-haystack
  agent/          ReAct orchestrator, actor-critic, tools, memory
tests/            57 unit + integration tests
notebooks/        EDA, benchmarks, agent demo
scripts/          Eval runners, data loaders, PDF generator
config/           Model versions, project settings
```

## Data Sources

| Dataset | Size | Characteristics |
|---------|------|----------------|
| Amazon Product Reviews | ~74k | Short, opinionated, entity-sparse |
| Customer Support Tickets | ~29.8k | Template-heavy, action-oriented |
| CNN/DailyMail News | ~11.5k | Long-form, entity-rich, gold-standard summaries |

See [DEV_NOTES.md](DEV_NOTES.md) for setup details and data placement.
