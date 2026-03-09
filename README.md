# Intelligent Data Extraction & Summarisation with Agentic Workflows on GCP        

Extract entities, generate abstractive summaries, and orchestrate NLP capabilities through a ReAct-style agent, all backed by Google Cloud Platform services.

## Overview

This system takes heterogeneous unstructured text (product reviews, support tickets, news articles) and:

1. **Extracts** structured entities and information using Gemini, SpaCy, and Microsoft Presidio
2. **Summarises** documents using Gemini 2.5 Flash, benchmarked against extractive baselines via ROUGE and BERTScore
3. **Orchestrates** multi-step analytical queries through a ReAct agent with actor-critic validation

The architecture follows a dual-backend pattern: local tools (SpaCy, ROUGE) for development baselines; GCP-native services (Gemini, BigQuery, Firestore) for production.

## Architecture

```
User Query
    |
    v
+----------------------------------+
|  AGENT (Gemini reasoning loop)   |
|                                  |
|  THINK -> What do I need?        |
|  ACT   -> Call a tool            |
|  OBSERVE -> Process result       |
|  ...repeat until confident...    |
|  ANSWER -> Synthesise findings   |
|                                  |
|  CRITIC -> Validate answer       |
|  (completeness, grounding,       |
|   coherence)                     |
+----------------------------------+
    |           |           |           |
    v           v           v           v
 SEARCH    EXTRACT     SENTIMENT   SUMMARIZE
(BigQuery/  (Gemini/    (Gemini)   (Gemini)
 DataFrame)  SpaCy)
    |           |           |           |
    v           v           v           v
+------------------------------------------+
|  MEMORY (LocalMemory / Firestore)        |
|  Session state persists across queries   |
+------------------------------------------+
```

## Results

### Summarisation

| Metric | Extractive Baseline | Gemini 2.5 Flash | Delta |
|--------|-------------------|-----------------|-------|
| ROUGE-1 F1 | 0.298 | **0.327** | +9.7% |
| BERTScore F1 | 0.8631 | **0.8780** | +1.7% |

### Entity Extraction (Needle-in-a-Haystack, 3-way comparison)

| Extractor | Avg Recall | PII Types | General NER |
|-----------|-----------|-----------|-------------|
| SpaCy | 90.3% | Partial | ORG, DATE, GPE, MONEY: 100% |
| Presidio | 44.4% | PERSON, DATE, GPE: 100% | ORG, MONEY: 0% |
| Gemini 2.5 Flash | **90.3%** | 100% | 100% (precise boundaries) |

## GCP Services

| Service | Role |
|---------|------|
| **Gemini 2.5 Flash** | Summarisation, extraction, sentiment, agent reasoning |
| **Cloud Storage** | Raw data landing zone |
| **BigQuery** | Production document store and search |
| **Firestore** | Agent session memory |

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export GOOGLE_API_KEY="your-key"
```

**Notebooks** (run in order):
1. `notebooks/01_data_exploration.ipynb`: EDA across all 3 datasets
2. `notebooks/02_extraction_and_eval.ipynb`: SpaCy vs Gemini extraction benchmarks
3. `notebooks/03_agent_demo.ipynb`: live agent queries with critic validation

**Tests** (57 tests, all mocked, no API key needed):
```bash
python -m pytest tests/ -v
```

## Project Structure

```
src/
  data/           Data loading and preprocessing
  extraction/     SpaCy, Presidio, and Gemini NER
  summarisation/  Extractive baseline and Gemini abstractive
  evaluation/     ROUGE, BERTScore, needle-in-a-haystack
  agent/          ReAct orchestrator, actor-critic, tools, memory
tests/            57 unit + integration tests
notebooks/        3 Jupyter notebooks (EDA, benchmarks, agent demo)
scripts/          Evaluation runners, data loaders, PDF generator
config/           Pinned model versions and project settings
```

## Data Sources

| Dataset | Size | Characteristics |
|---------|------|----------------|
| Amazon Product Reviews | ~74k docs | Short, opinionated, entity-sparse |
| Customer Support Tickets | ~29.8k docs | Template-heavy, action-oriented |
| CNN/DailyMail News | ~11.5k docs | Long-form, entity-rich, gold-standard summaries |

See [DEV_NOTES.md](DEV_NOTES.md) for detailed setup instructions, full project layout, and data placement.
