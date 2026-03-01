# Intelligent Data Extraction & Summarisation on GCP

A prototype system for extracting entities, generating abstractive summaries, and orchestrating NLP capabilities through a ReAct-style agent on Google Cloud Platform.

## What This Does

Takes heterogeneous unstructured text (product reviews, support tickets, news articles), extracts structured information, summarises content, and answers complex analytical questions through an agentic workflow. The architecture follows a dual-backend pattern: local tools (SpaCy, ROUGE) for development baselines, GCP-native services (Gemini, BigQuery, Firestore) for production.

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────┐
│  AGENT (Gemini reasoning loop)   │
│                                  │
│  THINK → What do I need?         │
│  ACT   → Call a tool             │
│  OBSERVE → Process result        │
│  ...repeat until confident...    │
│  ANSWER → Synthesise findings    │
│                                  │
│  CRITIC → Validate answer        │
│  (completeness, grounding,       │
│   coherence)                     │
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
│  Session state persists across queries   │
└──────────────────────────────────────────┘
```

## Prerequisites

- **Python 3.10+**
- **A GCP project** with billing enabled
- **GCP APIs enabled:** Generative Language API, BigQuery, Cloud Storage, Firestore (optional)

## Getting Started

```bash
cd nlp_parsing_gcp

# Set up virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# SpaCy model for baseline comparisons
python -m spacy download en_core_web_sm

# Set your API key
export GOOGLE_API_KEY="your-api-key"
```

## Data Sources

Three deliberately heterogeneous sources to stress-test extraction:

1. **Amazon Product Reviews** (~74k docs): short, opinionated, sentiment-heavy. Place CSVs in `data/raw/reviews/archive/`.
2. **Customer Support Tickets** (~29.8k docs): template-heavy, noisy. Place CSV in `data/raw/support_tickets/`.
3. **CNN/DailyMail News** (~11.5k test split): long-form with gold-standard human summaries. Downloads automatically from HuggingFace.

## How to Run

### 1. Explore the data
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Run extraction & evaluate quality
```bash
jupyter notebook notebooks/02_extraction_and_eval.ipynb
```

### 3. Agent demo with critic validation
```bash
jupyter notebook notebooks/03_agent_demo.ipynb
```

### 4. Run tests (57 tests, all mocked, no API key needed)
```bash
python -m pytest tests/ -v
```

### 5. Fire up the agent programmatically
```python
from src.agent import CustomerInsightAgent

agent = CustomerInsightAgent(
    api_key="your-api-key",
    documents_df=your_dataframe,
    enable_critic=True,
)
response = agent.query("What are the top complaints in support tickets?")
print(response.answer)
```

## Project Layout

```
nlp_parsing_gcp/
├── src/
│   ├── data/
│   │   ├── loader.py            # Unified data loading across all sources
│   │   └── preprocessing.py     # Text cleanup
│   ├── extraction/
│   │   ├── gcp_nlp.py           # GCP Natural Language API wrapper
│   │   ├── spacy_baseline.py    # SpaCy NER for baseline comparison
│   │   ├── presidio_extract.py  # Microsoft Presidio PII-focused NER
│   │   └── vertex_extract.py    # Gemini structured extraction
│   ├── summarisation/
│   │   ├── vertex_summarise.py  # Gemini summarisation (single, multi, comparative)
│   │   └── evaluation.py        # ROUGE scoring
│   ├── evaluation/
│   │   ├── rouge_eval.py        # ROUGE metrics
│   │   ├── needle_haystack.py   # Needle-in-a-haystack extraction validation
│   │   └── bertscore_eval.py    # BERTScore semantic similarity
│   └── agent/
│       ├── agent.py             # ReAct orchestrator
│       ├── critic.py            # Actor-critic quality validation
│       ├── tools.py             # Tool wrappers (search, extract, sentiment, summarise)
│       └── memory.py            # Session persistence (local / Firestore)
├── tests/
│   ├── test_agent.py            # 27 unit tests (parsing, search, tool dispatch, loop)
│   ├── test_critic.py           # 7 unit tests (verdict, scoring, error handling)
│   ├── test_presidio.py         # 10 unit tests (PII extraction, batch, interface)
│   └── test_integration.py      # 13 integration tests (multi-step, actor-critic, E2E)
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA across all 3 datasets
│   ├── 02_extraction_and_eval.ipynb  # SpaCy vs Gemini benchmarks
│   └── 03_agent_demo.ipynb           # Live agent queries with critic validation
├── scripts/
│   ├── run_bertscore.py         # BERTScore evaluation runner
│   ├── run_needle_comparison.py # SpaCy vs Presidio vs Gemini extraction benchmark
│   ├── run_agent_e2e.py         # End-to-end agent validation
│   ├── prep_bigquery_jsonl.py   # BigQuery data loader
│   └── generate_pdf.py          # Architecture report PDF generator
├── config/
│   └── gcp_config.yaml          # Pinned model versions, project settings
├── architecture_report.pdf      # 2-page technical report
├── requirements.txt
└── README.md
```

## GCP Services

| Service | Role | Why |
|---------|------|-----|
| **Gemini 2.5 Flash** | Summarisation, extraction, agent reasoning | Single model; JSON mode output |
| **Cloud Storage** | Raw data landing zone | Standard ingest layer for pipelines |
| **BigQuery** | Document store (400 docs loaded) | SQL scales to PB; parameterised queries |
| **Firestore** | Agent session memory | Real-time, document-oriented |

## Evaluation Results

| Metric | Baseline | Gemini 2.5 Flash | Delta |
|--------|----------|-----------------|-------|
| ROUGE-1 F1 | 0.298 | **0.327** | +9.7% |
| BERTScore F1 | 0.8631 | **0.8780** | +1.7% |

**Needle-in-a-Haystack Entity Recall (3-way comparison):**

| Extractor | Avg Entity Recall | Strengths |
|-----------|------------------|-----------|
| SpaCy | 90.3% | General NER: ORG, DATE, GPE, MONEY |
| Presidio | 44.4% | PII-focused: PERSON 100%, DATE 100%, GPE 100% |
| Gemini 2.5 Flash | **90.3%** | Comprehensive: all types, precise boundaries |
