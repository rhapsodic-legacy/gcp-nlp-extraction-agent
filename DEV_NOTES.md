# Dev Notes

## Setup

```bash
cd nlp_parsing_gcp
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

API key goes in `.env`:
```
GOOGLE_API_KEY=your-key
```

Or export directly:
```bash
export GOOGLE_API_KEY="your-key"
```

## Data Placement

```
data/raw/
├── reviews/archive/     # Amazon CSVs (Datafiniti format)
├── support_tickets/     # customer_support_tickets.csv
└── reddit/              # reddit_wsb.csv (optional)
```

CNN/DailyMail streams from HuggingFace automatically — no download needed.

## Running

Notebooks in order:
1. `01_data_exploration.ipynb` — EDA
2. `02_extraction_and_eval.ipynb` — extraction + ROUGE benchmarks
3. `03_agent_demo.ipynb` — agent demo (needs API key)

Tests (all mocked):
```bash
python -m pytest tests/ -v
```

## Project Structure

```
src/
├── data/
│   ├── loader.py              # Unified loading: reviews, tickets, news, reddit
│   └── preprocessing.py       # Text cleanup
├── extraction/
│   ├── gcp_nlp.py             # GCP NL API wrapper
│   ├── spacy_baseline.py      # SpaCy NER baseline
│   ├── presidio_extract.py    # Presidio PII detection
│   └── vertex_extract.py      # Gemini structured extraction
├── summarization/
│   ├── vertex_summarize.py    # Gemini (single, multi, comparative)
│   └── evaluation.py          # ROUGE scoring
├── evaluation/
│   ├── rouge_eval.py          # ROUGE metrics
│   ├── needle_haystack.py     # Needle-in-a-haystack validation
│   └── bertscore_eval.py      # BERTScore semantic similarity
└── agent/
    ├── agent.py               # ReAct orchestrator
    ├── critic.py              # Actor-critic validation
    ├── tools.py               # Tool wrappers
    └── memory.py              # LocalMemory / Firestore

tests/
├── test_agent.py              # 27 tests
├── test_critic.py             # 7 tests
├── test_presidio.py           # 10 tests
└── test_integration.py        # 13 tests

scripts/
├── run_bertscore.py           # BERTScore eval runner
├── run_needle_comparison.py   # 3-way extraction benchmark
├── run_agent_e2e.py           # End-to-end agent validation
├── prep_bigquery_jsonl.py     # BigQuery data prep
├── load_bigquery.py           # BigQuery loader
└── generate_pdf.py            # Architecture report PDF
```

## Notes

- GCP clients lazy-load — local baselines work without cloud packages installed
- CNN/DailyMail uses `streaming=True` to avoid downloading 1.3GB
- All tests are mocked — no API key needed to run the test suite
- Config in `config/gcp_config.yaml` — model versions pinned, not "latest"
