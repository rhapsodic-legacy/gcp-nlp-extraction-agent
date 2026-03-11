# Dev Notes

## Setup

    cd nlp_parsing_gcp
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

API key goes in .env:

    GOOGLE_API_KEY=your-key

Or export directly:

    export GOOGLE_API_KEY="your-key"

## Data Placement

    data/raw/
    +-- reviews/archive/     # Amazon CSVs (Datafiniti format)
    +-- support_tickets/     # customer_support_tickets.csv
    +-- reddit/              # reddit_wsb.csv (optional)

CNN/DailyMail streams from HuggingFace automatically -- no download needed.

## Running

**Notebooks** (in order):
1. 01_data_exploration.ipynb -- EDA
2. 02_extraction_and_eval.ipynb -- extraction + ROUGE benchmarks
3. 03_agent_demo.ipynb -- agent demo (needs API key)

**Streaming API:**

    uvicorn src.api.app:app --reload --port 8000

Visit localhost:8000 for the built-in frontend, or localhost:8000/docs for the OpenAPI spec.

**Evaluation Dashboard:**

    streamlit run src/dashboard/app.py

**Docker (API + Dashboard together):**

    docker-compose up --build

API on port 8080, Dashboard on port 8501.

**Tests** (all mocked, no API key needed):

    python -m pytest tests/ -v

## Project Structure

    src/
    +-- api/
    |   +-- app.py                 # FastAPI + SSE streaming (single-agent + planner)
    |   +-- static/index.html      # Built-in dark-theme frontend
    +-- api_utils.py               # Gemini retry/fallback, token/cost tracking
    +-- agent/
    |   +-- planner.py             # Hierarchical planner, DAG executor, escalation
    |   +-- agent.py               # ReAct orchestrator, multi-turn, self-correction
    |   +-- critic.py              # Actor-critic validation (completeness/grounding/coherence)
    |   +-- tools.py               # Tool wrappers (search, extract, sentiment, summarize)
    |   +-- memory.py              # LocalMemory / FirestoreMemory
    +-- dashboard/
    |   +-- app.py                 # Streamlit eval dashboard (4 tabs)
    +-- data/
    |   +-- loader.py              # Unified loading: reviews, tickets, news, reddit
    |   +-- preprocessing.py       # Text cleanup
    +-- extraction/
    |   +-- ensemble.py            # Ensemble: SpaCy + Presidio + Gemini, fuzzy dedup
    |   +-- gcp_nlp.py             # GCP NL API wrapper
    |   +-- spacy_baseline.py      # SpaCy NER baseline
    |   +-- presidio_extract.py    # Presidio PII detection
    |   +-- vertex_extract.py      # Gemini structured extraction
    +-- summarization/
    |   +-- vertex_summarize.py    # Gemini (single, multi, comparative, map-reduce)
    |   +-- evaluation.py          # ROUGE scoring
    +-- evaluation/
        +-- rouge_eval.py          # ROUGE metrics
        +-- needle_haystack.py     # Needle-in-a-haystack validation
        +-- bertscore_eval.py      # BERTScore semantic similarity

    tests/
    +-- test_agent.py              # 27 tests -- ReAct loop, parsing, tool dispatch
    +-- test_api.py                # 20 tests -- SSE format, endpoints, response parsing
    +-- test_critic.py             # 7 tests  -- critic scoring and verdicts
    +-- test_critic_selfcorrect.py # 8 tests  -- self-correction loop, fallback
    +-- test_dashboard.py          # 7 tests  -- dashboard data loading, heatmaps
    +-- test_ensemble.py           # 18 tests -- fuzzy matching, dedup, confidence
    +-- test_integration.py        # 13 tests -- end-to-end agent flows
    +-- test_multiturn.py          # 11 tests -- session history, multi-turn context
    +-- test_planner.py            # 15 tests -- plan creation, DAG execution, escalation
    +-- test_presidio.py           # 10 tests -- PII detection patterns
    +-- test_summarize_long.py     # 16 tests -- chunking, map-reduce, routing
    +-- test_usage_tracking.py     # 9 tests  -- token tracking, cost estimation
    (161 tests total)

    scripts/
    +-- run_bertscore.py           # BERTScore eval runner
    +-- run_needle_comparison.py   # 3-way extraction benchmark
    +-- run_agent_e2e.py           # End-to-end agent validation
    +-- prep_bigquery_jsonl.py     # BigQuery data prep
    +-- load_bigquery.py           # BigQuery loader
    +-- generate_pdf.py            # Architecture report PDF

    Dockerfile                     # Python 3.12-slim, Cloud Run ready
    docker-compose.yml             # API (:8080) + Dashboard (:8501)
    .github/workflows/ci.yml      # pytest (3.11-3.13), ruff lint, syntax check

## API Endpoints

    POST /api/query/stream      -- Stream single-agent ReAct reasoning (SSE)
    POST /api/planner/stream    -- Stream hierarchical planner execution (SSE)
    POST /api/load-data         -- Load documents for search tool
    GET  /api/health            -- Health check
    GET  /api/usage             -- Token usage and estimated cost
    POST /api/usage/reset       -- Reset usage tracker

## Notes

- GCP clients lazy-load -- local baselines work without cloud packages installed
- CNN/DailyMail uses streaming=True to avoid downloading 1.3GB
- All 161 tests are mocked -- no API key needed to run the test suite
- Config in config/gcp_config.yaml -- model versions pinned, not "latest"
- Gemini rate limiting handled by exponential backoff with fallback to flash-lite
- Token/cost tracking is automatic -- every Gemini call records usage globally
- Hierarchical planner falls back to single-agent mode if planning fails
- Escalation triggers: sensitive keywords (legal, PII, GDPR) or critic score below 2.5/5
