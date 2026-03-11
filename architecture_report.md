# NLP Extraction & Agentic Workflows on GCP -- Architecture Report

## System Overview

Dual-backend NLP pipeline: local tools (SpaCy, ROUGE) for development baselines, GCP-native services (Gemini, NL API) for production. Processes three heterogeneous text sources to stress-test extraction across different text characteristics. Includes a hierarchical multi-agent system with parallel execution, quality validation, and human escalation.

**Datasets:**
- Amazon Product Reviews (~74k rows) -- short, opinionated, entity-sparse
- Customer Support Tickets (~29.8k rows) -- template-heavy, action-oriented
- CNN/DailyMail News (~11.5k, streamed) -- long-form, entity-rich, gold-standard summaries

**GCP Services:**

| Service | Role |
|---------|------|
| Gemini 2.5 Flash (Generative Language API) | Summarization, extraction, sentiment, agent reasoning, planning, critic evaluation |
| Cloud Storage | Raw data landing (gs://nlp-extraction-data-333762334828/) |
| BigQuery | Document store & search (400 docs loaded, parameterized queries) |
| Firestore | Agent session memory |

## Results

### Summarization (25 CNN/DailyMail articles)

| Metric | Extractive Baseline | Gemini 2.5 Flash | Delta |
|--------|---------------------|------------------|-------|
| ROUGE-1 F1 | 0.298 | **0.327** | **+9.7%** |
| ROUGE-2 F1 | **0.127** | 0.100 | -21.5% |
| ROUGE-L F1 | **0.217** | 0.207 | -4.6% |

ROUGE-1 improvement reflects better information coverage. Lower ROUGE-2/L is expected -- abstractive summaries paraphrase rather than copy, which penalizes n-gram overlap metrics. BERTScore compensates for this.

Long documents (>8000 chars) use recursive map-reduce: chunk at sentence boundaries, summarize each chunk, then synthesize chunk summaries. Handles arbitrarily long inputs without hitting context limits.

### BERTScore (25 CNN/DailyMail articles)

| Metric | Extractive Baseline | Gemini 2.5 Flash | Delta |
|--------|---------------------|------------------|-------|
| BERTScore Precision | 0.8550 | **0.8703** | **+1.8%** |
| BERTScore Recall | 0.8717 | **0.8861** | **+1.6%** |
| BERTScore F1 | 0.8631 | **0.8780** | **+1.7%** |

Gemini outperforms on all semantic similarity dimensions, confirming quality despite lower n-gram overlap.

### Entity Extraction -- Needle-in-a-Haystack

Six synthetic needles with known entities injected at controlled positions (early/middle/deep) across a 230-document corpus. Three extractors benchmarked individually and as an ensemble.

| Metric | SpaCy | Presidio | Gemini 2.5 Flash | Ensemble |
|--------|-------|----------|------------------|----------|
| Detection Rate | 100% | 100% | **100%** | **100%** |
| Avg Entity Recall | **90.3%** | 44.4% | **90.3%** | **Best of all** |
| PERSON | 100% | **100%** | **100%** | **100%** |
| DATE | 100% | **100%** | **100%** | **100%** |
| GPE / LOC | 100% | **100%** | **100%** | **100%** |
| ORG | **100%** | 0% | **100%** | **100%** |
| MONEY | **100%** | 0% | **100%** | **100%** |
| PRODUCT | 50% | 50% | **50%** | **50%** |

Complementary strengths. SpaCy and Gemini match on aggregate recall but differ in boundary precision (SpaCy: "Evelyn Thorncastle"; Gemini: "Dr. Evelyn Thorncastle"). Presidio hits 100% on PII types but doesn't target general NER. The ensemble pipeline runs all three concurrently, deduplicates via fuzzy matching (substring + SequenceMatcher), normalizes entity types, and assigns confidence scores based on cross-extractor agreement.

## Agentic Workflow: Hierarchical Multi-Agent System

### Architecture

    User Query
        |
        v
    +--------------------------------------------------+
    |  PLANNER (Gemini)                                |
    |  Analyzes query complexity, decomposes into       |
    |  a DAG of 2-5 sub-tasks with dependency edges    |
    +--------------------------------------------------+
        |              |              |
        v              v              v       <-- parallel execution
    +----------+  +----------+  +----------+
    | Agent 1  |  | Agent 2  |  | Agent 3  |  <-- specialist ReAct agents
    | (search  |  | (search  |  | (depends |
    |  reviews)|  |  tickets)|  |  on 1+2) |
    +----------+  +----------+  +----------+
        |              |              |
        v              v              v
    +--------------------------------------------------+
    |  SYNTHESIZER                                     |
    |  Merges sub-agent results into cohesive answer   |
    +--------------------------------------------------+
        |
        v
    +--------------------------------------------------+
    |  CRITIC (actor-critic validation)                |
    |  Scores completeness, grounding, coherence (1-5) |
    |  Self-correction: feeds critique back to agent   |
    +--------------------------------------------------+
        |                              |
        v                              v
      ANSWER                    ESCALATE
      (to user)                 (if score < 2.5
                                 or sensitive topic)

### Layer 1: Hierarchical Planner

The planner receives the user query and produces an ExecutionPlan -- a DAG of PlanStep objects with dependency edges. Gemini generates the plan as structured JSON, specifying which steps can run in parallel and which depend on prior results.

Fallback: if planning fails (API error, malformed JSON), the planner falls back to a single-step plan that delegates everything to one ReAct agent.

### Layer 2: DAG Executor (Parallel Sub-Agents)

The DAG executor uses topological ordering to identify steps with no unresolved dependencies. Independent steps run concurrently via asyncio.gather, each spawning a specialist CustomerInsightAgent with its own ReAct reasoning loop.

Dependent steps receive the results of their prerequisites as context, enabling queries like: "Search reviews for battery complaints" (step 1) then "Analyze sentiment of those results" (step 2, depends on step 1).

### Layer 3: Specialist ReAct Agents

Each sub-agent runs the standard ReAct loop: THINK, ACT (call a tool), OBSERVE (process result), repeat until confident, then ANSWER. Multi-turn support: session memory (LocalMemory or Firestore) persists conversation history, enabling follow-up queries that reference prior exchanges.

**7 Tools:** SEARCH, EXTRACT_ENTITIES, EXTRACT_STRUCTURED, ANALYZE_SENTIMENT, SUMMARIZE, SUMMARIZE_MULTIPLE, COMPARE. Each wraps a GCP service. Backend-agnostic -- swap BigQuery for Elasticsearch without changing agent logic.

### Layer 4: Critic and Self-Correction

The critic evaluates the synthesized answer on three axes (each 1-5):

- **Completeness:** Does the answer address all parts of the query?
- **Grounding:** Are claims supported by the evidence gathered?
- **Coherence:** Is the answer well-structured and internally consistent?

Verdict: PASS (all >= 4), REVISE (any 2-3), or FAIL (any 1).

On REVISE or FAIL, the self-correction loop feeds the critique back into the agent for a second reasoning pass, rather than blindly returning the critic's revised answer. This gives the agent access to its full tool set and evidence context when improving the answer.

### Layer 5: Human-in-the-Loop Escalation

Two escalation triggers:

1. **Keyword detection** -- queries containing sensitive terms (legal, compliance, PII, GDPR, HIPAA, etc.) are flagged at planning time
2. **Critic score threshold** -- if the critic's overall score falls below 2.5/5 after self-correction, the response is marked as requiring human review

Escalated responses are still returned to the user but tagged with an escalation flag and reason. The SSE streaming API emits a dedicated "escalation" event so the frontend can surface an approval prompt.

### Token and Cost Tracking

Every Gemini API call automatically records token usage (input + output) and estimated cost via a thread-safe global accumulator. Costs are broken down by model (flash vs flash-lite). Exposed via GET /api/usage for monitoring. Useful for tracking cost per query across the planner, sub-agents, and critic.

## Streaming API

FastAPI application with Server-Sent Events for real-time visibility into agent reasoning.

**Single-agent endpoint** (POST /api/query/stream): streams THOUGHT, ACTION, OBSERVATION, and ANSWER events as the ReAct agent works through the query.

**Hierarchical planner endpoint** (POST /api/planner/stream): streams the full orchestration lifecycle:
- plan: the decomposed DAG of sub-tasks
- parallel_batch: which steps are executing concurrently
- step_start / step_complete: per-sub-agent progress
- synthesis: merging sub-results
- escalation: human review flag (if triggered)
- answer: final synthesized response

Built-in dark-theme frontend at the root URL for interactive demos.

## Evaluation Dashboard

Streamlit application with four tabs:

1. **Ensemble Extraction** -- run all three extractors on custom text, view merged results with confidence scores, entity-type-by-extractor heatmap
2. **Needle-in-a-Haystack** -- recall by extractor and entity type, position-sensitivity heatmap (early/middle/deep)
3. **Summarization Metrics** -- ROUGE and BERTScore distributions, extractive vs. abstractive comparison
4. **Document Explorer** -- browse loaded documents by source type, view metadata

## Deployment

Dockerfile (Python 3.12-slim) configured for Cloud Run. docker-compose.yml bundles the API (port 8080) and Streamlit dashboard (port 8501) as two services sharing the same image. Data directory is mounted read-only.

CI/CD via GitHub Actions: pytest across Python 3.11-3.13, ruff linting, and syntax checking on every push and pull request.

## Challenges & Trade-offs

1. **ROUGE vs. quality:** ROUGE penalizes paraphrasing, inflating extractive scores. Addressed with BERTScore. Production needs both metrics plus human eval.
2. **Rate limiting:** Gemini API limits required batching and backoff. Handled by exponential backoff with automatic fallback from flash to flash-lite.
3. **Streaming data:** CNN/DailyMail (1.3GB) exceeded disk. Solved with HuggingFace streaming.
4. **Planning accuracy:** LLM-generated plans can be suboptimal. Mitigated by capping step count, falling back to single-agent mode on failure, and validating results via the critic.
5. **Parallel cost:** Hierarchical execution uses more tokens than a single agent (planner + N sub-agents + synthesizer + critic). Token tracking makes this visible and controllable.

## Productionization

**Orchestration:** Vertex AI Pipelines for batch extraction/summarization. Cloud Functions + Pub/Sub for real-time ingest. BigQuery for petabyte-scale document storage. Vertex AI Agent Builder as production replacement for the custom ReAct loop.

**Security:** IAM service accounts with least-privilege. CMEK for sensitive corpora. VPC Service Controls for data exfiltration prevention. Escalation system ensures sensitive queries are flagged for human review.

**Monitoring:** Cloud Logging with structured metadata (doc ID, stage, latency). Cloud Monitoring dashboards for error rates, ROUGE drift, extraction recall, agent step counts. Dead-letter queues for failed extractions. Built-in token/cost tracking per API call.

**Cost:** Gemini Flash over Pro for routine work (10x cheaper). Automatic fallback to flash-lite when rate limited. BigQuery on-demand for prototype, reserved slots for prod. Memorystore caching for repeated queries. Budget alerts at 50/80/100%. Per-query cost visibility via the usage tracking API.

**CI/CD:** GitHub Actions: lint (ruff) -> unit tests (pytest, 3.11-3.13) -> syntax check -> deploy. Pinned model versions in config. Dockerfile for Cloud Run. 161 mocked tests cover agent logic, planning, DAG execution, escalation, streaming API, and all extraction/summarization pipelines.
