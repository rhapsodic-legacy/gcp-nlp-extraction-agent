# NLP Extraction & Agentic Workflows on GCP — Architecture Report

## System Overview

Dual-backend NLP pipeline: local tools (SpaCy, ROUGE) for development baselines, GCP-native services (Gemini, NL API) for production. Processes three heterogeneous text sources to stress-test extraction across different text characteristics.

**Datasets:**
- Amazon Product Reviews (~74k rows) — short, opinionated, entity-sparse
- Customer Support Tickets (~29.8k rows) — template-heavy, action-oriented
- CNN/DailyMail News (~11.5k, streamed) — long-form, entity-rich, gold-standard summaries

**GCP Services:**
| Service | Role |
|---------|------|
| Gemini 2.5 Flash (Generative Language API) | Summarization, structured extraction, entity recognition, agent reasoning |
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

ROUGE-1 improvement reflects better information coverage. Lower ROUGE-2/L is expected — abstractive summaries paraphrase rather than copy, which penalizes n-gram overlap metrics. BERTScore compensates for this.

### BERTScore (25 CNN/DailyMail articles)

| Metric | Extractive Baseline | Gemini 2.5 Flash | Delta |
|--------|---------------------|------------------|-------|
| BERTScore Precision | 0.8550 | **0.8703** | **+1.8%** |
| BERTScore Recall | 0.8717 | **0.8861** | **+1.6%** |
| BERTScore F1 | 0.8631 | **0.8780** | **+1.7%** |

Gemini outperforms on all semantic similarity dimensions, confirming quality despite lower n-gram overlap.

### Entity Extraction — Needle-in-a-Haystack

Six synthetic needles with known entities injected at controlled positions (early/middle/deep) across a 230-document corpus. Three extractors benchmarked.

| Metric | SpaCy | Presidio | Gemini 2.5 Flash |
|--------|-------|----------|------------------|
| Detection Rate | 100% | 100% | **100%** |
| Avg Entity Recall | **90.3%** | 44.4% | **90.3%** |
| PERSON | 100% | **100%** | **100%** |
| DATE | 100% | **100%** | **100%** |
| GPE / LOC | 100% | **100%** | **100%** |
| ORG | **100%** | 0% | **100%** |
| MONEY | **100%** | 0% | **100%** |
| PRODUCT | 50% | 50% | **50%** |

Complementary strengths. SpaCy and Gemini match on aggregate recall but differ in boundary precision (SpaCy: "Evelyn Thorncastle"; Gemini: "Dr. Evelyn Thorncastle"). Presidio hits 100% on PII types but doesn't target general NER. Production recommendation: ensemble all three.

## Agentic Workflow: Customer Insight Agent

ReAct pattern — multi-step analytical queries requiring search, extraction, sentiment, and synthesis across document types.

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
│  THINK → Do I have enough?       │
│  ...repeat until confident...    │
│  ANSWER → Synthesize findings    │
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

**7 Tools:** SEARCH, EXTRACT_ENTITIES, EXTRACT_STRUCTURED, ANALYZE_SENTIMENT, SUMMARIZE, SUMMARIZE_MULTIPLE, COMPARE. Each wraps a GCP service. Backend-agnostic — swap BigQuery for Elasticsearch without changing agent logic.

**State Management:** LocalMemory (dev) / Firestore (prod). Follow-up queries build on previous findings.

**Critic Validation:** Optional post-answer pass scores completeness, grounding, and coherence (1-5 each). Flags unsupported claims. One additional API call; doubles as a production audit layer.

### Scaling Patterns

**Actor-Critic:** Critic evaluates each output — entity recall, sentiment plausibility, synthesis completeness. Implemented as optional validation; ready to make mandatory for user-facing traffic.

**Hierarchical Decomposition:** Coordinator decomposes complex queries into sub-goals, dispatches to specialized workers, synthesizes. Enables parallelism and fault isolation. Activation criteria: tool set >15, queries >10 steps, or SLAs requiring parallel execution.

## Challenges & Trade-offs

1. **ROUGE vs. quality:** ROUGE penalizes paraphrasing, inflating extractive scores. Addressed with BERTScore. Production needs both metrics plus human eval.
2. **Rate limiting:** Gemini API limits required batching and backoff. Production: Vertex AI batch prediction or Pub/Sub async queuing.
3. **Streaming data:** CNN/DailyMail (1.3GB) exceeded disk. Solved with HuggingFace streaming.

## Productionization

**Orchestration:** Vertex AI Pipelines for batch extraction/summarization. Cloud Functions + Pub/Sub for real-time ingest. BigQuery for petabyte-scale document storage. Vertex AI Agent Builder as production replacement for the custom ReAct loop.

**Security:** IAM service accounts with least-privilege. CMEK for sensitive corpora. VPC Service Controls for data exfiltration prevention.

**Monitoring:** Cloud Logging with structured metadata (doc ID, stage, latency). Cloud Monitoring dashboards for error rates, ROUGE drift, extraction recall, agent step counts. Dead-letter queues for failed extractions.

**Cost:** Gemini Flash over Pro for routine work (10x cheaper). BigQuery on-demand for prototype, reserved slots for prod. Memorystore caching for repeated queries. Budget alerts at 50/80/100%.

**CI/CD:** Cloud Build: lint → unit tests → integration tests (mocked) → deploy. Pinned model versions in config. Terraform for all GCP resources.
