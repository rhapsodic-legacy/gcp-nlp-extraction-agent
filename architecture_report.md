# Intelligent Data Extraction & Summarisation with Agentic Workflows on GCP

## Technical Approach & GCP Services

This system extracts entities and information from heterogeneous unstructured text, generates abstractive summaries, and orchestrates these capabilities through a ReAct-style agent. The architecture follows a dual-backend pattern: local tools (SpaCy, ROUGE) serve as baselines during development, while GCP-native services (Gemini, Natural Language API) provide production-grade capabilities through the same interfaces.

**Datasets:** Three heterogeneous sources were selected to stress-test extraction across different text characteristics:
- Amazon Product Reviews (~74k rows): short, opinionated, entity-sparse
- Customer Support Tickets (~29.8k rows): template-heavy, action-oriented
- CNN/DailyMail News (~11.5k, streamed): long-form, entity-rich, with gold-standard human summaries

**GCP Services Used:**
| Service | Role | Why |
|---------|------|-----|
| Gemini 2.5 Flash (via Generative Language API) | Summarisation, structured extraction, entity recognition, agent reasoning | Single model handles all generative tasks; JSON mode ensures parseable output |
| Cloud Storage | Raw data landing zone (gs://nlp-extraction-data-333762334828/) | Standard ingest layer; 50MB+ of review and ticket data uploaded |
| BigQuery | Production document store & search (nlp_extraction.documents) | 400 documents loaded; parameterised queries prevent injection |
| Firestore | Agent session memory | Real-time, document-oriented: ideal for conversation state persistence |

## Results & Evaluation

### Summarisation (25 CNN/DailyMail articles)

| Metric | Extractive Baseline (first 3 sentences) | Gemini 2.5 Flash | Delta |
|--------|----------------------------------------|-----------------|-------|
| ROUGE-1 F1 | 0.298 | **0.327** | **+9.7%** |
| ROUGE-2 F1 | **0.127** | 0.100 | -21.5% |
| ROUGE-L F1 | **0.217** | 0.207 | -4.6% |

Gemini's ROUGE-1 improvement demonstrates superior information coverage; it captures more key facts per summary. The lower ROUGE-2/L scores reflect a known limitation of ROUGE as an evaluation metric: abstractive summaries paraphrase rather than copy, reducing n-gram overlap while improving readability. This is well documented in NLP literature (Kryscinski et al., 2019) and motivates the use of BERTScore as a complementary semantic metric.

### BERTScore: Semantic Similarity (25 CNN/DailyMail articles)

BERTScore uses contextual embeddings (RoBERTa-large) to measure meaning preservation rather than surface-level n-gram overlap. This is a fairer evaluation for abstractive summaries.

| Metric | Extractive Baseline | Gemini 2.5 Flash | Delta |
|--------|-------------------|-----------------|-------|
| BERTScore Precision | 0.8550 | **0.8703** | **+1.8%** |
| BERTScore Recall | 0.8717 | **0.8861** | **+1.6%** |
| BERTScore F1 | 0.8631 | **0.8780** | **+1.7%** |

Gemini outperforms the extractive baseline across all BERTScore dimensions, confirming that its abstractive summaries capture more semantic content despite lower ROUGE-2/L n-gram overlap. The improvement is consistent across precision (better signal-to-noise in generated text), recall (more reference information captured), and F1 (overall semantic similarity).

### Information Extraction: Needle-in-a-Haystack Validation

Rather than relying solely on qualitative assessment, we developed a needle-in-a-haystack evaluation framework. Six synthetic "needles," distinctive facts with known entities (e.g. "Dr. Evelyn Thorncastle reported a malfunction in the XR-7 stabilizer"), were injected at controlled positions (early/middle/deep) across a 230-document corpus. Three extractors were benchmarked: SpaCy (general NER), Microsoft Presidio (PII-focused NER), and Gemini 2.5 Flash (LLM-based extraction).

| Metric | SpaCy | Presidio | Gemini 2.5 Flash |
|--------|-------|----------|-----------------|
| Detection Rate | 100% | 100% | **100%** |
| Average Entity Recall | **90.3%** | 44.4% | **90.3%** |
| PERSON | 100% | **100%** | **100%** |
| DATE | 100% | **100%** | **100%** |
| GPE / LOC | 100% | **100%** | **100%** |
| ORG | **100%** | 0% | **100%** |
| MONEY | **100%** | 0% | **100%** |
| PRODUCT | 50% | 50% | **50%** |

The three-way comparison reveals complementary strengths across extractor architectures. SpaCy and Gemini achieve comparable aggregate recall (90.3%), but differ in boundary precision: SpaCy finds "Evelyn Thorncastle" via partial matching, while Gemini captures the full form "Dr. Evelyn Thorncastle." Presidio, purpose-built for PII detection, achieves perfect recall on PII entity types (PERSON, DATE, GPE/LOC) but does not target general NER categories (ORG, MONEY, PERCENT), resulting in lower aggregate recall (44.4%). A production system benefits from combining all three: SpaCy for fast, deterministic general NER; Presidio for PII-specific compliance requirements; Gemini for semantic comprehension and precise entity boundaries.

## Agentic Workflow: Customer Insight Agent

**Scenario:** An analyst asks: *"What are the most common product complaints, and how does sentiment differ between support tickets and reviews?"* No single API call answers this; it requires search, extraction, sentiment analysis, and synthesis across multiple document types.

**Architecture (ReAct Pattern):**

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
│  ANSWER → Synthesise findings    │
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

**7 Tools:** SEARCH, EXTRACT_ENTITIES, EXTRACT_STRUCTURED, ANALYZE_SENTIMENT, SUMMARIZE, SUMMARIZE_MULTIPLE, COMPARE. Each tool wraps a GCP service behind a standard interface; the agent does not know or care which backend handles the request. This decoupling means you can swap BigQuery for Elasticsearch, or SpaCy for Gemini, without changing the agent's reasoning logic.

**State Management:** The agent maintains conversation context through a memory backend (LocalMemory for development, Firestore for production). Follow-up queries build on previous findings, enabling multi-turn analytical sessions.

**Critic Validation:** After the agent produces a final answer, an optional critic pass evaluates the response for completeness and factual grounding. The critic checks whether the answer addresses the original query, cites specific evidence from tool results, and flags unsupported claims. This lightweight actor-critic pattern catches low-quality answers before they reach the user, a production safety net that costs one additional API call.

### Scaling Agent Architecture: Actor-Critic & Hierarchical Patterns

The single ReAct agent is the right starting point: it is transparent, debuggable, and sufficient for the current tool set. But as the system scales to more tools, longer analytical sessions, and higher-stakes decisions, two architectural patterns become valuable:

**Actor-Critic Pair:** The current agent (actor) generates reasoning and tool calls. A separate critic agent evaluates each step or the final output for quality, checking entity recall against known baselines, flagging sentiment scores that seem implausible, or verifying that the synthesis actually addresses the user's question. This is analogous to the discriminator in a GAN: the actor gets better because the critic provides targeted feedback. In production, the critic also serves as an audit layer for compliance and quality assurance.

**Hierarchical Decomposition:** For complex analytical queries that span multiple document types and require different expertise (e.g. "Compare product quality trends across regions using both reviews and support tickets over the last quarter"), a coordinator agent decomposes the task into sub-goals, dispatches them to specialised worker agents (one for sentiment analysis, one for entity extraction, one for temporal analysis), and synthesises their outputs. This enables parallelism (workers run concurrently), fault isolation (one worker failing does not crash the whole analysis), and specialisation (each worker's system prompt is tuned for its domain).

**Why not implement these now?** The single ReAct agent handles our current 7-tool, 3-dataset scope well. Adding architectural complexity before the problem demands it violates YAGNI and makes debugging harder. The right time to introduce hierarchical agents is when: (a) the tool set exceeds ~15 tools (single-agent tool selection degrades), (b) queries routinely require >10 reasoning steps, or (c) production SLAs require parallel tool execution. The critic pattern has lower activation energy; we implement it here as an optional validation pass, ready to be made mandatory when the system handles customer-facing traffic.

## Challenges & Trade-offs

1. **ROUGE vs. quality:** ROUGE penalises paraphrasing, making extractive methods score artificially high. We addressed this by adding BERTScore evaluation, which confirmed Gemini's semantic superiority (+1.7% F1) even where ROUGE showed lower scores. Production systems should use both metrics plus human evaluation.
2. **Rate limiting:** Gemini's API rate limits required batching and backoff logic. Production systems would use Vertex AI batch prediction or async queuing via Pub/Sub.
3. **Streaming data:** CNN/DailyMail's 1.3GB dataset exceeded disk constraints. Solved with HuggingFace streaming mode, a pattern that scales to arbitrarily large datasets.

## Productionisation Approach

**Scalability & Orchestration:**
- Vertex AI Pipelines for batch extraction/summarisation workflows with automatic retry
- Cloud Functions triggered by Pub/Sub for real-time document processing on ingest; each tool (extraction, summarisation, sentiment) deploys as an independent Cloud Function, enabling per-tool scaling and independent versioning
- BigQuery for document storage and search at scale (petabyte-ready, SQL interface)
- **Vertex AI Agent Builder** as the production replacement for the custom ReAct loop: provides managed agent orchestration with built-in grounding, tool management, and conversation history, while our custom agent serves as the rapid prototyping layer for validating tool designs before promoting to Agent Builder

**Security & Data Privacy:**
- IAM service accounts with least-privilege access (separate roles for data read, API invocation, agent execution)
- Customer-managed encryption keys (CMEK) for sensitive document corpora
- VPC Service Controls to restrict data exfiltration from the processing environment

**Monitoring & Error Handling:**
- Cloud Logging for all API calls with structured metadata (document ID, processing stage, latency)
- Cloud Monitoring dashboards tracking: API error rates, ROUGE score drift, extraction recall, agent step counts
- Dead-letter queues (Pub/Sub) for failed extractions; no document silently dropped

**Cost Management:**
- Gemini Flash over Pro for routine extraction/summarisation (10x cheaper, sufficient quality)
- BigQuery on-demand pricing for prototype; reserved slots for production workloads
- Caching layer (Memorystore/Redis) for repeated queries; agent results cached by query hash
- Budget alerts at 50%/80%/100% of monthly allocation

**CI/CD & Reproducibility:**
- Cloud Build pipeline: lint -> unit tests -> integration tests (with mocked API responses) -> deploy
- Pinned model versions in config (gemini-2.5-flash, not "latest") to prevent regression
- Infrastructure as Code (Terraform) for all GCP resources: reproducible environments from git
