"""Generate PDF report from architecture_report.md using fpdf2.

Tuned to fit all content on exactly 2 pages with professional formatting.
"""

from fpdf import FPDF

REPORT_DIR = "/Users/jessepassmore/Desktop/Programming_Pizazz/nlp_fun/nlp_parsing_gcp"


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(128)
        self.cell(0, 4, "NLP Extraction & Summarisation with Agentic Workflows on GCP", align="R")
        self.ln(5)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(128)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(0, 102, 204)
        self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def subsection_title(self, text):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(51, 51, 51)
        self.cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def body_text(self, text, spacing=1.5):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(33, 33, 33)
        self.multi_cell(0, 3.8, text)
        self.ln(spacing)

    def bold_text(self, text):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(33, 33, 33)
        self.multi_cell(0, 3.8, text)
        self.ln(0.5)

    def table_row(self, cells, bold=False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 7)
        col_widths = [max(25, 190 / len(cells))] * len(cells)
        if len(cells) == 4:
            col_widths = [45, 50, 50, 45]
        elif len(cells) == 3:
            col_widths = [70, 60, 60]
        for i, cell in enumerate(cells):
            w = col_widths[i] if i < len(col_widths) else 40
            self.cell(w, 4.2, str(cell).strip(), border=1)
        self.ln()


def build_pdf():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 51, 102)
    pdf.multi_cell(0, 6, "Intelligent Data Extraction & Summarisation\nwith Agentic Workflows on GCP", align="C")
    pdf.ln(2)

    # Section 1: Technical Approach
    pdf.section_title("Technical Approach & GCP Services")
    pdf.body_text(
        "Entity extraction, abstractive summarisation, and multi-step analytical queries orchestrated "
        "through a ReAct-style agent. Dual-backend architecture: local tools (SpaCy, ROUGE) as development "
        "baselines; GCP-native services (Gemini, BigQuery, Firestore) for production."
    )

    pdf.bold_text("Datasets: Three heterogeneous sources to stress-test extraction:")
    pdf.body_text(
        "- Amazon Product Reviews (~74k rows): short, opinionated, entity-sparse\n"
        "- Customer Support Tickets (~29.8k rows): template-heavy, action-oriented\n"
        "- CNN/DailyMail News (~11.5k, streamed): long-form, entity-rich, with gold-standard human summaries",
        spacing=1,
    )

    pdf.bold_text("GCP Services Used:")
    pdf.table_row(["Service", "Role", "Why"], bold=True)
    pdf.table_row(["Gemini 2.5 Flash", "Summarisation, extraction, agent reasoning", "Single model; JSON mode output"])
    pdf.table_row(["Cloud Storage", "Raw data landing zone (50MB+)", "Standard ingest layer for pipelines"])
    pdf.table_row(["BigQuery", "Document store (400 docs loaded)", "SQL scales to PB; parameterised queries"])
    pdf.table_row(["Firestore", "Agent session memory", "Real-time, document-oriented"])
    pdf.ln(2)

    # Section 2: Results
    pdf.section_title("Results & Evaluation")

    pdf.subsection_title("Summarisation: ROUGE Scores (25 CNN/DailyMail articles)")
    pdf.table_row(["Metric", "Extractive Baseline", "Gemini 2.5 Flash", "Delta"], bold=True)
    pdf.table_row(["ROUGE-1 F1", "0.298", "0.327", "+9.7%"])
    pdf.table_row(["ROUGE-2 F1", "0.127", "0.100", "-21.5%"])
    pdf.table_row(["ROUGE-L F1", "0.217", "0.207", "-4.6%"])
    pdf.ln(0.5)
    pdf.body_text(
        "ROUGE-1 improvement reflects superior information coverage. Lower ROUGE-2/L is expected: "
        "abstractive paraphrasing reduces n-gram overlap (Kryscinski et al., 2019). BERTScore addresses this.",
        spacing=1,
    )

    pdf.subsection_title("BERTScore: Semantic Similarity (25 CNN/DailyMail articles)")
    pdf.table_row(["Metric", "Extractive Baseline", "Gemini 2.5 Flash", "Delta"], bold=True)
    pdf.table_row(["BERTScore Precision", "0.8550", "0.8703", "+1.8%"])
    pdf.table_row(["BERTScore Recall", "0.8717", "0.8861", "+1.6%"])
    pdf.table_row(["BERTScore F1", "0.8631", "0.8780", "+1.7%"])
    pdf.ln(0.5)
    pdf.body_text(
        "Gemini outperforms across all BERTScore dimensions (RoBERTa-large embeddings), confirming "
        "semantic superiority despite lower ROUGE-2/L.",
        spacing=1,
    )

    pdf.subsection_title("Information Extraction: Needle-in-a-Haystack (3-way comparison)")
    pdf.body_text(
        "Six synthetic needles with known entities injected at controlled positions across a "
        "230-document corpus. Three extractors benchmarked: SpaCy (general NER), Presidio "
        "(PII-focused), Gemini 2.5 Flash (LLM-based).",
        spacing=0.5,
    )
    pdf.table_row(["Metric", "SpaCy", "Presidio", "Gemini"], bold=True)
    pdf.table_row(["Detection Rate", "100%", "100%", "100%"])
    pdf.table_row(["Avg Entity Recall", "90.3%", "44.4%", "90.3%"])
    pdf.table_row(["PERSON / DATE", "100%", "100%", "100%"])
    pdf.table_row(["GPE / LOC", "100%", "100%", "100%"])
    pdf.table_row(["ORG / MONEY", "100%", "0%", "100%"])
    pdf.table_row(["PRODUCT", "50%", "50%", "50%"])
    pdf.ln(0.5)
    pdf.body_text(
        "Complementary strengths: Presidio achieves 100% on PII types but does not target general NER (ORG, MONEY). "
        "SpaCy and Gemini match on aggregate recall; Gemini captures more precise boundaries. "
        "Production recommendation: combine all three.",
        spacing=1,
    )

    # Section 3: Agentic Workflow
    pdf.section_title("Agentic Workflow: Customer Insight Agent")
    pdf.body_text(
        "Multi-step analytical queries requiring search, extraction, sentiment analysis, and synthesis "
        "across document types. No single API call suffices.",
        spacing=1,
    )
    pdf.bold_text("Architecture (ReAct Pattern):")
    pdf.body_text(
        "User Query -> THINK (what do I need?) -> ACT (call a tool) -> OBSERVE (process result) "
        "-> THINK (do I have enough?) -> ...repeat... -> ANSWER (synthesise findings)",
        spacing=1,
    )
    pdf.body_text(
        "7 Tools: SEARCH, EXTRACT_ENTITIES, EXTRACT_STRUCTURED, ANALYZE_SENTIMENT, SUMMARIZE, "
        "SUMMARIZE_MULTIPLE, COMPARE. Each wraps a GCP service behind a standard interface. "
        "Backend-agnostic: swap BigQuery for Elasticsearch without changing agent logic.",
        spacing=1,
    )
    pdf.body_text(
        "State Management: Session memory via LocalMemory (dev) or Firestore (prod). Follow-up queries "
        "build on previous findings, enabling multi-turn analytical sessions.",
        spacing=1,
    )
    pdf.body_text(
        "Critic Validation: Optional post-answer pass evaluates completeness, factual grounding, and "
        "coherence. Flags unsupported claims before they reach the user. One additional API call; "
        "doubles as a production audit layer.",
        spacing=1,
    )

    pdf.subsection_title("Scaling Agent Architecture")
    pdf.body_text(
        "Actor-Critic Pair: Separate critic evaluates each output for entity recall, sentiment "
        "plausibility, and synthesis completeness. Implemented as optional validation pass; "
        "ready to be mandatory for customer-facing traffic.",
        spacing=1,
    )
    pdf.body_text(
        "Hierarchical Decomposition: Coordinator agent decomposes complex queries into sub-goals, "
        "dispatches to specialised workers, synthesises outputs. Enables parallelism and fault isolation. "
        "Activation criteria: tools >15, queries >10 steps, or SLAs requiring parallel execution.",
        spacing=1,
    )

    # Section 4: Challenges
    pdf.section_title("Challenges & Trade-offs")
    pdf.body_text(
        "1. ROUGE vs. quality: ROUGE penalises paraphrasing. Addressed with BERTScore (+1.7% F1), "
        "confirming Gemini's semantic superiority. Production needs both metrics plus human evaluation.\n"
        "2. Rate limiting: Gemini API limits required batching and backoff. Production would use "
        "Vertex AI batch prediction or Pub/Sub async queuing.\n"
        "3. Streaming data: CNN/DailyMail's 1.3GB exceeded disk. Solved with HuggingFace streaming; "
        "scales to arbitrarily large datasets.",
        spacing=1,
    )

    # Section 5: Productionisation
    pdf.section_title("Productionisation Approach")

    pdf.bold_text("Scalability & Orchestration:")
    pdf.body_text(
        "- Vertex AI Pipelines for batch extraction/summarisation with retry\n"
        "- Cloud Functions + Pub/Sub for real-time processing; each tool as an independent function\n"
        "- BigQuery for document storage and search at scale (petabyte-ready)\n"
        "- Vertex AI Agent Builder as production replacement for custom ReAct loop: managed orchestration "
        "with grounding and tool management; custom agent validates tool designs before promotion",
        spacing=1,
    )

    pdf.bold_text("Security & Data Privacy:")
    pdf.body_text(
        "- IAM service accounts with least-privilege (separate roles per function)\n"
        "- Customer-managed encryption keys (CMEK) for sensitive corpora\n"
        "- VPC Service Controls to restrict data exfiltration",
        spacing=1,
    )

    pdf.bold_text("Monitoring & Cost Management:")
    pdf.body_text(
        "- Cloud Logging + Monitoring dashboards (error rates, score drift, latency)\n"
        "- Dead-letter queues for failed extractions; no document silently dropped\n"
        "- Gemini Flash over Pro (10x cheaper); BigQuery reserved slots for production\n"
        "- Memorystore/Redis caching; budget alerts at 50%/80%/100%",
        spacing=1,
    )

    pdf.bold_text("CI/CD & Reproducibility:")
    pdf.body_text(
        "- Cloud Build: lint -> unit tests -> integration tests (mocked APIs) -> deploy\n"
        "- Pinned model versions (gemini-2.5-flash, not 'latest') to prevent regression\n"
        "- Infrastructure as Code (Terraform) for all GCP resources",
        spacing=2,
    )

    # Section 6: Repository Structure
    pdf.section_title("Repository & Deliverables")
    pdf.set_font("Courier", "", 7)
    pdf.set_text_color(33, 33, 33)
    pdf.multi_cell(0, 3.5,
        "nlp_parsing_gcp/\n"
        "  src/\n"
        "    data/           loader.py, preprocessing.py: unified Document format\n"
        "    extraction/     spacy_extract.py, presidio_extract.py, vertex_extract.py\n"
        "    summarisation/  extractive.py, vertex_summarise.py (Gemini abstractive)\n"
        "    evaluation/     rouge_eval.py, needle_haystack.py, bertscore_eval.py\n"
        "    agent/          agent.py (ReAct), critic.py (actor-critic), tools.py, memory.py\n"
        "  tests/\n"
        "    test_agent.py              : 27 unit tests (parsing, search, tool dispatch, loop)\n"
        "    test_critic.py             : 7 unit tests (verdict, scoring, error handling)\n"
        "    test_presidio.py           : 10 unit tests (PII extraction, batch, interface)\n"
        "    test_integration.py        : 13 integration tests (multi-step, actor-critic, E2E)\n"
        "  notebooks/\n"
        "    01_data_exploration.ipynb   : EDA across all 3 datasets\n"
        "    02_extraction_and_eval.ipynb: SpaCy vs Gemini benchmarks\n"
        "    03_agent_demo.ipynb         : live agent queries with critic validation\n"
        "  config/gcp_config.yaml       : pinned model versions, project settings\n"
        "  scripts/                     : BERTScore runner, BigQuery loader, E2E runner\n"
        "  architecture_report.pdf      : this document"
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(100)
    pdf.cell(0, 3.5, "GCP resources: gs://nlp-extraction-data-333762334828/ | BigQuery: nlp_extraction.documents (400 rows)", align="C")

    out = f"{REPORT_DIR}/architecture_report.pdf"
    pdf.output(out)
    print(f"PDF generated: {out} ({pdf.page_no()} pages)")


if __name__ == "__main__":
    build_pdf()
