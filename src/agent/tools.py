"""Tool definitions for the Customer Insight Agent.

In any good system, the tools are the interface between the brain and the
world. The agent thinks and plans, but the tools are what actually *do*
things — search the database, extract entities, analyze sentiment, write
summaries. Each tool wraps a GCP service (or a local fallback) and exposes
a clean, callable interface.

The design principle here is the same one I used designing peripheral cards
for the Apple II: each tool has a standard interface, does one thing well,
and the system doesn't care about its internals. You could swap BigQuery
for Elasticsearch, or GCP NL API for a custom model, and the agent wouldn't
know the difference. That's modularity done right.

All tools use lazy initialization via @property — they don't create GCP
clients until you actually call them. This means you can import and
configure the agent without needing live credentials, which is a big
quality-of-life win for testing and development.
"""

import json
import os
from typing import Optional

import pandas as pd

from ..extraction.vertex_extract import GeminiExtractor
from ..summarization.vertex_summarize import GeminiSummarizer


class SearchTool:
    """Search documents by keywords and filters — the agent's way of finding relevant data.

    This is the "eyes" of the agent. Before it can extract or summarize anything,
    it needs to find the right documents. The search tool has two paths:

    1. BigQuery (production): Full SQL-powered search over the entire corpus.
       Scales to millions of documents without breaking a sweat.
    2. Local DataFrame (MVP/dev): Simple keyword matching on an in-memory DataFrame.
       No credentials needed, instant results, perfect for prototyping.

    The agent doesn't know or care which path is active — it just calls search()
    and gets results. That's the whole point.
    """

    def __init__(self, project_id: str = None, dataset: str = "nlp_fun", documents_df: pd.DataFrame = None):
        self.project_id = project_id
        self.dataset = dataset
        self.documents_df = documents_df
        self._bq_client = None

    @property
    def bq_client(self):
        """Lazy BigQuery client — only connects when you actually need it."""
        if self._bq_client is None and self.project_id:
            from google.cloud import bigquery
            self._bq_client = bigquery.Client(project=self.project_id)
        return self._bq_client

    def search(
        self,
        query: str,
        source_type: Optional[str] = None,
        max_results: int = 10,
    ) -> list[dict]:
        """Search documents matching query terms.

        Automatically picks the right backend — local DataFrame if one was
        provided at init, BigQuery otherwise. The agent just calls this
        and gets results. Clean separation of concerns.
        """
        if self.documents_df is not None:
            return self._search_local(query, source_type, max_results)
        return self._search_bigquery(query, source_type, max_results)

    def _search_local(self, query: str, source_type: Optional[str], max_results: int) -> list[dict]:
        """Simple keyword search on a local DataFrame.

        Not fancy, but it works. Case-insensitive, requires ALL terms to match.
        For a prototype, this gets the job done without any infrastructure.
        """
        df = self.documents_df.copy()
        if source_type:
            df = df[df["source_type"] == source_type]

        # All query terms must appear in the text (case-insensitive AND logic)
        terms = query.lower().split()
        mask = df["text"].str.lower().apply(lambda t: all(term in t for term in terms))
        results = df[mask].head(max_results)

        return results.to_dict("records")

    def _search_bigquery(self, query: str, source_type: Optional[str], max_results: int) -> list[dict]:
        """Search via BigQuery using parameterized LIKE matching.

        Uses parameterized queries to prevent SQL injection — because even
        in a prototype, you should never concatenate user input into SQL.
        That's just basic hygiene. Like washing your hands before soldering.
        """
        from google.cloud import bigquery

        where_clauses = []
        params = []

        for i, term in enumerate(query.split()):
            where_clauses.append(f"LOWER(text) LIKE @term_{i}")
            params.append(bigquery.ScalarQueryParameter(f"term_{i}", "STRING", f"%{term.lower()}%"))

        if source_type:
            where_clauses.append("source_type = @source_type")
            params.append(bigquery.ScalarQueryParameter("source_type", "STRING", source_type))

        where = " AND ".join(where_clauses) if where_clauses else "TRUE"

        sql = f"""
            SELECT id, text, source_type, metadata, created_at
            FROM `{self.project_id}.{self.dataset}.documents`
            WHERE {where}
            LIMIT {max_results}
        """

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        results = self.bq_client.query(sql, job_config=job_config)
        return [dict(row) for row in results]


class ExtractTool:
    """Entity extraction tool — uses Gemini for both NER and structured extraction.

    Two extraction modes, one interface. Entity extraction handles named entity
    recognition (people, places, orgs, dates, products). Structured extraction
    handles higher-level semantic information (core issues, attributes, action
    items, topics). Together they cover both the "what's mentioned" and "what
    does it mean" questions.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._gemini_extractor = None

    @property
    def gemini_extractor(self):
        """Lazy init — Gemini model loads only when extraction is needed."""
        if self._gemini_extractor is None:
            self._gemini_extractor = GeminiExtractor(api_key=self.api_key)
        return self._gemini_extractor

    def extract_entities(self, text: str) -> dict:
        """Extract named entities via Gemini. Comprehensive and context-aware."""
        entities = self.gemini_extractor.extract_entities(text)
        return {"entities": [{"text": e.text, "type": e.type, "salience": e.salience} for e in entities]}

    def extract_structured(self, text: str) -> dict:
        """Extract semantic structure (issues, attributes, topics) via Gemini.

        This is the deeper extraction — it requires comprehension, not just
        pattern matching. Gemini reads the text and figures out what the
        core problems, attributes, and action items are.
        """
        result = self.gemini_extractor.extract_structured(text)
        return result.to_dict()


class SentimentTool:
    """Sentiment analysis — uses Gemini to assess document sentiment.

    Simple and focused: give it text, get back a score (-1 to +1) and
    magnitude. The agent uses this to understand how customers feel about
    things without having to read every single review.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy init — no API client until the first sentiment call."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def analyze(self, text: str) -> dict:
        """Analyze document-level sentiment. Returns score and magnitude."""
        from google.genai import types

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""Analyze the sentiment of the following text.
Return a JSON object with:
- "score": a float from -1.0 (very negative) to 1.0 (very positive)
- "magnitude": a float from 0.0 to 5.0 indicating how emotionally charged the text is

Text:
{text[:4000]}""",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=256,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        try:
            parsed = json.loads(response.text)
            return {"score": parsed.get("score", 0.0), "magnitude": parsed.get("magnitude", 0.0)}
        except (json.JSONDecodeError, AttributeError):
            return {"score": 0.0, "magnitude": 0.0}


class SummarizeTool:
    """Summarization tool — wraps Gemini's text generation for summaries.

    Three capabilities in one tool: single-doc summaries, multi-doc
    synthesis, and comparative analysis. The agent picks whichever mode
    fits its current reasoning step.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._summarizer = None

    @property
    def summarizer(self):
        """Lazy init — Gemini model loads on first summarize call."""
        if self._summarizer is None:
            self._summarizer = GeminiSummarizer(api_key=self.api_key)
        return self._summarizer

    def summarize(self, text: str) -> str:
        """Summarize a single document into 2-3 sentences."""
        return self.summarizer.summarize(text)

    def summarize_multiple(self, texts: list[str]) -> str:
        """Synthesize multiple documents into one coherent summary."""
        return self.summarizer.summarize_multiple(texts)

    def compare(self, summaries: list[str]) -> str:
        """Compare and contrast multiple summaries for the agent's analysis."""
        return self.summarizer.compare(summaries)
