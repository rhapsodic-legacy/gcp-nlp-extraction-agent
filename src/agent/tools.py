"""Tool definitions for the Customer Insight Agent.

Each tool wraps a GCP service (or local fallback) behind a uniform callable
interface. GCP clients are lazily initialized so the module can be imported
without live credentials.
"""

import json
import os
from typing import Optional

import pandas as pd

from ..extraction.vertex_extract import GeminiExtractor
from ..summarization.vertex_summarize import GeminiSummarizer
from ..api_utils import generate_with_retry, DEFAULT_MODEL


class SearchTool:
    """Keyword search over documents via BigQuery or a local DataFrame.

    The active backend is selected automatically based on whether a
    ``documents_df`` was provided at init. The agent does not know or
    care which backend is active.
    """

    def __init__(self, project_id: str = None, dataset: str = "nlp_fun", documents_df: pd.DataFrame = None):
        self.project_id = project_id
        self.dataset = dataset
        self.documents_df = documents_df
        self._bq_client = None

    @property
    def bq_client(self):
        """Lazily initialized BigQuery client."""
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

        Dispatches to the local DataFrame backend or BigQuery depending
        on initialization.
        """
        if self.documents_df is not None:
            return self._search_local(query, source_type, max_results)
        return self._search_bigquery(query, source_type, max_results)

    def _search_local(self, query: str, source_type: Optional[str], max_results: int) -> list[dict]:
        """Case-insensitive keyword search on the local DataFrame.

        All query terms must appear in the text (AND logic).
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
        """Search via BigQuery using parameterized LIKE matching."""
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
    """Entity and structured extraction via Gemini.

    Supports two modes: named-entity recognition (people, orgs, products,
    dates) and semantic structured extraction (core issues, attributes,
    topics, action items).
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._gemini_extractor = None

    @property
    def gemini_extractor(self):
        """Lazily initialized GeminiExtractor."""
        if self._gemini_extractor is None:
            self._gemini_extractor = GeminiExtractor(api_key=self.api_key)
        return self._gemini_extractor

    def extract_entities(self, text: str) -> dict:
        """Extract named entities from text via Gemini."""
        entities = self.gemini_extractor.extract_entities(text)
        return {"entities": [{"text": e.text, "type": e.type, "salience": e.salience} for e in entities]}

    def extract_structured(self, text: str) -> dict:
        """Extract semantic structure (issues, attributes, topics, action items) via Gemini."""
        result = self.gemini_extractor.extract_structured(text)
        return result.to_dict()


class SentimentTool:
    """Sentiment analysis via Gemini.

    Returns a score (-1 to +1) and magnitude for the input text.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazily initialized genai client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def analyze(self, text: str) -> dict:
        """Analyze document-level sentiment. Returns score and magnitude."""
        from google.genai import types

        response = generate_with_retry(
            self.client,
            model=DEFAULT_MODEL,
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
    """Summarization via Gemini.

    Supports single-document summaries, multi-document synthesis, and
    comparative analysis.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._summarizer = None

    @property
    def summarizer(self):
        """Lazily initialized GeminiSummarizer."""
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
