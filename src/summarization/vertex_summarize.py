"""Abstractive summarisation via Gemini.

Three modes: single-doc summary, multi-doc synthesis, and comparative analysis.
Temperature 0.3 for natural readability without hallucination.
"""

import os

from google import genai
from google.genai import types

from ..api_utils import generate_with_retry, DEFAULT_MODEL


# Prompts as module-level constants for visibility and versioning.

SINGLE_DOC_PROMPT = """Summarize the following text in 2-3 concise sentences.
Focus on the key facts, main argument, and any actionable information.
Be factual — do not add information not present in the text.

Text:
{text}
"""

MULTI_DOC_PROMPT = """Summarize the following {n} documents into a single coherent summary.
Identify common themes, key differences, and the overall narrative.
Keep it to 3-5 sentences.

Documents:
{documents}
"""

COMPARATIVE_PROMPT = """Compare and contrast the following document summaries.
Identify: (1) common themes, (2) key differences, (3) notable outliers.
Be specific and cite which documents support each point.

Summaries:
{summaries}
"""


class GeminiSummarizer:
    """Abstractive summarisation using Gemini via Google AI Studio.

    Three modes: single-doc, multi-doc synthesis, and comparative analysis.
    All prompt-driven; no separate models or fine-tuning required.
    """

    def __init__(self, api_key: str = None, model_name: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def summarize(self, text: str) -> str:
        """Summarise a single document into 2-3 sentences. Input capped at 8000 chars."""
        prompt = SINGLE_DOC_PROMPT.format(text=text[:8000])
        response = generate_with_retry(
            self.client,
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=512,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()

    def summarize_multiple(self, texts: list[str]) -> str:
        """Synthesise multiple documents into one coherent summary.

        Each input truncated to 2000 chars to fit multiple documents in one prompt.
        """
        formatted = "\n\n---\n\n".join(
            f"[Document {i + 1}]:\n{t[:2000]}" for i, t in enumerate(texts)
        )
        prompt = MULTI_DOC_PROMPT.format(n=len(texts), documents=formatted)
        response = generate_with_retry(
            self.client,
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()

    def compare(self, summaries: list[str]) -> str:
        """Compare and contrast multiple document summaries."""
        formatted = "\n\n".join(
            f"[Summary {i + 1}]: {s}" for i, s in enumerate(summaries)
        )
        prompt = COMPARATIVE_PROMPT.format(summaries=formatted)
        response = generate_with_retry(
            self.client,
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()

    def summarize_batch(self, texts: list[str]) -> list[str]:
        """Summarise multiple documents individually. Sequential; production: async."""
        return [self.summarize(text) for text in texts]
