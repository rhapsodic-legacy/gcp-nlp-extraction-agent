"""Abstractive summarisation via Gemini.

Four modes: single-doc summary, recursive map-reduce for long documents,
multi-doc synthesis, and comparative analysis.
Temperature 0.3 for natural readability without hallucination.
"""

import logging
import os

from google import genai
from google.genai import types

from ..api_utils import generate_with_retry, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Prompts as module-level constants for visibility and versioning.

SINGLE_DOC_PROMPT = """Summarize the following text in 2-3 concise sentences.
Focus on the key facts, main argument, and any actionable information.
Be factual — do not add information not present in the text.

Text:
{text}
"""

CHUNK_SUMMARY_PROMPT = """Summarize the following section of a larger document in 2-3 sentences.
Preserve all key facts, names, numbers, and actionable information.
This summary will be combined with summaries of other sections, so be specific.

Section {chunk_num} of {total_chunks}:
{text}
"""

SYNTHESIS_PROMPT = """The following are summaries of consecutive sections of a single document.
Synthesize them into one coherent summary of 3-5 sentences.
Preserve all key facts and maintain the narrative flow.
Do not add information not present in the section summaries.

Section summaries:
{summaries}
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

# Chunk size for recursive summarization (chars). Chosen to stay well
# within Gemini's context window while leaving room for the prompt.
DEFAULT_CHUNK_SIZE = 6000
# Documents shorter than this go through the single-doc path.
LONG_DOC_THRESHOLD = 8000


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """Split text into chunks at sentence boundaries.

    Tries to break on '. ' to avoid splitting mid-sentence. Falls back
    to hard splits if no sentence boundary is found within the chunk.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Look for the last sentence boundary within the chunk
        boundary = text.rfind(". ", start, end)
        if boundary > start:
            end = boundary + 2  # include the ". "
        else:
            # No sentence boundary — try newline
            boundary = text.rfind("\n", start, end)
            if boundary > start:
                end = boundary + 1

        chunks.append(text[start:end])
        start = end

    return chunks


class GeminiSummarizer:
    """Abstractive summarisation using Gemini via Google AI Studio.

    Four modes: single-doc, recursive (map-reduce for long docs),
    multi-doc synthesis, and comparative analysis.
    All prompt-driven; no separate models or fine-tuning required.
    """

    def __init__(self, api_key: str = None, model_name: str = DEFAULT_MODEL,
                 chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.chunk_size = chunk_size

    def _generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Shared generation call with standard config."""
        response = generate_with_retry(
            self.client,
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()

    def summarize(self, text: str) -> str:
        """Summarise a single document. Automatically uses recursive
        map-reduce for documents longer than the threshold."""
        if len(text) <= LONG_DOC_THRESHOLD:
            return self._summarize_short(text)
        return self.summarize_long(text)

    def _summarize_short(self, text: str) -> str:
        """Direct summarisation for short documents."""
        prompt = SINGLE_DOC_PROMPT.format(text=text[:LONG_DOC_THRESHOLD])
        return self._generate(prompt)

    def summarize_long(self, text: str) -> str:
        """Recursive map-reduce summarisation for long documents.

        1. MAP:   Split into chunks, summarise each independently
        2. REDUCE: Synthesise chunk summaries into a final summary

        If the combined chunk summaries are still too long, the reduce
        step is applied recursively until the output fits in one pass.
        """
        chunks = chunk_text(text, self.chunk_size)

        if len(chunks) == 1:
            return self._summarize_short(chunks[0])

        logger.info("Map-reduce: %d chunks (%.0f chars total)", len(chunks), len(text))

        # MAP phase: summarise each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            prompt = CHUNK_SUMMARY_PROMPT.format(
                chunk_num=i + 1,
                total_chunks=len(chunks),
                text=chunk,
            )
            summary = self._generate(prompt)
            chunk_summaries.append(summary)

        # REDUCE phase: synthesise chunk summaries
        return self._reduce_summaries(chunk_summaries)

    def _reduce_summaries(self, summaries: list[str]) -> str:
        """Recursively reduce chunk summaries until they fit in one prompt."""
        combined = "\n\n".join(
            f"[Section {i + 1}]: {s}" for i, s in enumerate(summaries)
        )

        # If combined summaries fit in one pass, synthesise directly
        if len(combined) <= LONG_DOC_THRESHOLD:
            prompt = SYNTHESIS_PROMPT.format(summaries=combined)
            return self._generate(prompt, max_tokens=1024)

        # Still too long — recurse: re-chunk the summaries and reduce again
        logger.info("Reduce: %d summaries still too long, recursing", len(summaries))
        mid = len(summaries) // 2
        left = self._reduce_summaries(summaries[:mid])
        right = self._reduce_summaries(summaries[mid:])

        combined = f"[Part 1]: {left}\n\n[Part 2]: {right}"
        prompt = SYNTHESIS_PROMPT.format(summaries=combined)
        return self._generate(prompt, max_tokens=1024)

    def summarize_multiple(self, texts: list[str]) -> str:
        """Synthesise multiple documents into one coherent summary.

        Each input truncated to 2000 chars to fit multiple documents in one prompt.
        """
        formatted = "\n\n---\n\n".join(
            f"[Document {i + 1}]:\n{t[:2000]}" for i, t in enumerate(texts)
        )
        prompt = MULTI_DOC_PROMPT.format(n=len(texts), documents=formatted)
        return self._generate(prompt, max_tokens=1024)

    def compare(self, summaries: list[str]) -> str:
        """Compare and contrast multiple document summaries."""
        formatted = "\n\n".join(
            f"[Summary {i + 1}]: {s}" for i, s in enumerate(summaries)
        )
        prompt = COMPARATIVE_PROMPT.format(summaries=formatted)
        return self._generate(prompt, max_tokens=1024)

    def summarize_batch(self, texts: list[str]) -> list[str]:
        """Summarise multiple documents individually. Sequential; production: async."""
        return [self.summarize(text) for text in texts]
