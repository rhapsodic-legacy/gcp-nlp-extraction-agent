"""Abstractive summarization via Gemini on Vertex AI.

Summarization is one of those problems that sounds simple until you actually
try to do it well. "Just make it shorter" — sure, but which parts do you
keep? What's the main point? What's actionable? A good summary is like a
good schematic: it shows you everything important and nothing that isn't.

This module handles three flavors of summarization, all powered by Gemini:
1. Single document → 2-3 sentence summary (the bread and butter)
2. Multi-document synthesis → one coherent summary from many sources
3. Comparative analysis → what's similar, what's different, what's weird

Temperature is set to 0.3 — a touch of creativity helps summaries read
naturally, but we don't want the model making stuff up. It's summarization,
not fiction writing.
"""

import os

from google import genai
from google.genai import types


# Prompts are separated out as module-level constants because they're the
# "interface contract" with the model. Treat them like API specs — visible,
# versioned, and easy to iterate on.

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
    """Abstractive summarization using Gemini via Google AI Studio.

    Three modes, one class. Single-doc summarization is the workhorse —
    you'll call it thousands of times across your corpus. Multi-doc synthesis
    is for when the agent needs to combine findings. Comparative analysis
    is for when you want to understand how different document sets relate.

    I love that Gemini handles all three with just prompt engineering.
    No separate models, no retraining, no fuss. You just describe what
    you want and it figures out how to do it. That's good tool design.
    """

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def summarize(self, text: str) -> str:
        """Summarize a single document into 2-3 sentences.

        Caps input at 8000 chars — Gemini can handle much more, but
        there's a point of diminishing returns and increasing cost.
        Most documents in our corpus are well under this limit anyway.
        """
        prompt = SINGLE_DOC_PROMPT.format(text=text[:8000])
        response = self.client.models.generate_content(
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
        """Synthesize multiple documents into one coherent summary.

        Each input gets truncated to 2000 chars to fit several documents
        into a single prompt. The model identifies common threads and
        key differences — it's surprisingly good at finding the signal
        when you give it multiple perspectives on the same topic.
        """
        formatted = "\n\n---\n\n".join(
            f"[Document {i + 1}]:\n{t[:2000]}" for i, t in enumerate(texts)
        )
        prompt = MULTI_DOC_PROMPT.format(n=len(texts), documents=formatted)
        response = self.client.models.generate_content(
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
        """Compare and contrast multiple document summaries.

        This is the analytical mode — give it a set of summaries and
        it'll tell you what they have in common, where they disagree,
        and which ones are outliers. Great for the agent when it needs
        to synthesize findings across a search result set.
        """
        formatted = "\n\n".join(
            f"[Summary {i + 1}]: {s}" for i, s in enumerate(summaries)
        )
        prompt = COMPARATIVE_PROMPT.format(summaries=formatted)
        response = self.client.models.generate_content(
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
        """Summarize multiple documents individually. Sequential for the MVP.

        Same story as batch extraction — sequential now, async later.
        Each text gets its own summarize() call. Simple, predictable,
        easy to debug when one document produces a weird summary.
        """
        return [self.summarize(text) for text in texts]
