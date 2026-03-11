"""Structured information extraction via Gemini.

Complements traditional NER with semantic extraction: core issues,
key attributes, action items, and topics. JSON mode with temperature 0.1
for consistent, parseable output.
"""

import json
import os
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from ..api_utils import generate_with_retry, DEFAULT_MODEL


@dataclass
class StructuredExtraction:
    """Semantic extraction output: issues, attributes, actions, and topics."""

    core_issues: list[str] = field(default_factory=list)
    key_attributes: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "core_issues": self.core_issues,
            "key_attributes": self.key_attributes,
            "action_items": self.action_items,
            "topics": self.topics,
        }


@dataclass
class EntityResult:
    """A named entity extracted by Gemini — compatible with the needle evaluator."""

    text: str
    type: str
    salience: float = 0.0


@dataclass
class GeminiExtractionResult:
    """Combined entity + structured extraction from Gemini."""

    entities: list[EntityResult] = field(default_factory=list)
    structured: StructuredExtraction = field(default_factory=StructuredExtraction)


# Minimal, schema-focused extraction prompt.
EXTRACTION_PROMPT = """Analyze the following text and extract structured information.
Return a JSON object with these fields:
- "core_issues": list of main problems, complaints, or concerns mentioned
- "key_attributes": list of specific features, products, or qualities discussed
- "action_items": list of any requested actions or next steps
- "topics": list of high-level topics or themes

Be concise. Each item should be a short phrase, not a full sentence.
If a field has no relevant content, return an empty list.

Text:
{text}
"""


ENTITY_EXTRACTION_PROMPT = """Extract all named entities from the following text.
Return a JSON object with a single field "entities" containing a list of objects,
each with "text" (the entity as it appears), "type" (one of: PERSON, ORG, GPE, LOC,
DATE, MONEY, PERCENT, CARDINAL, PRODUCT, EVENT, WORK_OF_ART, OTHER), and "salience"
(0.0 to 1.0, how central this entity is to the document).

Be thorough — extract every person, organization, location, date, product name,
monetary amount, and percentage mentioned.

Text:
{text}
"""


class GeminiExtractor:
    """Structured extraction using Gemini via Google AI Studio.

    JSON mode with temperature 0.1 for deterministic, parseable output.
    Complements SpaCy's pattern-based NER with semantic comprehension.
    """

    def __init__(self, api_key: str = None, model_name: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def extract(self, text: str) -> GeminiExtractionResult:
        """Extract both entities and structured info from a single text.

        Returns a GeminiExtractionResult with .entities (for needle evaluator
        compatibility) and .structured (for semantic extraction).
        """
        entities = self.extract_entities(text)
        structured = self.extract_structured(text)
        return GeminiExtractionResult(entities=entities, structured=structured)

    def extract_entities(self, text: str) -> list[EntityResult]:
        """Extract named entities via Gemini. Returns objects compatible
        with the needle-in-a-haystack evaluator (each has .text attribute).
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:4000])

        response = generate_with_retry(
            self.client,
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        try:
            parsed = json.loads(response.text)
            return [
                EntityResult(
                    text=e.get("text", ""),
                    type=e.get("type", "OTHER"),
                    salience=e.get("salience", 0.0),
                )
                for e in parsed.get("entities", [])
            ]
        except (json.JSONDecodeError, AttributeError):
            return []

    def extract_structured(self, text: str) -> StructuredExtraction:
        """Extract structured information from a single text.

        Input capped at 4000 chars to stay within context limits.
        """
        prompt = EXTRACTION_PROMPT.format(text=text[:4000])

        response = generate_with_retry(
            self.client,
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        try:
            parsed = json.loads(response.text)
            return StructuredExtraction(
                core_issues=parsed.get("core_issues", []),
                key_attributes=parsed.get("key_attributes", []),
                action_items=parsed.get("action_items", []),
                topics=parsed.get("topics", []),
                raw_response=response.text,
            )
        except (json.JSONDecodeError, AttributeError):
            return StructuredExtraction(raw_response=getattr(response, "text", ""))

    def extract_batch(self, texts: list[str]) -> list[GeminiExtractionResult]:
        """Process multiple texts sequentially. Production path: async or batch prediction."""
        return [self.extract(text) for text in texts]
