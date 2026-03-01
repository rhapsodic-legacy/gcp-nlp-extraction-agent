"""Presidio-based PII extraction as a second baseline comparison.

Microsoft Presidio is purpose-built for PII detection: person names, emails,
phone numbers, addresses, and financial identifiers. Unlike SpaCy (general NER)
or Gemini (semantic comprehension), Presidio combines regex patterns, NLP
models, and context-aware rules specifically tuned for personally identifiable
information.

This makes it a useful third benchmark point:
  - SpaCy: fast, broad NER (orgs, dates, locations, monetary values)
  - Presidio: specialised PII detection (persons, emails, phones, addresses)
  - Gemini: semantic comprehension across all entity types

Presidio uses SpaCy under the hood for its NLP engine, but adds PII-specific
recognisers (regex patterns for credit cards, phone formats, etc.) that pure
NER models do not cover. The comparison highlights where domain-specific tools
outperform general-purpose ones, and vice versa.
"""

from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine


# Map Presidio entity types to the types used in needle-in-a-haystack evaluation
PRESIDIO_TO_NEEDLE_TYPE = {
    "PERSON": "PERSON",
    "LOCATION": "GPE",
    "NRP": "GPE",
    "DATE_TIME": "DATE",
    "PHONE_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "CREDIT_CARD": "FINANCIAL",
    "US_SSN": "PII",
    "IP_ADDRESS": "IP",
}


@dataclass
class PresidioEntity:
    """A single entity detected by Presidio."""

    text: str
    type: str
    score: float
    start: int
    end: int

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "type": self.type,
            "score": self.score,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class PresidioExtractionResult:
    """Extraction result from Presidio, compatible with the needle evaluator.

    The evaluator checks `extraction.entities` for objects with a `.text`
    attribute, so this matches the same interface as SpacyExtractionResult.
    """

    entities: list[PresidioEntity] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"entities": [e.to_dict() for e in self.entities]}


class PresidioExtractor:
    """PII-focused entity extraction using Microsoft Presidio.

    Uses the default AnalyzerEngine with the en_core_web_lg SpaCy model
    (falls back to en_core_web_sm if lg is not installed). Detects person
    names, locations, dates, phone numbers, emails, and other PII types.
    """

    def __init__(self):
        self.analyzer = AnalyzerEngine()

    def extract(self, text: str) -> PresidioExtractionResult:
        """Extract PII entities from text.

        Returns a PresidioExtractionResult compatible with the
        NeedleHaystackEvaluator interface.
        """
        results = self.analyzer.analyze(
            text=text,
            language="en",
        )

        entities = []
        for result in results:
            entity_text = text[result.start:result.end]
            entities.append(PresidioEntity(
                text=entity_text,
                type=result.entity_type,
                score=result.score,
                start=result.start,
                end=result.end,
            ))

        return PresidioExtractionResult(entities=entities)

    def extract_batch(self, texts: list[str]) -> list[PresidioExtractionResult]:
        """Batch extraction across multiple texts."""
        return [self.extract(text) for text in texts]
