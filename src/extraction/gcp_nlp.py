"""Entity extraction and sentiment analysis via GCP Natural Language API (v2).

Managed NER and sentiment analysis. Returns entities with types, salience,
and document-level sentiment. All outputs wrapped in dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional

from google.cloud import language_v2


@dataclass
class Entity:
    """A named entity with type, salience (0.0-1.0), and optional sentiment."""

    text: str
    type: str
    salience: float = 0.0
    sentiment_score: Optional[float] = None
    sentiment_magnitude: Optional[float] = None
    mentions: int = 1

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "type": self.type,
            "salience": self.salience,
            "sentiment_score": self.sentiment_score,
            "sentiment_magnitude": self.sentiment_magnitude,
            "mentions": self.mentions,
        }


@dataclass
class SentimentResult:
    """Document-level sentiment. Score: -1.0 to 1.0. Magnitude: emotional intensity."""

    score: float
    magnitude: float

    def to_dict(self) -> dict:
        return {"score": self.score, "magnitude": self.magnitude}


@dataclass
class ExtractionResult:
    """Combined entity + sentiment extraction result."""

    entities: list[Entity] = field(default_factory=list)
    sentiment: Optional[SentimentResult] = None
    language: str = "en"

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "language": self.language,
        }


# Map GCP entity type enums to readable strings.
_ENTITY_TYPE_MAP = {
    language_v2.Entity.Type.UNKNOWN: "UNKNOWN",
    language_v2.Entity.Type.PERSON: "PERSON",
    language_v2.Entity.Type.LOCATION: "LOCATION",
    language_v2.Entity.Type.ORGANIZATION: "ORGANIZATION",
    language_v2.Entity.Type.EVENT: "EVENT",
    language_v2.Entity.Type.WORK_OF_ART: "WORK_OF_ART",
    language_v2.Entity.Type.CONSUMER_GOOD: "CONSUMER_GOOD",
    language_v2.Entity.Type.OTHER: "OTHER",
    language_v2.Entity.Type.PHONE_NUMBER: "PHONE_NUMBER",
    language_v2.Entity.Type.ADDRESS: "ADDRESS",
    language_v2.Entity.Type.DATE: "DATE",
    language_v2.Entity.Type.NUMBER: "NUMBER",
    language_v2.Entity.Type.PRICE: "PRICE",
}


class GCPEntityExtractor:
    """Entity and sentiment extraction via GCP Natural Language API v2.

    Sequential per-document calls. Production: parallelise or use batch endpoints.
    """

    def __init__(self):
        self.client = language_v2.LanguageServiceClient()

    def _make_document(self, text: str) -> language_v2.Document:
        """Wrap raw text into the API's Document format."""
        return language_v2.Document(
            content=text,
            type_=language_v2.Document.Type.PLAIN_TEXT,
            language_code="en",
        )

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract named entities with type labels and salience scores."""
        doc = self._make_document(text)
        response = self.client.analyze_entities(
            request={"document": doc, "encoding_type": language_v2.EncodingType.UTF8}
        )

        entities = []
        for entity in response.entities:
            entities.append(
                Entity(
                    text=entity.name,
                    type=_ENTITY_TYPE_MAP.get(entity.type_, "UNKNOWN"),
                    salience=entity.salience,
                    mentions=len(entity.mentions),
                )
            )
        return entities

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Document-level sentiment analysis."""
        doc = self._make_document(text)
        response = self.client.analyze_sentiment(
            request={"document": doc, "encoding_type": language_v2.EncodingType.UTF8}
        )
        return SentimentResult(
            score=response.document_sentiment.score,
            magnitude=response.document_sentiment.magnitude,
        )

    def extract(self, text: str) -> ExtractionResult:
        """Full extraction: entities + sentiment in one call."""
        entities = self.extract_entities(text)
        sentiment = self.analyze_sentiment(text)

        return ExtractionResult(
            entities=entities,
            sentiment=sentiment,
        )

    def extract_batch(self, texts: list[str]) -> list[ExtractionResult]:
        """Process multiple texts sequentially. Production: async or batch API."""
        return [self.extract(text) for text in texts]
