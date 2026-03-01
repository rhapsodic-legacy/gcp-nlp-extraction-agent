"""Entity extraction and sentiment analysis via GCP Natural Language API.

This module is the workhorse of the GCP-native extraction path. The Natural
Language API is one of those managed services that just makes you smile —
you hand it text, it hands you back entities with types, salience scores,
and sentiment. No training, no fine-tuning, no fiddling with hyperparameters.

It's like having a really good multimeter: you probe the signal and it tells
you exactly what's there. The v2 API gives us entity analysis (who, what,
where, when) and sentiment analysis (how does the author feel about it) in
a clean, reliable interface.

We wrap it all in dataclasses because I believe in structured data the way
I believe in labeled wires — you should always know what you're looking at.
"""

from dataclasses import dataclass, field
from typing import Optional

from google.cloud import language_v2


@dataclass
class Entity:
    """A named entity pulled out of text — a person, place, org, date, etc.

    Salience tells you how important this entity is to the document overall
    (0.0 to 1.0). It's the NLP equivalent of signal strength — high salience
    means this entity is central to what the text is about.
    """

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
    """Document-level sentiment — the overall emotional tone.

    Score ranges from -1.0 (very negative) to 1.0 (very positive).
    Magnitude tells you how much emotional content there is regardless
    of direction — a rant and a love letter can both have high magnitude
    but opposite scores.
    """

    score: float
    magnitude: float

    def to_dict(self) -> dict:
        return {"score": self.score, "magnitude": self.magnitude}


@dataclass
class ExtractionResult:
    """Combined output from a single document extraction — entities + sentiment.

    One extraction call, one result object. Everything you pulled out of
    the document in one tidy package.
    """

    entities: list[Entity] = field(default_factory=list)
    sentiment: Optional[SentimentResult] = None
    language: str = "en"

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "language": self.language,
        }


# Map GCP's numeric entity type enums to human-readable strings.
# Because nobody wants to debug "entity type 7" at 2am.
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
    """Extract entities and sentiment using GCP Natural Language API.

    This is the managed-service extraction path. The NL API handles the
    heavy lifting of NER (Named Entity Recognition) and sentiment analysis
    so we don't have to train or host any models ourselves. It uses the v2
    API which supports entity analysis and sentiment analysis.

    For the MVP, we call these sequentially per document. In production you'd
    want to parallelize or use batch endpoints — but for prototyping, sequential
    is simple and debuggable. Get it right first, make it fast second.
    """

    def __init__(self):
        self.client = language_v2.LanguageServiceClient()

    def _make_document(self, text: str) -> language_v2.Document:
        """Wrap raw text into the API's Document format. Just plumbing."""
        return language_v2.Document(
            content=text,
            type_=language_v2.Document.Type.PLAIN_TEXT,
            language_code="en",
        )

    def extract_entities(self, text: str) -> list[Entity]:
        """Pull named entities out of text — the who, what, where, when.

        Returns a list of Entity objects with type labels and salience scores.
        The API is surprisingly good at this, even on messy review text.
        """
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
        """Get the overall emotional tone of a document.

        Score tells you positive vs. negative. Magnitude tells you
        how strongly the author feels about it. A neutral review has
        low magnitude; an angry rant has high magnitude and negative score.
        """
        doc = self._make_document(text)
        response = self.client.analyze_sentiment(
            request={"document": doc, "encoding_type": language_v2.EncodingType.UTF8}
        )
        return SentimentResult(
            score=response.document_sentiment.score,
            magnitude=response.document_sentiment.magnitude,
        )

    def extract(self, text: str) -> ExtractionResult:
        """Full extraction: entities + sentiment in one call.

        This is the convenience method that does everything. Two API calls
        under the hood (analyze_entities + analyze_sentiment), one result
        object out. Most of the time, this is the one you want.
        """
        entities = self.extract_entities(text)
        sentiment = self.analyze_sentiment(text)

        return ExtractionResult(
            entities=entities,
            sentiment=sentiment,
        )

    def extract_batch(self, texts: list[str]) -> list[ExtractionResult]:
        """Process multiple texts. Sequential for now — parallelism comes later.

        In a production system you'd want async calls or the batch API,
        but for an MVP, sequential is honest and debuggable. Premature
        optimization is the root of all evil (Knuth said that, not me,
        but I agree with him completely).
        """
        return [self.extract(text) for text in texts]
