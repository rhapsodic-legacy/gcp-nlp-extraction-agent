"""SpaCy-based NER baseline for comparison against GCP-native extractors.

Local, offline NER using en_core_web_sm. Swap in en_core_web_trf for
transformer-based accuracy at the cost of speed.
"""

from dataclasses import dataclass, field

import spacy
from spacy.language import Language


@dataclass
class SpacyEntity:
    """A single entity found by SpaCy — the basic unit of NER output."""

    text: str
    label: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class SpacyExtractionResult:
    """Entities and noun chunks extracted from a document."""

    entities: list[SpacyEntity] = field(default_factory=list)
    noun_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "noun_chunks": self.noun_chunks,
        }


class SpacyExtractor:
    """Local NER baseline using SpaCy.

    Default: en_core_web_sm. Use en_core_web_trf for transformer-based
    accuracy (slower, heavier).
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp: Language = spacy.load(model_name)
        except OSError:
            # Auto-download if not installed
            print(f"Downloading SpaCy model '{model_name}'...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

    def extract(self, text: str) -> SpacyExtractionResult:
        """Extract entities and noun chunks from a single text."""
        doc = self.nlp(text)

        entities = [
            SpacyEntity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
            )
            for ent in doc.ents
        ]

        noun_chunks = [chunk.text for chunk in doc.noun_chunks]

        return SpacyExtractionResult(entities=entities, noun_chunks=noun_chunks)

    def extract_batch(self, texts: list[str]) -> list[SpacyExtractionResult]:
        """Batch extraction using SpaCy's pipe() for efficiency."""
        results = []
        for doc in self.nlp.pipe(texts, batch_size=32):
            entities = [
                SpacyEntity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                )
                for ent in doc.ents
            ]
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            results.append(SpacyExtractionResult(entities=entities, noun_chunks=noun_chunks))
        return results
