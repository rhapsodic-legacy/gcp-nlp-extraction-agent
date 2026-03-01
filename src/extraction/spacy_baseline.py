"""SpaCy-based NER as a baseline comparison to GCP Natural Language API.

You never really know how good your fancy managed service is until you
compare it to something simpler. That's what this module is — the baseline.
SpaCy gives us a local, free, fast NER engine that we can run without any
cloud credentials. It's like comparing your shiny new calculator chip
against doing the math by hand: you want to know the managed service is
actually earning its keep.

SpaCy's en_core_web_sm model is small and fast. It won't win accuracy
contests against GCP's NL API (which has Google-scale training data behind
it), but it's a perfectly respectable baseline. And it runs offline, which
is handy for development when you don't want every test hitting a paid API.

For even better accuracy you can swap in en_core_web_trf (transformer-based),
but that's slower and heavier. For baseline comparison purposes, the small
model tells you what you need to know.
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
    """Everything SpaCy found in a document — entities and noun chunks.

    Noun chunks are a nice bonus that SpaCy gives us for free. They're
    noun phrases like "the battery life" or "customer support team" —
    useful for understanding what topics a document is about even if
    they're not formally named entities.
    """

    entities: list[SpacyEntity] = field(default_factory=list)
    noun_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "noun_chunks": self.noun_chunks,
        }


class SpacyExtractor:
    """Local NER using SpaCy — the trusty baseline comparison.

    Default model: en_core_web_sm (fast, lightweight, good enough for
    comparison). If you want the big guns, pass model_name="en_core_web_trf"
    for transformer-based extraction — it's significantly more accurate
    but also significantly slower. Pick your trade-off.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp: Language = spacy.load(model_name)
        except OSError:
            # Auto-download if the model isn't installed yet.
            # I love that SpaCy makes this easy.
            print(f"Downloading SpaCy model '{model_name}'...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

    def extract(self, text: str) -> SpacyExtractionResult:
        """Extract entities and noun chunks from a single text.

        One document in, structured results out. SpaCy's pipeline handles
        tokenization, POS tagging, and NER all in one pass — efficient design.
        """
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
        """Batch extraction using SpaCy's pipe() for efficiency.

        SpaCy's nlp.pipe() is smart about batching — it processes multiple
        documents in parallel under the hood. This is one of those cases
        where the library does the optimization for you. I appreciate that.
        """
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
