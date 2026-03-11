"""Ensemble extraction pipeline combining SpaCy, Presidio, and Gemini.

Runs all three extractors in parallel, deduplicates entities via fuzzy
matching, and assigns confidence scores based on cross-extractor agreement.
Compatible with the needle-in-a-haystack evaluator interface.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from .spacy_baseline import SpacyExtractor, SpacyExtractionResult
from .presidio_extract import PresidioExtractor, PresidioExtractionResult, PRESIDIO_TO_NEEDLE_TYPE
from .vertex_extract import GeminiExtractor, GeminiExtractionResult


# Normalise extractor-specific type labels to a shared vocabulary.
TYPE_ALIASES = {
    # Presidio types
    "LOCATION": "GPE",
    "NRP": "GPE",
    "DATE_TIME": "DATE",
    "PHONE_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "CREDIT_CARD": "FINANCIAL",
    "US_SSN": "PII",
    "IP_ADDRESS": "IP",
    # SpaCy types that overlap
    "LOC": "GPE",
    "FAC": "GPE",
    "NORP": "GPE",
    "ORDINAL": "CARDINAL",
    "QUANTITY": "CARDINAL",
}


@dataclass
class EnsembleEntity:
    """A deduplicated entity with provenance and confidence metadata."""

    text: str
    type: str
    confidence: float  # 0.0-1.0 based on cross-extractor agreement
    sources: list[str] = field(default_factory=list)  # e.g. ["spacy", "gemini"]
    salience: float = 0.0
    variants: list[str] = field(default_factory=list)  # surface forms from each extractor

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "type": self.type,
            "confidence": round(self.confidence, 3),
            "sources": self.sources,
            "salience": round(self.salience, 3),
            "variants": self.variants,
        }


@dataclass
class EnsembleExtractionResult:
    """Combined extraction from all backends with agreement-based scoring."""

    entities: list[EnsembleEntity] = field(default_factory=list)
    per_extractor: dict = field(default_factory=dict)  # raw results keyed by name

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "extractor_counts": {
                name: len(getattr(res, "entities", []))
                for name, res in self.per_extractor.items()
            },
        }


def _normalise_type(entity_type: str) -> str:
    """Map extractor-specific type labels to a shared vocabulary."""
    return TYPE_ALIASES.get(entity_type, entity_type)


def _fuzzy_match(a: str, b: str, threshold: float = 0.75) -> bool:
    """Bidirectional substring or fuzzy match between two entity texts."""
    a_low, b_low = a.lower().strip(), b.lower().strip()

    # Exact match
    if a_low == b_low:
        return True

    # Substring containment (handles "Dr. Evelyn Thorncastle" vs "Evelyn Thorncastle")
    if a_low in b_low or b_low in a_low:
        return True

    # SequenceMatcher for fuzzy overlap
    return SequenceMatcher(None, a_low, b_low).ratio() >= threshold


class EnsembleExtractor:
    """Runs SpaCy, Presidio, and Gemini extraction in parallel,
    deduplicates via fuzzy matching, and scores by cross-extractor agreement.

    Confidence scoring:
      - 1 extractor found it: 0.33 (low confidence)
      - 2 extractors found it: 0.67 (medium)
      - 3 extractors found it: 1.00 (high)

    Compatible with the NeedleHaystackEvaluator (exposes .entities with .text).
    """

    def __init__(
        self,
        api_key: str = None,
        spacy_model: str = "en_core_web_sm",
        enable_gemini: bool = True,
        fuzzy_threshold: float = 0.75,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.enable_gemini = enable_gemini

        # Initialise extractors
        self._spacy = SpacyExtractor(model_name=spacy_model)
        self._presidio = PresidioExtractor()
        self._gemini = None
        if enable_gemini:
            self._gemini = GeminiExtractor(api_key=api_key)

    def extract(self, text: str) -> EnsembleExtractionResult:
        """Run all extractors and merge results."""
        raw_results = self._run_extractors(text)
        merged = self._merge_entities(raw_results)
        return EnsembleExtractionResult(entities=merged, per_extractor=raw_results)

    def extract_batch(self, texts: list[str]) -> list[EnsembleExtractionResult]:
        """Batch extraction across multiple texts."""
        return [self.extract(text) for text in texts]

    def _run_extractors(self, text: str) -> dict:
        """Run extractors concurrently and collect results."""
        results = {}

        def run_spacy():
            return "spacy", self._spacy.extract(text)

        def run_presidio():
            return "presidio", self._presidio.extract(text)

        def run_gemini():
            return "gemini", self._gemini.extract(text)

        tasks = [run_spacy, run_presidio]
        if self._gemini:
            tasks.append(run_gemini)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(fn): fn for fn in tasks}
            for future in as_completed(futures):
                try:
                    name, result = future.result()
                    results[name] = result
                except Exception as e:
                    # Log but don't block — partial results are still valuable.
                    fn = futures[future]
                    print(f"Ensemble: {fn.__name__} failed: {e}")

        return results

    def _merge_entities(self, raw_results: dict) -> list[EnsembleEntity]:
        """Deduplicate and score entities across extractors."""
        # Collect all (text, type, source, salience) tuples
        candidates: list[tuple[str, str, str, float]] = []

        for name, result in raw_results.items():
            if not hasattr(result, "entities"):
                continue
            for entity in result.entities:
                etype = _normalise_type(
                    getattr(entity, "label", None)
                    or getattr(entity, "type", "OTHER")
                )
                salience = getattr(entity, "salience", 0.0)
                candidates.append((entity.text, etype, name, salience))

        # Greedy clustering: assign each candidate to the first matching cluster.
        clusters: list[dict] = []

        for text, etype, source, salience in candidates:
            matched = False
            for cluster in clusters:
                # Same normalised type + fuzzy text match → merge
                if cluster["type"] == etype and _fuzzy_match(
                    text, cluster["canonical"], self.fuzzy_threshold
                ):
                    if source not in cluster["sources"]:
                        cluster["sources"].append(source)
                    if text not in cluster["variants"]:
                        cluster["variants"].append(text)
                    cluster["salience"] = max(cluster["salience"], salience)
                    matched = True
                    break

            if not matched:
                clusters.append({
                    "canonical": text,
                    "type": etype,
                    "sources": [source],
                    "variants": [text],
                    "salience": salience,
                })

        # Convert clusters to EnsembleEntity with confidence scores
        num_extractors = len(raw_results)
        entities = []
        for cluster in clusters:
            # Pick the longest variant as canonical text (most complete form)
            canonical = max(cluster["variants"], key=len)
            confidence = len(cluster["sources"]) / max(num_extractors, 1)

            entities.append(EnsembleEntity(
                text=canonical,
                type=cluster["type"],
                confidence=confidence,
                sources=sorted(cluster["sources"]),
                salience=cluster["salience"],
                variants=sorted(set(cluster["variants"])),
            ))

        # Sort by confidence (high first), then salience
        entities.sort(key=lambda e: (-e.confidence, -e.salience))
        return entities
