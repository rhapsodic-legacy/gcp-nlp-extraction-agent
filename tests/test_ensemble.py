"""Tests for the ensemble extraction pipeline.

Tests deduplication, fuzzy matching, confidence scoring, and the merge
logic — all without live API calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.extraction.ensemble import (
    EnsembleExtractor,
    EnsembleEntity,
    EnsembleExtractionResult,
    _fuzzy_match,
    _normalise_type,
)


# ── Unit tests for helper functions ──────────────────────────────────

class TestFuzzyMatch:
    def test_exact_match(self):
        assert _fuzzy_match("Apple Inc", "Apple Inc")

    def test_case_insensitive(self):
        assert _fuzzy_match("apple inc", "Apple Inc")

    def test_substring_containment(self):
        assert _fuzzy_match("Evelyn Thorncastle", "Dr. Evelyn Thorncastle")

    def test_reverse_substring(self):
        assert _fuzzy_match("Dr. Evelyn Thorncastle", "Evelyn Thorncastle")

    def test_no_match(self):
        assert not _fuzzy_match("Apple Inc", "Google LLC")

    def test_partial_overlap(self):
        # "Brisbane" vs "Brisbane warehouse" — substring match
        assert _fuzzy_match("Brisbane", "Brisbane warehouse")

    def test_threshold(self):
        # Below default threshold
        assert not _fuzzy_match("cat", "catastrophe", threshold=0.9)


class TestNormaliseType:
    def test_location_to_gpe(self):
        assert _normalise_type("LOCATION") == "GPE"

    def test_date_time_to_date(self):
        assert _normalise_type("DATE_TIME") == "DATE"

    def test_passthrough(self):
        assert _normalise_type("PERSON") == "PERSON"
        assert _normalise_type("ORG") == "ORG"

    def test_spacy_loc(self):
        assert _normalise_type("LOC") == "GPE"


# ── Integration tests with mocked extractors ─────────────────────────

class TestEnsembleMerge:
    """Test the merge logic by providing pre-built raw results."""

    def _make_spacy_entity(self, text, label):
        e = MagicMock()
        e.text = text
        e.label = label
        e.label_ = label
        e.start_char = 0
        e.end_char = len(text)
        return e

    def _make_presidio_entity(self, text, etype, score=0.85):
        e = MagicMock()
        e.text = text
        e.type = etype
        e.score = score
        return e

    def _make_gemini_entity(self, text, etype, salience=0.5):
        e = MagicMock()
        e.text = text
        e.type = etype
        e.salience = salience
        return e

    def test_deduplication_same_entity(self):
        """Same entity from multiple extractors should be merged."""
        extractor = EnsembleExtractor.__new__(EnsembleExtractor)
        extractor.fuzzy_threshold = 0.75

        spacy_result = MagicMock()
        spacy_result.entities = [self._make_spacy_entity("Apple Inc", "ORG")]

        gemini_result = MagicMock()
        gemini_result.entities = [self._make_gemini_entity("Apple Inc", "ORG")]

        raw = {"spacy": spacy_result, "gemini": gemini_result}
        merged = extractor._merge_entities(raw)

        assert len(merged) == 1
        assert merged[0].text == "Apple Inc"
        assert "spacy" in merged[0].sources
        assert "gemini" in merged[0].sources
        assert merged[0].confidence == 1.0

    def test_different_entities_not_merged(self):
        """Different entities should remain separate."""
        extractor = EnsembleExtractor.__new__(EnsembleExtractor)
        extractor.fuzzy_threshold = 0.75

        spacy_result = MagicMock()
        spacy_result.entities = [
            self._make_spacy_entity("Apple Inc", "ORG"),
            self._make_spacy_entity("New York", "GPE"),
        ]

        raw = {"spacy": spacy_result}
        merged = extractor._merge_entities(raw)

        assert len(merged) == 2

    def test_partial_name_match(self):
        """'Evelyn Thorncastle' and 'Dr. Evelyn Thorncastle' should merge."""
        extractor = EnsembleExtractor.__new__(EnsembleExtractor)
        extractor.fuzzy_threshold = 0.75

        spacy_result = MagicMock()
        spacy_result.entities = [self._make_spacy_entity("Evelyn Thorncastle", "PERSON")]

        gemini_result = MagicMock()
        gemini_result.entities = [self._make_gemini_entity("Dr. Evelyn Thorncastle", "PERSON")]

        raw = {"spacy": spacy_result, "gemini": gemini_result}
        merged = extractor._merge_entities(raw)

        # Should merge into one entity
        person_entities = [e for e in merged if e.type == "PERSON"]
        assert len(person_entities) == 1
        # Canonical should be the longer form
        assert "Dr. Evelyn Thorncastle" in person_entities[0].text

    def test_confidence_scoring_single_source(self):
        """Entity from 1 of 3 extractors should have ~0.33 confidence."""
        extractor = EnsembleExtractor.__new__(EnsembleExtractor)
        extractor.fuzzy_threshold = 0.75

        spacy_result = MagicMock()
        spacy_result.entities = [self._make_spacy_entity("Apple", "ORG")]

        presidio_result = MagicMock()
        presidio_result.entities = []

        gemini_result = MagicMock()
        gemini_result.entities = []

        raw = {"spacy": spacy_result, "presidio": presidio_result, "gemini": gemini_result}
        merged = extractor._merge_entities(raw)

        assert len(merged) == 1
        assert abs(merged[0].confidence - 1 / 3) < 0.01

    def test_type_normalisation_in_merge(self):
        """LOCATION from Presidio and GPE from SpaCy should merge."""
        extractor = EnsembleExtractor.__new__(EnsembleExtractor)
        extractor.fuzzy_threshold = 0.75

        spacy_result = MagicMock()
        spacy_result.entities = [self._make_spacy_entity("Brisbane", "GPE")]

        presidio_result = MagicMock()
        presidio_result.entities = [self._make_presidio_entity("Brisbane", "LOCATION")]

        raw = {"spacy": spacy_result, "presidio": presidio_result}
        merged = extractor._merge_entities(raw)

        gpe_entities = [e for e in merged if e.type == "GPE"]
        assert len(gpe_entities) == 1
        assert len(gpe_entities[0].sources) == 2

    def test_sorted_by_confidence(self):
        """Results should be sorted by confidence descending."""
        extractor = EnsembleExtractor.__new__(EnsembleExtractor)
        extractor.fuzzy_threshold = 0.75

        spacy_result = MagicMock()
        spacy_result.entities = [
            self._make_spacy_entity("Apple", "ORG"),
            self._make_spacy_entity("New York", "GPE"),
        ]

        gemini_result = MagicMock()
        gemini_result.entities = [
            self._make_gemini_entity("Apple", "ORG"),
        ]

        raw = {"spacy": spacy_result, "gemini": gemini_result}
        merged = extractor._merge_entities(raw)

        # Apple (2 sources) should come before New York (1 source)
        assert merged[0].text == "Apple"
        assert merged[0].confidence > merged[1].confidence


class TestEnsembleExtractionResult:
    def test_to_dict(self):
        entity = EnsembleEntity(
            text="Apple Inc",
            type="ORG",
            confidence=0.667,
            sources=["spacy", "gemini"],
            salience=0.8,
            variants=["Apple Inc", "Apple"],
        )
        result = EnsembleExtractionResult(entities=[entity], per_extractor={})
        d = result.to_dict()
        assert len(d["entities"]) == 1
        assert d["entities"][0]["confidence"] == 0.667
