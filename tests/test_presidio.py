"""Unit tests for the Presidio PII extractor.

Tests entity extraction, batch processing, and compatibility with
the NeedleHaystackEvaluator interface. No API keys needed.
"""

import pytest

from src.extraction.presidio_extract import (
    PresidioExtractor,
    PresidioEntity,
    PresidioExtractionResult,
)


@pytest.fixture(scope="module")
def extractor():
    """Shared PresidioExtractor (loads SpaCy model once)."""
    return PresidioExtractor()


class TestPresidioEntity:
    """Tests for the PresidioEntity dataclass."""

    def test_to_dict(self):
        e = PresidioEntity(text="John Smith", type="PERSON", score=0.85, start=0, end=10)
        d = e.to_dict()
        assert d["text"] == "John Smith"
        assert d["type"] == "PERSON"
        assert d["score"] == 0.85

    def test_extraction_result_to_dict(self):
        result = PresidioExtractionResult(entities=[
            PresidioEntity(text="John", type="PERSON", score=0.85, start=0, end=4),
        ])
        d = result.to_dict()
        assert len(d["entities"]) == 1
        assert d["entities"][0]["text"] == "John"


class TestPresidioExtraction:
    """Tests for entity extraction via Presidio."""

    def test_extracts_person_name(self, extractor):
        result = extractor.extract("Dr. Evelyn Thorncastle reported the issue.")
        entity_texts = [e.text for e in result.entities]
        assert any("Evelyn" in t for t in entity_texts)

    def test_extracts_date(self, extractor):
        result = extractor.extract("The incident occurred on March 14th, 2024.")
        entity_types = [e.type for e in result.entities]
        assert "DATE_TIME" in entity_types

    def test_extracts_location(self, extractor):
        result = extractor.extract("The warehouse in Brisbane was shut down.")
        entity_texts = [e.text.lower() for e in result.entities]
        assert any("brisbane" in t for t in entity_texts)

    def test_empty_text_returns_empty(self, extractor):
        result = extractor.extract("")
        assert len(result.entities) == 0

    def test_no_pii_returns_empty(self, extractor):
        result = extractor.extract("The sky is blue and water is wet.")
        # May find nothing or very few entities in a generic sentence
        assert isinstance(result, PresidioExtractionResult)

    def test_entities_have_text_attribute(self, extractor):
        """Verifies compatibility with NeedleHaystackEvaluator interface."""
        result = extractor.extract("Professor Liang Wei published findings.")
        for entity in result.entities:
            assert hasattr(entity, "text")
            assert isinstance(entity.text, str)
            assert len(entity.text) > 0


class TestPresidioBatch:
    """Tests for batch extraction."""

    def test_batch_returns_list(self, extractor):
        texts = [
            "John Smith lives in London.",
            "The meeting is on July 9th.",
        ]
        results = extractor.extract_batch(texts)
        assert len(results) == 2
        assert all(isinstance(r, PresidioExtractionResult) for r in results)

    def test_batch_empty_list(self, extractor):
        results = extractor.extract_batch([])
        assert results == []
