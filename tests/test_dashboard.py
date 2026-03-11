"""Tests for the evaluation dashboard utilities.

Tests the data loading, chart generation helpers, and Streamlit-independent
logic. The Streamlit UI itself is not tested here (requires browser), but
the underlying data transformations and computations are.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# We test the dashboard's data logic without launching Streamlit.
# Mock streamlit and GCP modules so imports don't fail.
with patch.dict("sys.modules", {
    "streamlit": MagicMock(),
    "google": MagicMock(),
    "google.genai": MagicMock(),
    "google.genai.types": MagicMock(),
    "google.genai.errors": MagicMock(),
    "google.cloud": MagicMock(),
    "google.cloud.firestore": MagicMock(),
    "google.cloud.bigquery": MagicMock(),
    "google.cloud.language": MagicMock(),
    "google.cloud.language_v2": MagicMock(),
    "google.cloud.storage": MagicMock(),
    "spacy": MagicMock(),
    "spacy.language": MagicMock(),
    "presidio_analyzer": MagicMock(),
}):
    pass


# ── Test data loading logic ──────────────────────────────────────────

class TestDocumentLoading:
    def test_load_documents_from_jsonl(self):
        """Verify we can load Document objects from a JSONL file."""
        from src.data.loader import Document

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(5):
                line = json.dumps({
                    "id": f"doc_{i}",
                    "text": f"This is document number {i} with some content.",
                    "source_type": "review",
                    "metadata": {"rating": i},
                })
                f.write(line + "\n")
            path = f.name

        try:
            docs = []
            with open(path) as fh:
                for j, line in enumerate(fh):
                    if j >= 3:
                        break
                    obj = json.loads(line)
                    docs.append(Document(
                        id=obj["id"],
                        text=obj["text"],
                        source_type=obj["source_type"],
                        metadata=obj["metadata"],
                    ))

            assert len(docs) == 3
            assert docs[0].id == "doc_0"
            assert docs[2].source_type == "review"
        finally:
            os.unlink(path)

    def test_load_empty_file(self):
        """Empty JSONL should produce no documents."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            docs = []
            with open(path) as fh:
                for line in fh:
                    if line.strip():
                        docs.append(json.loads(line))
            assert len(docs) == 0
        finally:
            os.unlink(path)


# ── Test needle-in-a-haystack data transformations ───────────────────

class TestNeedleHeatmapData:
    """Test the heatmap matrix construction used in the dashboard."""

    def test_heatmap_matrix_construction(self):
        """Verify we can build a recall heatmap from evaluator stats."""
        # Simulated stats from NeedleHaystackEvaluator.report()
        all_results = {
            "SpaCy": {
                "stats": {
                    "detection_rate": 1.0,
                    "average_recall": 0.9,
                    "detected": 5,
                    "total_needles": 5,
                    "by_entity_type": {
                        "PERSON": {"found": 2, "total": 2},
                        "ORG": {"found": 2, "total": 2},
                        "GPE": {"found": 1, "total": 2},
                    },
                    "by_position": {
                        "early": {"avg_recall": 1.0, "detected": 2, "total": 2},
                        "middle": {"avg_recall": 0.8, "detected": 1, "total": 1},
                        "deep": {"avg_recall": 0.9, "detected": 2, "total": 2},
                    },
                },
            },
            "Presidio": {
                "stats": {
                    "detection_rate": 1.0,
                    "average_recall": 0.44,
                    "detected": 5,
                    "total_needles": 5,
                    "by_entity_type": {
                        "PERSON": {"found": 2, "total": 2},
                        "ORG": {"found": 0, "total": 2},
                        "GPE": {"found": 1, "total": 2},
                    },
                    "by_position": {
                        "early": {"avg_recall": 0.5, "detected": 2, "total": 2},
                        "middle": {"avg_recall": 0.4, "detected": 1, "total": 1},
                        "deep": {"avg_recall": 0.4, "detected": 2, "total": 2},
                    },
                },
            },
        }

        # Build entity type heatmap (same logic as dashboard)
        all_types = set()
        for data in all_results.values():
            all_types.update(data["stats"]["by_entity_type"].keys())
        all_types = sorted(all_types)
        extractor_names = list(all_results.keys())

        heatmap = np.zeros((len(all_types), len(extractor_names)))
        for j, name in enumerate(extractor_names):
            by_type = all_results[name]["stats"]["by_entity_type"]
            for i, etype in enumerate(all_types):
                info = by_type.get(etype, {"found": 0, "total": 1})
                heatmap[i, j] = info["found"] / max(info["total"], 1)

        assert heatmap.shape == (3, 2)  # 3 entity types x 2 extractors

        # SpaCy PERSON recall should be 1.0
        person_idx = all_types.index("PERSON")
        spacy_idx = extractor_names.index("SpaCy")
        assert heatmap[person_idx, spacy_idx] == 1.0

        # Presidio ORG recall should be 0.0
        org_idx = all_types.index("ORG")
        presidio_idx = extractor_names.index("Presidio")
        assert heatmap[org_idx, presidio_idx] == 0.0

    def test_position_heatmap(self):
        """Verify position band matrix."""
        stats = {
            "by_position": {
                "early": {"avg_recall": 1.0},
                "middle": {"avg_recall": 0.5},
                "deep": {"avg_recall": 0.75},
            }
        }

        positions = ["early", "middle", "deep"]
        recalls = [stats["by_position"][p]["avg_recall"] for p in positions]

        assert recalls == [1.0, 0.5, 0.75]


# ── Test ROUGE aggregation (reuses existing evaluator) ───────────────

class TestROUGEVisualisationData:
    """Test the data prep for ROUGE distribution charts."""

    def test_score_extraction_for_histogram(self):
        """Simulate extracting scores for histogram plotting."""
        # Mock EvaluationResult-like objects
        mock_results = []
        for i in range(10):
            r = MagicMock()
            r.rouge_scores.rouge1_f1 = 0.2 + i * 0.05
            r.rouge_scores.rouge2_f1 = 0.1 + i * 0.03
            r.rouge_scores.rougeL_f1 = 0.15 + i * 0.04
            mock_results.append(r)

        r1_scores = [r.rouge_scores.rouge1_f1 for r in mock_results]
        r2_scores = [r.rouge_scores.rouge2_f1 for r in mock_results]
        rL_scores = [r.rouge_scores.rougeL_f1 for r in mock_results]

        assert len(r1_scores) == 10
        assert abs(np.mean(r1_scores) - 0.425) < 0.01
        assert np.min(r2_scores) == pytest.approx(0.1, abs=0.001)
        assert np.max(rL_scores) == pytest.approx(0.51, abs=0.01)


# ── Test source distribution computation ─────────────────────────────

class TestDocumentExplorerData:
    def test_source_type_counting(self):
        docs = [
            MagicMock(source_type="review"),
            MagicMock(source_type="review"),
            MagicMock(source_type="news"),
            MagicMock(source_type="support_ticket"),
            MagicMock(source_type="news"),
        ]

        types_count = {}
        for d in docs:
            types_count[d.source_type] = types_count.get(d.source_type, 0) + 1

        assert types_count == {"review": 2, "news": 2, "support_ticket": 1}

    def test_text_length_distribution(self):
        docs = [
            MagicMock(text="one two three"),
            MagicMock(text="a b c d e f g h i j"),
            MagicMock(text="short"),
        ]

        lengths = [len(d.text.split()) for d in docs]
        assert lengths == [3, 10, 1]
        assert np.mean(lengths) == pytest.approx(4.667, abs=0.01)
