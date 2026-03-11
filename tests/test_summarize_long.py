"""Tests for recursive map-reduce summarization.

Tests chunking logic, map-reduce pipeline, and recursive reduction
with fully mocked Gemini calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.summarization.vertex_summarize import (
    chunk_text,
    GeminiSummarizer,
    DEFAULT_CHUNK_SIZE,
    LONG_DOC_THRESHOLD,
)


# ── Chunking tests ──────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "This is short."
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, chunk_size=35)
        # Should split at ". " boundaries
        assert len(chunks) >= 2
        # Each chunk should end at a sentence boundary (except possibly the last)
        for chunk in chunks[:-1]:
            assert chunk.rstrip().endswith(".")

    def test_preserves_all_content(self):
        text = "A" * 100 + ". " + "B" * 100 + ". " + "C" * 100
        chunks = chunk_text(text, chunk_size=120)
        reconstructed = "".join(chunks)
        assert reconstructed == text

    def test_handles_no_sentence_boundaries(self):
        """Text with no periods should still be chunked (hard split)."""
        text = "word " * 500  # 2500 chars, no periods
        chunks = chunk_text(text, chunk_size=200)
        assert len(chunks) > 1
        reconstructed = "".join(chunks)
        assert reconstructed == text

    def test_respects_newline_fallback(self):
        text = "First paragraph\nSecond paragraph\nThird paragraph"
        chunks = chunk_text(text, chunk_size=20)
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_exact_chunk_size(self):
        text = "X" * 100
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 1


# ── Summarizer routing tests ────────────────────────────────────────

class TestSummarizeRouting:
    """Test that summarize() routes to short vs long path correctly."""

    @patch("src.summarization.vertex_summarize.generate_with_retry")
    @patch("src.summarization.vertex_summarize.genai")
    def test_short_text_uses_direct_path(self, mock_genai, mock_gen):
        mock_response = MagicMock()
        mock_response.text = "A short summary."
        mock_gen.return_value = mock_response

        summarizer = GeminiSummarizer(api_key="fake")
        result = summarizer.summarize("Short text." * 10)

        assert result == "A short summary."
        # Should be called exactly once (direct path)
        assert mock_gen.call_count == 1

    @patch("src.summarization.vertex_summarize.generate_with_retry")
    @patch("src.summarization.vertex_summarize.genai")
    def test_long_text_uses_map_reduce(self, mock_genai, mock_gen):
        mock_response = MagicMock()
        mock_response.text = "A chunk summary."
        mock_gen.return_value = mock_response

        summarizer = GeminiSummarizer(api_key="fake", chunk_size=1000)
        long_text = "This is a sentence. " * 1000  # ~20000 chars

        result = summarizer.summarize(long_text)

        # Should be called multiple times: once per chunk + synthesis
        assert mock_gen.call_count > 1

    @patch("src.summarization.vertex_summarize.generate_with_retry")
    @patch("src.summarization.vertex_summarize.genai")
    def test_threshold_boundary(self, mock_genai, mock_gen):
        """Text exactly at the threshold should use the short path."""
        mock_response = MagicMock()
        mock_response.text = "Summary."
        mock_gen.return_value = mock_response

        summarizer = GeminiSummarizer(api_key="fake")
        text = "X" * LONG_DOC_THRESHOLD  # exactly at threshold

        summarizer.summarize(text)
        # Short path = 1 call
        assert mock_gen.call_count == 1


# ── Map-reduce pipeline tests ───────────────────────────────────────

class TestMapReduce:
    @patch("src.summarization.vertex_summarize.generate_with_retry")
    @patch("src.summarization.vertex_summarize.genai")
    def test_summarize_long_produces_result(self, mock_genai, mock_gen):
        """Verify the full map-reduce pipeline produces a string result."""
        call_count = [0]

        def fake_generate(*args, **kwargs):
            call_count[0] += 1
            resp = MagicMock()
            resp.text = f"Summary for call {call_count[0]}."
            return resp

        mock_gen.side_effect = fake_generate
        summarizer = GeminiSummarizer(api_key="fake", chunk_size=500)

        long_text = "This is test content. " * 200  # ~4400 chars
        result = summarizer.summarize_long(long_text)

        assert isinstance(result, str)
        assert len(result) > 0

    @patch("src.summarization.vertex_summarize.generate_with_retry")
    @patch("src.summarization.vertex_summarize.genai")
    def test_map_phase_calls_per_chunk(self, mock_genai, mock_gen):
        """Each chunk should get its own summarization call."""
        mock_response = MagicMock()
        mock_response.text = "Chunk summary."
        mock_gen.return_value = mock_response

        summarizer = GeminiSummarizer(api_key="fake", chunk_size=100)
        text = "Sentence one. " * 50  # ~700 chars, should make ~7 chunks

        summarizer.summarize_long(text)

        chunks = chunk_text(text, 100)
        # At minimum: 1 call per chunk + 1 synthesis call
        assert mock_gen.call_count >= len(chunks) + 1

    @patch("src.summarization.vertex_summarize.generate_with_retry")
    @patch("src.summarization.vertex_summarize.genai")
    def test_single_chunk_skips_reduce(self, mock_genai, mock_gen):
        """A single chunk should skip the reduce phase."""
        mock_response = MagicMock()
        mock_response.text = "Direct summary."
        mock_gen.return_value = mock_response

        summarizer = GeminiSummarizer(api_key="fake", chunk_size=10000)
        text = "Short text."

        result = summarizer.summarize_long(text)
        assert result == "Direct summary."
        assert mock_gen.call_count == 1  # just the direct summarize


# ── Existing methods still work ──────────────────────────────────────

class TestExistingMethods:
    @patch("src.summarization.vertex_summarize.generate_with_retry")
    @patch("src.summarization.vertex_summarize.genai")
    def test_summarize_multiple(self, mock_genai, mock_gen):
        mock_response = MagicMock()
        mock_response.text = "Multi-doc summary."
        mock_gen.return_value = mock_response

        summarizer = GeminiSummarizer(api_key="fake")
        result = summarizer.summarize_multiple(["text1", "text2"])
        assert result == "Multi-doc summary."

    @patch("src.summarization.vertex_summarize.generate_with_retry")
    @patch("src.summarization.vertex_summarize.genai")
    def test_compare(self, mock_genai, mock_gen):
        mock_response = MagicMock()
        mock_response.text = "Comparison."
        mock_gen.return_value = mock_response

        summarizer = GeminiSummarizer(api_key="fake")
        result = summarizer.compare(["sum1", "sum2"])
        assert result == "Comparison."

    @patch("src.summarization.vertex_summarize.generate_with_retry")
    @patch("src.summarization.vertex_summarize.genai")
    def test_summarize_batch(self, mock_genai, mock_gen):
        mock_response = MagicMock()
        mock_response.text = "Batch item."
        mock_gen.return_value = mock_response

        summarizer = GeminiSummarizer(api_key="fake")
        results = summarizer.summarize_batch(["a", "b", "c"])
        assert len(results) == 3
