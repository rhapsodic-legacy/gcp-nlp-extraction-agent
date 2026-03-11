"""Tests for token/cost tracking in api_utils."""

from unittest.mock import MagicMock, patch

import pytest

from src.api_utils import UsageStats, reset_usage, _extract_usage, usage, MODEL_COSTS


class TestUsageStats:
    def test_initial_state(self):
        stats = UsageStats()
        assert stats.total_calls == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.estimated_cost_usd == 0.0

    def test_record_single_call(self):
        stats = UsageStats()
        stats.record("gemini-2.5-flash", 100, 50)
        assert stats.total_calls == 1
        assert stats.total_input_tokens == 100
        assert stats.total_output_tokens == 50
        assert stats.calls_by_model["gemini-2.5-flash"]["calls"] == 1

    def test_record_multiple_models(self):
        stats = UsageStats()
        stats.record("gemini-2.5-flash", 100, 50)
        stats.record("gemini-2.5-flash-lite", 200, 100)
        assert stats.total_calls == 2
        assert stats.total_input_tokens == 300
        assert stats.total_output_tokens == 150
        assert len(stats.calls_by_model) == 2

    def test_cost_estimation(self):
        stats = UsageStats()
        stats.record("gemini-2.5-flash", 1_000_000, 1_000_000)
        # input: $0.15, output: $0.60
        expected = 0.15 + 0.60
        assert abs(stats.estimated_cost_usd - expected) < 0.001

    def test_to_dict(self):
        stats = UsageStats()
        stats.record("gemini-2.5-flash", 100, 50)
        d = stats.to_dict()
        assert d["total_tokens"] == 150
        assert d["total_calls"] == 1
        assert "estimated_cost_usd" in d
        assert "by_model" in d

    def test_summary_string(self):
        stats = UsageStats()
        stats.record("gemini-2.5-flash", 100, 50)
        s = stats.summary()
        assert "1 calls" in s
        assert "150 tokens" in s
        assert "$" in s


class TestExtractUsage:
    def test_extracts_from_response(self):
        stats = UsageStats()

        response = MagicMock()
        response.usage_metadata.prompt_token_count = 500
        response.usage_metadata.candidates_token_count = 200

        # Patch the global usage to use our local stats
        with patch("src.api_utils.usage", stats):
            _extract_usage(response, "gemini-2.5-flash")

        assert stats.total_input_tokens == 500
        assert stats.total_output_tokens == 200

    def test_handles_missing_metadata(self):
        stats = UsageStats()
        response = MagicMock(spec=[])  # no usage_metadata attr

        with patch("src.api_utils.usage", stats):
            _extract_usage(response, "gemini-2.5-flash")

        # Should still record the call with 0 tokens
        assert stats.total_calls == 1
        assert stats.total_input_tokens == 0


class TestResetUsage:
    def test_reset_clears_stats(self):
        import src.api_utils as mod
        mod.usage.record("gemini-2.5-flash", 100, 50)
        assert mod.usage.total_calls > 0

        reset_usage()

        assert mod.usage.total_calls == 0
        assert mod.usage.total_input_tokens == 0
