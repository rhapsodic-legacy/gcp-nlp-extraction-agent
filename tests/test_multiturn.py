"""Tests for multi-turn conversational agent support.

Tests session history building, memory get_messages(), and that
conversation context is properly injected into the agent's prompt.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agent.memory import LocalMemory
from src.agent.agent import CustomerInsightAgent, MAX_HISTORY_CHARS


# ── LocalMemory.get_messages() ───────────────────────────────────────

class TestLocalMemoryGetMessages:
    def test_get_messages_empty_session(self):
        mem = LocalMemory()
        assert mem.get_messages("nonexistent") == []

    def test_get_messages_returns_all(self):
        mem = LocalMemory()
        mem.add_message("s1", "user", "hello")
        mem.add_message("s1", "assistant", "hi there")
        mem.add_message("s1", "user", "follow up")

        msgs = mem.get_messages("s1")
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"
        assert msgs[2]["content"] == "follow up"

    def test_get_messages_respects_limit(self):
        mem = LocalMemory()
        for i in range(10):
            mem.add_message("s1", "user", f"msg {i}")

        msgs = mem.get_messages("s1", limit=3)
        assert len(msgs) == 3
        # Should return the LAST 3 messages
        assert msgs[0]["content"] == "msg 7"
        assert msgs[2]["content"] == "msg 9"

    def test_get_messages_preserves_timestamps(self):
        mem = LocalMemory()
        mem.add_message("s1", "user", "test")
        msgs = mem.get_messages("s1")
        assert "timestamp" in msgs[0]


# ── History context building ─────────────────────────────────────────

class TestBuildHistoryContext:
    @patch("src.agent.agent.genai")
    def _make_agent(self, mock_genai):
        df = pd.DataFrame([
            {"id": "1", "text": "test", "source_type": "review", "metadata": "{}"},
        ])
        agent = CustomerInsightAgent(
            api_key="fake",
            documents_df=df,
        )
        return agent

    def test_empty_session_returns_empty(self):
        agent = self._make_agent()
        result = agent._build_history_context("brand-new-session")
        assert result == ""

    def test_existing_session_returns_history(self):
        agent = self._make_agent()
        agent.memory.add_message("s1", "user", "What are the top complaints?")
        agent.memory.add_message("s1", "assistant", "The top complaints are about batteries.")

        result = agent._build_history_context("s1")
        assert "CONVERSATION HISTORY" in result
        assert "What are the top complaints?" in result
        assert "The top complaints are about batteries." in result

    def test_history_respects_char_limit(self):
        agent = self._make_agent()
        # Add many long messages
        for i in range(50):
            agent.memory.add_message("s1", "user", f"Long message {i}: {'x' * 200}")

        result = agent._build_history_context("s1")
        # Should be capped
        assert len(result) <= MAX_HISTORY_CHARS + 500  # some overhead for formatting

    def test_history_preserves_most_recent(self):
        agent = self._make_agent()
        agent.memory.add_message("s1", "user", "old question")
        agent.memory.add_message("s1", "assistant", "old answer")
        agent.memory.add_message("s1", "user", "recent question")
        agent.memory.add_message("s1", "assistant", "recent answer")

        result = agent._build_history_context("s1")
        # Most recent messages should definitely be present
        assert "recent question" in result
        assert "recent answer" in result


# ── Multi-turn query integration ─────────────────────────────────────

class TestMultiTurnQuery:
    @patch("src.agent.agent.genai")
    def test_first_query_no_history(self, mock_genai):
        """First query in a session should work without history."""
        mock_response = MagicMock()
        mock_response.text = "THOUGHT: I can answer directly\nANSWER: The answer is 42."

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        df = pd.DataFrame([
            {"id": "1", "text": "test", "source_type": "review", "metadata": "{}"},
        ])
        agent = CustomerInsightAgent(api_key="fake", documents_df=df)
        response = agent.query("What is the answer?", session_id="new-session")

        assert response.answer == "The answer is 42."

    @patch("src.agent.agent.genai")
    def test_second_query_includes_history(self, mock_genai):
        """Second query should include prior exchange in the prompt."""
        mock_response = MagicMock()
        mock_response.text = "THOUGHT: Using context from history\nANSWER: Drilling into shipping."

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        df = pd.DataFrame([
            {"id": "1", "text": "test", "source_type": "review", "metadata": "{}"},
        ])
        agent = CustomerInsightAgent(api_key="fake", documents_df=df)

        # Simulate prior conversation
        agent.memory.add_message("s1", "user", "What are the top complaints?")
        agent.memory.add_message("s1", "assistant", "The top complaints are about shipping delays.")

        response = agent.query("Drill into those", session_id="s1")

        # Verify the prompt sent to Gemini includes history
        call_args = mock_client.models.generate_content.call_args
        prompt_text = call_args.kwargs.get("contents") or call_args[1].get("contents", "")
        assert "CONVERSATION HISTORY" in prompt_text
        assert "top complaints" in prompt_text
        assert "Drill into those" in prompt_text

    @patch("src.agent.agent.genai")
    def test_session_accumulates_messages(self, mock_genai):
        """Multiple queries on the same session should accumulate history."""
        mock_response = MagicMock()
        mock_response.text = "THOUGHT: done\nANSWER: Response."

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        df = pd.DataFrame([
            {"id": "1", "text": "test", "source_type": "review", "metadata": "{}"},
        ])
        agent = CustomerInsightAgent(api_key="fake", documents_df=df)

        agent.query("First question", session_id="s1")
        agent.query("Second question", session_id="s1")
        agent.query("Third question", session_id="s1")

        msgs = agent.memory.get_messages("s1")
        # 3 user + 3 assistant = 6 messages
        assert len(msgs) == 6
