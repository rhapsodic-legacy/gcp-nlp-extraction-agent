"""Unit tests for the Customer Insight Agent.

Tests the agent's parsing, tool dispatch, and reasoning loop with mocked
Gemini responses. No live API calls — fast, deterministic, CI-friendly.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agent.agent import CustomerInsightAgent, AgentStep, AgentResponse, SYSTEM_PROMPT


@pytest.fixture
def sample_df():
    """A minimal DataFrame for search tool testing."""
    return pd.DataFrame([
        {"id": "1", "text": "Battery drains too fast on my tablet", "source_type": "review", "metadata": "{}"},
        {"id": "2", "text": "Screen cracked after one drop", "source_type": "review", "metadata": "{}"},
        {"id": "3", "text": "Issue with battery on my GoPro", "source_type": "support_ticket", "metadata": "{}"},
        {"id": "4", "text": "Great product, love the camera quality", "source_type": "review", "metadata": "{}"},
    ])


@pytest.fixture
def agent(sample_df):
    """Agent with mocked Gemini client — no API calls."""
    with patch("src.agent.agent.genai") as mock_genai:
        mock_genai.Client.return_value = MagicMock()
        a = CustomerInsightAgent(
            api_key="fake-key",
            documents_df=sample_df,
            model_name="gemini-2.5-flash",
        )
        return a


class TestParseResponse:
    """Tests for _parse_response — the protocol parser."""

    def test_parses_thought_and_action(self, agent):
        text = 'THOUGHT: I need to search for complaints\nACTION: SEARCH("battery problem")'
        thought, action, answer = agent._parse_response(text)
        assert thought == "I need to search for complaints"
        assert action == 'SEARCH("battery problem")'
        assert answer is None

    def test_parses_thought_and_answer(self, agent):
        text = "THOUGHT: I have enough info\nANSWER: The main complaints are about batteries."
        thought, action, answer = agent._parse_response(text)
        assert thought == "I have enough info"
        assert action is None
        assert answer == "The main complaints are about batteries."

    def test_takes_first_action_only(self, agent):
        """If the model outputs multiple actions, only the first is taken."""
        text = (
            "THOUGHT: First search\n"
            "ACTION: SEARCH(\"battery\")\n"
            "OBSERVATION: some results\n"
            "THOUGHT: Now extract\n"
            "ACTION: EXTRACT_ENTITIES(\"text\")\n"
        )
        thought, action, answer = agent._parse_response(text)
        assert thought == "First search"
        assert action == 'SEARCH("battery")'
        assert answer is None

    def test_stops_at_observation(self, agent):
        """Parser stops when it hits a simulated OBSERVATION line."""
        text = (
            "THOUGHT: Search first\n"
            "ACTION: SEARCH(\"test\")\n"
            "OBSERVATION: fake results here\n"
        )
        thought, action, answer = agent._parse_response(text)
        assert action == 'SEARCH("test")'

    def test_multiline_answer(self, agent):
        """ANSWER can span multiple lines."""
        text = "THOUGHT: Done\nANSWER: Line one.\nLine two.\nLine three."
        thought, action, answer = agent._parse_response(text)
        assert "Line one" in answer
        assert "Line three" in answer

    def test_empty_response(self, agent):
        thought, action, answer = agent._parse_response("")
        assert thought is None
        assert action is None
        assert answer is None


class TestSearchTool:
    """Tests for the local DataFrame search backend."""

    def test_search_finds_matching_docs(self, agent):
        results = agent.tools["SEARCH"].search("battery")
        assert len(results) == 2  # "Battery drains..." and "Issue with battery..."

    def test_search_filters_by_source_type(self, agent):
        results = agent.tools["SEARCH"].search("battery", source_type="review")
        assert len(results) == 1
        assert results[0]["source_type"] == "review"

    def test_search_no_results(self, agent):
        results = agent.tools["SEARCH"].search("nonexistent_term_xyz")
        assert len(results) == 0

    def test_search_case_insensitive(self, agent):
        results = agent.tools["SEARCH"].search("BATTERY")
        assert len(results) == 2

    def test_search_respects_max_results(self, agent):
        results = agent.tools["SEARCH"].search("battery", max_results=1)
        assert len(results) == 1


class TestExecuteTool:
    """Tests for _execute_tool — the tool dispatch interpreter."""

    def test_search_dispatch(self, agent):
        result = agent._execute_tool('SEARCH("battery")')
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_search_with_source_filter(self, agent):
        result = agent._execute_tool('SEARCH("battery", "support_ticket")')
        parsed = json.loads(result)
        assert len(parsed) == 1

    def test_unknown_tool_returns_error(self, agent):
        result = agent._execute_tool('UNKNOWN_TOOL("arg")')
        assert "Unknown tool" in result

    def test_malformed_action_returns_error(self, agent):
        result = agent._execute_tool("not a valid tool call")
        assert "Tool error" in result

    def test_search_with_named_params(self, agent):
        """Handles Gemini's named parameter format: SEARCH(query="...", source_type="...")."""
        result = agent._execute_tool('SEARCH(query="battery", source_type="support_ticket")')
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["source_type"] == "support_ticket"

    def test_search_named_query_only(self, agent):
        """Handles query= prefix without source_type."""
        result = agent._execute_tool('SEARCH(query="battery")')
        parsed = json.loads(result)
        assert len(parsed) == 2


class TestAgentQuery:
    """Tests for the full query loop with mocked Gemini responses."""

    def test_direct_answer(self, agent, sample_df):
        """Agent returns answer on first step — no tool calls needed."""
        mock_response = MagicMock()
        mock_response.text = "THOUGHT: I can answer directly\nANSWER: The top complaint is battery drain."
        agent.client.models.generate_content = MagicMock(return_value=mock_response)

        result = agent.query("What is the top complaint?")
        assert isinstance(result, AgentResponse)
        assert "battery drain" in result.answer
        assert len(result.steps) == 1
        assert result.steps[0].action == "ANSWER"

    def test_tool_then_answer(self, agent):
        """Agent calls one tool, then answers."""
        response1 = MagicMock()
        response1.text = 'THOUGHT: I need to search first\nACTION: SEARCH("battery")'

        response2 = MagicMock()
        response2.text = "THOUGHT: I found results\nANSWER: Battery issues are common."

        agent.client.models.generate_content = MagicMock(
            side_effect=[response1, response2]
        )

        result = agent.query("What are the battery issues?")
        assert "Battery issues" in result.answer
        assert len(result.steps) == 2
        assert result.steps[0].action == 'SEARCH("battery")'
        assert result.steps[1].action == "ANSWER"

    def test_max_steps_safety_valve(self, agent):
        """Agent stops at max_steps and returns partial results."""
        agent.max_steps = 2

        loop_response = MagicMock()
        loop_response.text = 'THOUGHT: Keep searching\nACTION: SEARCH("battery")'
        agent.client.models.generate_content = MagicMock(return_value=loop_response)

        result = agent.query("What are the issues?")
        assert "unable to fully answer" in result.answer.lower()
        assert len(result.steps) == 2

    def test_graceful_degradation_on_bad_output(self, agent):
        """If Gemini doesn't follow protocol, treat response as the answer."""
        mock_response = MagicMock()
        mock_response.text = "I think the main issues are battery and screen problems."
        agent.client.models.generate_content = MagicMock(return_value=mock_response)

        result = agent.query("What are the issues?")
        assert "battery" in result.answer.lower()

    def test_session_id_generation(self, agent):
        mock_response = MagicMock()
        mock_response.text = "THOUGHT: done\nANSWER: Test answer"
        agent.client.models.generate_content = MagicMock(return_value=mock_response)

        result = agent.query("test")
        assert result.session_id != ""
        assert len(result.session_id) > 0


class TestMemoryIntegration:
    """Tests for session memory during agent queries."""

    def test_memory_stores_messages(self, agent):
        mock_response = MagicMock()
        mock_response.text = "THOUGHT: done\nANSWER: Test answer"
        agent.client.models.generate_content = MagicMock(return_value=mock_response)

        result = agent.query("What is the issue?", session_id="test-session")
        session = agent.memory.get_session("test-session")
        assert session is not None
        assert len(session["messages"]) == 2  # user + assistant

    def test_explicit_session_id(self, agent):
        mock_response = MagicMock()
        mock_response.text = "THOUGHT: done\nANSWER: Test"
        agent.client.models.generate_content = MagicMock(return_value=mock_response)

        result = agent.query("test", session_id="my-session-123")
        assert result.session_id == "my-session-123"


class TestSystemPrompt:
    """Tests for the system prompt — the agent's firmware."""

    def test_prompt_contains_all_tools(self):
        for tool in ["SEARCH", "EXTRACT_ENTITIES", "EXTRACT_STRUCTURED",
                      "ANALYZE_SENTIMENT", "SUMMARIZE", "SUMMARIZE_MULTIPLE", "COMPARE"]:
            assert tool in SYSTEM_PROMPT

    def test_prompt_enforces_single_action(self):
        assert "ONLY ONE THOUGHT and ONE ACTION" in SYSTEM_PROMPT

    def test_prompt_forbids_observation_simulation(self):
        assert "Do NOT write OBSERVATION" in SYSTEM_PROMPT
