"""Unit tests for the Critic Agent.

Tests the critic's parsing, verdict logic, and integration with
AgentResponse objects. All Gemini calls are mocked.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.critic import CriticAgent, CriticVerdict
from src.agent.agent import AgentStep, AgentResponse


@pytest.fixture
def critic():
    """CriticAgent with mocked Gemini client."""
    with patch("src.agent.critic.genai") as mock_genai:
        mock_genai.Client.return_value = MagicMock()
        c = CriticAgent(api_key="fake-key")
        return c


class TestCriticVerdict:
    """Tests for the CriticVerdict dataclass."""

    def test_to_dict(self):
        v = CriticVerdict(
            verdict="PASS",
            completeness_score=5,
            completeness_reason="Fully addresses the question",
            grounding_score=4,
            grounding_reason="Most claims supported",
            coherence_score=5,
            coherence_reason="Well structured",
            overall_score=4.67,
        )
        d = v.to_dict()
        assert d["verdict"] == "PASS"
        assert d["scores"]["completeness"] == 5
        assert d["scores"]["overall"] == 4.67
        assert d["revised_answer"] is None

    def test_revised_answer_included(self):
        v = CriticVerdict(
            verdict="REVISE",
            completeness_score=2,
            completeness_reason="Missing key points",
            grounding_score=3,
            grounding_reason="Partially supported",
            coherence_score=4,
            coherence_reason="Clear writing",
            overall_score=3.0,
            revised_answer="Better answer here.",
        )
        d = v.to_dict()
        assert d["revised_answer"] == "Better answer here."


class TestCriticEvaluate:
    """Tests for the evaluate() method with mocked API responses."""

    def test_pass_verdict(self, critic):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "verdict": "PASS",
            "completeness": {"score": 5, "reason": "Complete"},
            "grounding": {"score": 4, "reason": "Well grounded"},
            "coherence": {"score": 5, "reason": "Clear"},
            "revised_answer": None,
        })
        critic.client.models.generate_content = MagicMock(return_value=mock_response)

        verdict = critic.evaluate(
            query="What are the top complaints?",
            answer="Battery and screen issues are most common.",
            evidence=[('SEARCH("complaints")', '[{"text": "battery drain..."}]')],
        )
        assert verdict.verdict == "PASS"
        assert verdict.completeness_score == 5
        assert verdict.overall_score == pytest.approx(4.67, abs=0.01)
        assert verdict.revised_answer is None

    def test_revise_verdict(self, critic):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "verdict": "REVISE",
            "completeness": {"score": 2, "reason": "Missing details"},
            "grounding": {"score": 3, "reason": "Some unsupported claims"},
            "coherence": {"score": 4, "reason": "Readable"},
            "revised_answer": "A more complete answer with citations.",
        })
        critic.client.models.generate_content = MagicMock(return_value=mock_response)

        verdict = critic.evaluate(
            query="What are the issues?",
            answer="There are some issues.",
        )
        assert verdict.verdict == "REVISE"
        assert verdict.revised_answer == "A more complete answer with citations."
        assert verdict.completeness_score == 2

    def test_graceful_on_parse_error(self, critic):
        """If Gemini returns invalid JSON, critic defaults to PASS."""
        mock_response = MagicMock()
        mock_response.text = "not valid json at all"
        critic.client.models.generate_content = MagicMock(return_value=mock_response)

        verdict = critic.evaluate(query="test", answer="test answer")
        assert verdict.verdict == "PASS"
        assert verdict.overall_score == 3.0

    def test_no_evidence(self, critic):
        """Critic works even without evidence."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "verdict": "PASS",
            "completeness": {"score": 3, "reason": "Acceptable"},
            "grounding": {"score": 2, "reason": "No evidence to check"},
            "coherence": {"score": 4, "reason": "Clear"},
            "revised_answer": None,
        })
        critic.client.models.generate_content = MagicMock(return_value=mock_response)

        verdict = critic.evaluate(query="test", answer="answer")
        assert verdict.verdict == "PASS"


class TestEvaluateAgentResponse:
    """Tests for the convenience method that takes AgentResponse objects."""

    def test_extracts_evidence_from_steps(self, critic):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "verdict": "PASS",
            "completeness": {"score": 4, "reason": "Good"},
            "grounding": {"score": 5, "reason": "Evidence-based"},
            "coherence": {"score": 4, "reason": "Clear"},
            "revised_answer": None,
        })
        critic.client.models.generate_content = MagicMock(return_value=mock_response)

        agent_response = AgentResponse(
            answer="Battery issues found in 3 tickets.",
            steps=[
                AgentStep(thought="Search first", action='SEARCH("battery")', observation='[{"text": "battery drain"}]'),
                AgentStep(thought="Got enough", action="ANSWER", observation="Battery issues found in 3 tickets."),
            ],
            session_id="test",
        )

        verdict = critic.evaluate_agent_response(
            query="What are battery issues?",
            agent_response=agent_response,
        )
        assert verdict.verdict == "PASS"
        # Verify the API was called (evidence was extracted and passed)
        critic.client.models.generate_content.assert_called_once()
