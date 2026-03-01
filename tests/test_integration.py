"""Integration tests — multi-component interactions with mocked APIs.

These tests exercise the full agent pipeline: reasoning loop -> tool dispatch
-> result processing -> critic evaluation. They verify that components work
together correctly, not just in isolation.

All GCP calls (Gemini, BigQuery) are mocked. These tests should run in CI
without credentials.
"""

import json
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from src.agent.agent import CustomerInsightAgent, AgentStep, AgentResponse
from src.agent.critic import CriticAgent, CriticVerdict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def documents_df():
    """Realistic document set spanning multiple source types."""
    return pd.DataFrame([
        {"id": "r1", "text": "Battery drains in 2 hours. Worst tablet ever.", "source_type": "review", "metadata": '{"rating": 1}'},
        {"id": "r2", "text": "Love the battery life, lasts all day!", "source_type": "review", "metadata": '{"rating": 5}'},
        {"id": "r3", "text": "Screen quality is amazing but battery is weak.", "source_type": "review", "metadata": '{"rating": 3}'},
        {"id": "t1", "text": "Customer reports battery swelling on tablet model X.", "source_type": "support_ticket", "metadata": '{"priority": "high"}'},
        {"id": "t2", "text": "Screen replacement request for cracked display.", "source_type": "support_ticket", "metadata": '{"priority": "medium"}'},
        {"id": "t3", "text": "Battery not charging after firmware update.", "source_type": "support_ticket", "metadata": '{"priority": "high"}'},
    ])


@pytest.fixture
def agent(documents_df):
    """Agent with mocked Gemini — ready for multi-step scenarios."""
    with patch("src.agent.agent.genai") as mock_genai:
        mock_genai.Client.return_value = MagicMock()
        a = CustomerInsightAgent(
            api_key="fake-key",
            documents_df=documents_df,
            model_name="gemini-2.5-flash",
            max_steps=8,
        )
        return a


@pytest.fixture
def critic_agent(documents_df):
    """Agent with critic enabled and both Gemini clients mocked."""
    with patch("src.agent.agent.genai") as mock_genai:
        mock_genai.Client.return_value = MagicMock()
        a = CustomerInsightAgent(
            api_key="fake-key",
            documents_df=documents_df,
            model_name="gemini-2.5-flash",
            max_steps=8,
            enable_critic=True,
        )
        return a


# ---------------------------------------------------------------------------
# Multi-step reasoning
# ---------------------------------------------------------------------------

class TestMultiStepReasoning:
    """Tests for multi-tool query resolution."""

    def test_search_then_summarize_then_answer(self, agent):
        """Agent searches, summarizes results, then answers — 3-step chain."""
        responses = [
            # Step 1: Search
            MagicMock(text='THOUGHT: I need to find battery complaints\nACTION: SEARCH("battery")'),
            # Step 2: Summarize findings (mock the summarize tool)
            MagicMock(text='THOUGHT: Let me summarize what I found\nACTION: SUMMARIZE("Battery drains in 2 hours. Worst tablet ever.")'),
            # Step 3: Answer
            MagicMock(text='THOUGHT: I have enough information now\nANSWER: Battery complaints center on rapid drain and swelling issues.'),
        ]
        agent.client.models.generate_content = MagicMock(side_effect=responses)

        # Mock the summarizer to avoid real API call
        agent.tools["SUMMARIZE"]._summarizer = MagicMock()
        agent.tools["SUMMARIZE"]._summarizer.summarize.return_value = "Battery drain is the main complaint."

        result = agent.query("What are the battery complaints?")
        assert len(result.steps) == 3
        assert result.steps[0].action == 'SEARCH("battery")'
        assert "SUMMARIZE" in result.steps[1].action
        assert result.steps[2].action == "ANSWER"
        assert "battery" in result.answer.lower()

    def test_search_both_sources_then_compare(self, agent):
        """Agent searches reviews and tickets separately, then answers."""
        responses = [
            MagicMock(text='THOUGHT: Search reviews first\nACTION: SEARCH("battery", "review")'),
            MagicMock(text='THOUGHT: Now search tickets\nACTION: SEARCH("battery", "support_ticket")'),
            MagicMock(text=(
                'THOUGHT: Reviews show mixed sentiment; tickets show critical issues\n'
                'ANSWER: Reviews mention battery drain as an annoyance, while support '
                'tickets show more severe issues like swelling and charging failures.'
            )),
        ]
        agent.client.models.generate_content = MagicMock(side_effect=responses)

        result = agent.query("Compare battery issues across reviews and tickets")
        assert len(result.steps) == 3
        # First search: reviews only
        step1_result = json.loads(result.steps[0].observation)
        assert all(r["source_type"] == "review" for r in step1_result)
        # Second search: tickets only
        step2_result = json.loads(result.steps[1].observation)
        assert all(r["source_type"] == "support_ticket" for r in step2_result)

    def test_tool_error_recovery(self, agent):
        """Agent recovers from a tool error and tries a different approach."""
        responses = [
            # First attempt: malformed tool call
            MagicMock(text='THOUGHT: Extract entities\nACTION: EXTRACT_ENTITIES()'),
            # Agent sees error, tries search instead
            MagicMock(text='THOUGHT: That failed, let me search instead\nACTION: SEARCH("battery")'),
            # Now it can answer
            MagicMock(text='THOUGHT: Got results\nANSWER: Battery issues found across 4 documents.'),
        ]
        agent.client.models.generate_content = MagicMock(side_effect=responses)

        # Mock the extractor so it raises on empty input
        agent.tools["EXTRACT_ENTITIES"]._gemini_extractor = MagicMock()
        agent.tools["EXTRACT_ENTITIES"]._gemini_extractor.extract_entities.side_effect = ValueError("Empty text")

        result = agent.query("What entities appear in battery complaints?")
        assert "Tool error" in result.steps[0].observation
        assert len(result.steps) == 3
        assert result.answer is not None


# ---------------------------------------------------------------------------
# Actor-Critic integration
# ---------------------------------------------------------------------------

class TestActorCriticIntegration:
    """Tests for the full actor-critic pipeline."""

    def test_critic_pass_keeps_original_answer(self, critic_agent):
        """When critic says PASS, the original answer is returned unchanged."""
        # Actor response
        actor_response = MagicMock(text='THOUGHT: Done\nANSWER: Battery drain is the top complaint.')
        critic_agent.client.models.generate_content = MagicMock(return_value=actor_response)

        # Mock the critic's Gemini call
        critic_json = json.dumps({
            "verdict": "PASS",
            "completeness": {"score": 5, "reason": "Fully addresses the query"},
            "grounding": {"score": 4, "reason": "Supported by evidence"},
            "coherence": {"score": 5, "reason": "Clear and well-structured"},
            "revised_answer": None,
        })

        with patch("src.agent.critic.genai") as mock_critic_genai:
            mock_critic_client = MagicMock()
            mock_critic_response = MagicMock(text=critic_json)
            mock_critic_client.models.generate_content = MagicMock(return_value=mock_critic_response)
            mock_critic_genai.Client.return_value = mock_critic_client

            result = critic_agent.query("What is the top complaint?")

        assert "Battery drain is the top complaint" in result.answer
        assert critic_agent.last_critic_verdict is not None
        assert critic_agent.last_critic_verdict.verdict == "PASS"

    def test_critic_revise_replaces_answer(self, critic_agent):
        """When critic says REVISE, the revised answer replaces the original."""
        # Actor gives weak answer
        actor_response = MagicMock(text='THOUGHT: Done\nANSWER: There are some issues.')
        critic_agent.client.models.generate_content = MagicMock(return_value=actor_response)

        # Critic provides a better answer
        critic_json = json.dumps({
            "verdict": "REVISE",
            "completeness": {"score": 2, "reason": "Too vague, missing specifics"},
            "grounding": {"score": 2, "reason": "No evidence cited"},
            "coherence": {"score": 4, "reason": "Grammatically fine"},
            "revised_answer": "Battery drain and screen cracking are the primary complaints, with 4 out of 6 documents mentioning battery issues.",
        })

        with patch("src.agent.critic.genai") as mock_critic_genai:
            mock_critic_client = MagicMock()
            mock_critic_response = MagicMock(text=critic_json)
            mock_critic_client.models.generate_content = MagicMock(return_value=mock_critic_response)
            mock_critic_genai.Client.return_value = mock_critic_client

            result = critic_agent.query("What are the main issues?")

        assert "Battery drain and screen cracking" in result.answer
        assert critic_agent.last_critic_verdict.verdict == "REVISE"
        assert critic_agent.last_critic_verdict.completeness_score == 2

    def test_critic_failure_doesnt_block_answer(self, critic_agent):
        """If the critic itself errors, the original answer passes through."""
        actor_response = MagicMock(text='THOUGHT: Done\nANSWER: Battery issues are common.')
        critic_agent.client.models.generate_content = MagicMock(return_value=actor_response)

        # Critic throws an exception
        with patch("src.agent.critic.genai") as mock_critic_genai:
            mock_critic_genai.Client.side_effect = Exception("API unavailable")

            result = critic_agent.query("What are the issues?")

        # Original answer should still come through
        assert "Battery issues are common" in result.answer
        assert critic_agent.last_critic_verdict is None


# ---------------------------------------------------------------------------
# Critic standalone evaluation
# ---------------------------------------------------------------------------

class TestCriticStandalone:
    """Tests for the CriticAgent evaluating pre-built AgentResponses."""

    def test_evaluate_multi_step_response(self):
        """Critic correctly processes evidence from a multi-step agent trace."""
        with patch("src.agent.critic.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            critic = CriticAgent(api_key="fake-key")

            critic_json = json.dumps({
                "verdict": "PASS",
                "completeness": {"score": 5, "reason": "All aspects covered"},
                "grounding": {"score": 5, "reason": "Every claim backed by search results"},
                "coherence": {"score": 4, "reason": "Well organized"},
                "revised_answer": None,
            })
            mock_client.models.generate_content = MagicMock(
                return_value=MagicMock(text=critic_json)
            )

            agent_response = AgentResponse(
                answer="Battery issues found in 4 documents across reviews and tickets.",
                steps=[
                    AgentStep(
                        thought="Search reviews",
                        action='SEARCH("battery", "review")',
                        observation='[{"text": "Battery drains fast", "source_type": "review"}]',
                    ),
                    AgentStep(
                        thought="Search tickets",
                        action='SEARCH("battery", "support_ticket")',
                        observation='[{"text": "Battery swelling reported", "source_type": "support_ticket"}]',
                    ),
                    AgentStep(
                        thought="Enough info",
                        action="ANSWER",
                        observation="Battery issues found in 4 documents.",
                    ),
                ],
                session_id="test-123",
            )

            verdict = critic.evaluate_agent_response(
                query="What are the battery issues?",
                agent_response=agent_response,
            )

        assert verdict.verdict == "PASS"
        assert verdict.overall_score == pytest.approx(4.67, abs=0.01)
        # Verify the prompt included evidence (2 search steps, not the ANSWER step)
        call_args = mock_client.models.generate_content.call_args
        prompt = call_args.kwargs.get("contents", call_args.args[0] if call_args.args else "")
        assert "battery" in str(prompt).lower()

    def test_verdict_scores_calculation(self):
        """Overall score is correctly averaged from three axes."""
        with patch("src.agent.critic.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            critic = CriticAgent(api_key="fake-key")

            # Scores: 3 + 4 + 5 = 12 / 3 = 4.0
            critic_json = json.dumps({
                "verdict": "PASS",
                "completeness": {"score": 3, "reason": "Acceptable"},
                "grounding": {"score": 4, "reason": "Good"},
                "coherence": {"score": 5, "reason": "Excellent"},
                "revised_answer": None,
            })
            mock_client.models.generate_content = MagicMock(
                return_value=MagicMock(text=critic_json)
            )

            verdict = critic.evaluate(query="test", answer="test answer")
            assert verdict.overall_score == 4.0


# ---------------------------------------------------------------------------
# Session memory across queries
# ---------------------------------------------------------------------------

class TestSessionPersistence:
    """Tests for multi-turn conversations with memory."""

    def test_follow_up_query_uses_same_session(self, agent):
        """Second query with same session_id builds on the first."""
        response1 = MagicMock(text='THOUGHT: Done\nANSWER: Battery issues found.')
        response2 = MagicMock(text='THOUGHT: Done\nANSWER: Screen issues also found.')
        agent.client.models.generate_content = MagicMock(side_effect=[response1, response2])

        r1 = agent.query("What are the battery issues?", session_id="session-1")
        r2 = agent.query("What about screen issues?", session_id="session-1")

        session = agent.memory.get_session("session-1")
        messages = session["messages"]
        # 2 user messages + 2 assistant messages = 4 total
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"

    def test_separate_sessions_are_isolated(self, agent):
        """Different session_ids have independent memory."""
        response = MagicMock(text='THOUGHT: Done\nANSWER: Test answer')
        agent.client.models.generate_content = MagicMock(return_value=response)

        agent.query("Query A", session_id="session-a")
        agent.query("Query B", session_id="session-b")

        session_a = agent.memory.get_session("session-a")
        session_b = agent.memory.get_session("session-b")
        assert len(session_a["messages"]) == 2
        assert len(session_b["messages"]) == 2
        assert session_a["messages"][0]["content"] != session_b["messages"][0]["content"]


# ---------------------------------------------------------------------------
# End-to-end pipeline scenarios
# ---------------------------------------------------------------------------

class TestEndToEndScenarios:
    """Realistic query scenarios testing the full pipeline."""

    def test_sentiment_analysis_pipeline(self, agent):
        """Agent searches, analyzes sentiment, then reports findings."""
        responses = [
            MagicMock(text='THOUGHT: Find negative reviews\nACTION: SEARCH("battery", "review")'),
            MagicMock(text='THOUGHT: Analyze sentiment of the negative one\nACTION: ANALYZE_SENTIMENT("Battery drains in 2 hours. Worst tablet ever.")'),
            MagicMock(text=(
                'THOUGHT: Clear negative sentiment\n'
                'ANSWER: Sentiment analysis shows strong negative reaction (score: -0.8) '
                'to battery drain issues. Customers describe it as the "worst" experience.'
            )),
        ]
        agent.client.models.generate_content = MagicMock(side_effect=responses)

        # Mock sentiment tool
        agent.tools["ANALYZE_SENTIMENT"]._client = MagicMock()
        mock_sentiment_response = MagicMock(text='{"score": -0.8, "magnitude": 3.5}')
        agent.tools["ANALYZE_SENTIMENT"]._client.models.generate_content = MagicMock(
            return_value=mock_sentiment_response
        )

        result = agent.query("What's the sentiment around battery issues in reviews?")
        assert len(result.steps) == 3
        sentiment_obs = json.loads(result.steps[1].observation)
        assert sentiment_obs["score"] == -0.8
        assert "negative" in result.answer.lower()

    def test_entity_extraction_pipeline(self, agent):
        """Agent searches then extracts entities from results."""
        responses = [
            MagicMock(text='THOUGHT: Find support tickets\nACTION: SEARCH("battery", "support_ticket")'),
            MagicMock(text='THOUGHT: Extract entities from the first result\nACTION: EXTRACT_ENTITIES("Customer reports battery swelling on tablet model X.")'),
            MagicMock(text=(
                'THOUGHT: Found product entity\n'
                'ANSWER: Entity extraction identified "tablet model X" as the product '
                'with battery swelling reports.'
            )),
        ]
        agent.client.models.generate_content = MagicMock(side_effect=responses)

        # Mock entity extractor
        mock_entity = MagicMock(text="tablet model X", type="PRODUCT", salience=0.9)
        agent.tools["EXTRACT_ENTITIES"]._gemini_extractor = MagicMock()
        agent.tools["EXTRACT_ENTITIES"]._gemini_extractor.extract_entities.return_value = [mock_entity]

        result = agent.query("What products are mentioned in battery complaints?")
        assert len(result.steps) == 3
        entities_obs = json.loads(result.steps[1].observation)
        assert entities_obs["entities"][0]["type"] == "PRODUCT"

    def test_max_steps_returns_partial_findings(self, agent):
        """When hitting step limit, agent returns accumulated observations."""
        agent.max_steps = 3
        responses = [
            MagicMock(text='THOUGHT: Search reviews\nACTION: SEARCH("battery", "review")'),
            MagicMock(text='THOUGHT: Search tickets\nACTION: SEARCH("battery", "support_ticket")'),
            MagicMock(text='THOUGHT: Need more\nACTION: SEARCH("screen")'),
        ]
        agent.client.models.generate_content = MagicMock(side_effect=responses)

        result = agent.query("Comprehensive analysis of all issues")
        assert "unable to fully answer" in result.answer.lower()
        assert len(result.steps) == 3
        # All steps should have observations from actual search results
        for step in result.steps:
            assert step.observation != ""
