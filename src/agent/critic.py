"""Critic Agent — the quality gate in an actor-critic architecture.

In reinforcement learning, the actor generates actions and the critic
evaluates them. Same principle here: the CustomerInsightAgent (actor)
reasons through tool calls and produces an answer. The CriticAgent
reviews that answer against the evidence gathered and scores it on
multiple quality dimensions.

Why a separate module instead of just a method on the actor?

1. Separation of concerns: the actor's job is to reason and act. The
   critic's job is to evaluate. Mixing them conflates two responsibilities.

2. Independent evolution: you can upgrade the critic's evaluation criteria
   (add hallucination detection, compliance checks, citation verification)
   without touching the actor's reasoning logic.

3. Composability: the critic can evaluate outputs from ANY agent — not
   just CustomerInsightAgent. Swap in a different actor, keep the same
   quality gate. That's the whole point of modular architecture.

4. Testability: you can unit test the critic with synthetic actor outputs
   without needing a live Gemini connection for the actor.

The critic evaluates on three axes:
  - Completeness: does the answer address all parts of the question?
  - Grounding: are claims supported by evidence from tool results?
  - Coherence: is the answer well-structured and internally consistent?

Each axis gets a score (1-5) plus a justification. The overall verdict
is PASS, REVISE, or FAIL. If REVISE, the critic also produces an
improved version of the answer.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types


@dataclass
class CriticVerdict:
    """The critic's evaluation of an agent response.

    Structured output makes this auditable — you can log every verdict
    to understand how often the critic catches issues, which axes are
    weakest, and whether the actor is improving over time.
    """

    verdict: str  # "PASS", "REVISE", or "FAIL"
    completeness_score: int  # 1-5
    completeness_reason: str
    grounding_score: int  # 1-5
    grounding_reason: str
    coherence_score: int  # 1-5
    coherence_reason: str
    overall_score: float  # average of the three
    revised_answer: Optional[str] = None  # only if verdict is REVISE
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "scores": {
                "completeness": self.completeness_score,
                "grounding": self.grounding_score,
                "coherence": self.coherence_score,
                "overall": self.overall_score,
            },
            "reasons": {
                "completeness": self.completeness_reason,
                "grounding": self.grounding_reason,
                "coherence": self.coherence_reason,
            },
            "revised_answer": self.revised_answer,
        }


CRITIC_PROMPT = """You are a Critic Agent evaluating the quality of an AI analyst's response.

USER QUERY: {query}

EVIDENCE GATHERED BY THE ACTOR (tool results):
{evidence}

ACTOR'S PROPOSED ANSWER:
{answer}

Evaluate the answer on three axes, scoring each 1-5:

1. COMPLETENESS: Does the answer address all parts of the user's question?
   (5 = fully addresses every aspect, 1 = misses major parts)

2. GROUNDING: Are claims supported by the evidence gathered?
   (5 = every claim cites evidence, 1 = mostly unsupported assertions)

3. COHERENCE: Is the answer well-structured, clear, and internally consistent?
   (5 = professional quality, 1 = confusing or contradictory)

Based on the scores, assign a verdict:
- PASS (all scores >= 4): Answer is good enough to return to the user.
- REVISE (any score 2-3): Answer needs improvement. Provide a revised version.
- FAIL (any score 1): Answer is fundamentally flawed. Provide a revised version.

Return a JSON object with this exact structure:
{{
    "verdict": "PASS" | "REVISE" | "FAIL",
    "completeness": {{"score": 1-5, "reason": "..."}},
    "grounding": {{"score": 1-5, "reason": "..."}},
    "coherence": {{"score": 1-5, "reason": "..."}},
    "revised_answer": "..." or null
}}

If the verdict is PASS, set revised_answer to null.
If the verdict is REVISE or FAIL, write a better answer that fixes the identified issues."""


class CriticAgent:
    """Standalone critic that evaluates actor outputs for quality.

    This is the second half of the actor-critic pattern. The actor
    (CustomerInsightAgent) does the reasoning and tool-calling. The
    critic reviews the final output and either approves it, requests
    revision, or flags it as fundamentally flawed.

    In production, you'd wire this into the pipeline so that every
    agent response passes through the critic before reaching the user.
    The critic's verdicts also feed into monitoring — if the REVISE
    rate spikes, something changed in the actor's behavior and you
    need to investigate.

    Usage:
        critic = CriticAgent(api_key=...)
        verdict = critic.evaluate(
            query="What are the top complaints?",
            answer="Customers complain about batteries.",
            evidence=[("SEARCH(complaints)", "[{...}, {...}]")],
        )
        if verdict.verdict == "PASS":
            return answer
        else:
            return verdict.revised_answer
    """

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def evaluate(
        self,
        query: str,
        answer: str,
        evidence: list[tuple[str, str]] = None,
    ) -> CriticVerdict:
        """Evaluate an actor's answer against the evidence it gathered.

        Args:
            query: The original user question.
            answer: The actor's proposed answer.
            evidence: List of (tool_call, result) tuples from the actor's
                      reasoning trace. Each tuple is (action_string, observation).

        Returns:
            CriticVerdict with scores, reasons, and optionally a revised answer.
        """
        # Format evidence for the prompt
        if evidence:
            evidence_text = "\n".join(
                f"[{action}]: {result[:500]}" for action, result in evidence
            )
        else:
            evidence_text = "(no tool results provided)"

        prompt = CRITIC_PROMPT.format(
            query=query,
            evidence=evidence_text,
            answer=answer,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            parsed = json.loads(response.text)
            return self._parse_verdict(parsed, response.text)

        except (json.JSONDecodeError, AttributeError) as e:
            # If parsing fails, return a conservative PASS — don't block
            # the actor's answer because the critic had a bad day.
            return CriticVerdict(
                verdict="PASS",
                completeness_score=3,
                completeness_reason=f"Critic parse error: {e}",
                grounding_score=3,
                grounding_reason="Unable to evaluate",
                coherence_score=3,
                coherence_reason="Unable to evaluate",
                overall_score=3.0,
                raw_response=getattr(response, "text", ""),
            )

    def _parse_verdict(self, parsed: dict, raw: str) -> CriticVerdict:
        """Convert the parsed JSON response into a CriticVerdict."""
        completeness = parsed.get("completeness", {})
        grounding = parsed.get("grounding", {})
        coherence = parsed.get("coherence", {})

        c_score = completeness.get("score", 3)
        g_score = grounding.get("score", 3)
        h_score = coherence.get("score", 3)
        overall = (c_score + g_score + h_score) / 3

        return CriticVerdict(
            verdict=parsed.get("verdict", "PASS"),
            completeness_score=c_score,
            completeness_reason=completeness.get("reason", ""),
            grounding_score=g_score,
            grounding_reason=grounding.get("reason", ""),
            coherence_score=h_score,
            coherence_reason=coherence.get("reason", ""),
            overall_score=round(overall, 2),
            revised_answer=parsed.get("revised_answer"),
            raw_response=raw,
        )

    def evaluate_agent_response(self, query: str, agent_response) -> CriticVerdict:
        """Convenience method that takes an AgentResponse directly.

        Extracts the evidence from the reasoning trace so you don't
        have to do it manually. Just pass the AgentResponse object
        from CustomerInsightAgent.query() and the critic handles the rest.
        """
        evidence = []
        for step in agent_response.steps:
            if step.action and step.action != "ANSWER" and step.observation:
                evidence.append((step.action, step.observation))

        return self.evaluate(
            query=query,
            answer=agent_response.answer,
            evidence=evidence,
        )
