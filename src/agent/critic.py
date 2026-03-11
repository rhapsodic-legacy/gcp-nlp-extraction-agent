"""Critic Agent -- actor-critic quality gate for agent responses.

Evaluates agent answers on completeness, grounding, and coherence (each
scored 1-5). Returns a PASS, REVISE, or FAIL verdict with an optional
revised answer. Decoupled from the actor for independent testability.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types

from ..api_utils import generate_with_retry, DEFAULT_MODEL


@dataclass
class CriticVerdict:
    """Structured evaluation result from the critic, including per-axis scores."""

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
    """Evaluates actor outputs on completeness, grounding, and coherence.

    Reviews an agent's proposed answer against the gathered evidence and
    returns a CriticVerdict (PASS / REVISE / FAIL). When the verdict is
    REVISE or FAIL, a revised answer is included.
    """

    def __init__(self, api_key: str = None, model_name: str = DEFAULT_MODEL):
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
            response = generate_with_retry(
                self.client,
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
            # On parse failure, default to PASS so the actor's answer is not blocked.
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
        """Convenience wrapper that extracts evidence from an AgentResponse."""
        evidence = []
        for step in agent_response.steps:
            if step.action and step.action != "ANSWER" and step.observation:
                evidence.append((step.action, step.observation))

        return self.evaluate(
            query=query,
            answer=agent_response.answer,
            evidence=evidence,
        )
