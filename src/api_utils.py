"""Shared Gemini API call wrapper with rate-limit retry, model fallback,
and token/cost tracking.
"""

import logging
import threading
import time
from dataclasses import dataclass, field

from google.genai.errors import ClientError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.5-flash-lite"

# Approximate cost per 1M tokens (USD). Updated 2025-03.
# These are estimates for tracking purposes, not billing.
MODEL_COSTS = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
}


@dataclass
class UsageStats:
    """Accumulated token usage and estimated cost across API calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    estimated_cost_usd: float = 0.0
    calls_by_model: dict = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int):
        """Record a single API call's usage."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

        if model not in self.calls_by_model:
            self.calls_by_model[model] = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.calls_by_model[model]["calls"] += 1
        self.calls_by_model[model]["input_tokens"] += input_tokens
        self.calls_by_model[model]["output_tokens"] += output_tokens

        # Estimate cost
        costs = MODEL_COSTS.get(model, MODEL_COSTS[DEFAULT_MODEL])
        self.estimated_cost_usd += (
            input_tokens * costs["input"] / 1_000_000
            + output_tokens * costs["output"] / 1_000_000
        )

    def to_dict(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_calls": self.total_calls,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "by_model": dict(self.calls_by_model),
        }

    def summary(self) -> str:
        total = self.total_input_tokens + self.total_output_tokens
        return (
            f"{self.total_calls} calls | "
            f"{total:,} tokens ({self.total_input_tokens:,} in, {self.total_output_tokens:,} out) | "
            f"~${self.estimated_cost_usd:.4f}"
        )


# Global usage tracker (thread-safe via lock)
_usage_lock = threading.Lock()
usage = UsageStats()


def reset_usage():
    """Reset the global usage tracker."""
    global usage
    with _usage_lock:
        usage = UsageStats()


def _extract_usage(response, model: str):
    """Extract token counts from a Gemini response and record them."""
    try:
        metadata = response.usage_metadata
        input_tokens = getattr(metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(metadata, "candidates_token_count", 0) or 0
        with _usage_lock:
            usage.record(model, input_tokens, output_tokens)
        logger.debug("Tokens: %d in, %d out (model=%s)", input_tokens, output_tokens, model)
    except (AttributeError, TypeError):
        # Response may not have usage metadata in some configurations
        with _usage_lock:
            usage.record(model, 0, 0)


def _call_with_backoff(client, model, contents, config, max_retries=3):
    """Try a single model with exponential backoff on 429s."""
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config,
            )
            _extract_usage(response, model)
            return response
        except ClientError as e:
            if e.code != 429:
                raise
            if attempt < max_retries:
                wait = 2 ** attempt * 10  # 10s, 20s, 40s (faster cycle)
                logger.warning("Rate limited on %s, retrying in %ds (attempt %d/%d)",
                               model, wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                raise


def generate_with_retry(client, model, contents, config, max_retries=3):
    """Call generate_content with backoff, falling back to flash-lite if primary exhausted."""
    try:
        return _call_with_backoff(client, model, contents, config, max_retries)
    except ClientError as e:
        if e.code != 429 or model == FALLBACK_MODEL:
            raise
        logger.warning("Quota exhausted for %s, falling back to %s", model, FALLBACK_MODEL)
        return _call_with_backoff(client, FALLBACK_MODEL, contents, config, max_retries)
