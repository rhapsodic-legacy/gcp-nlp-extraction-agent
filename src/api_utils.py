"""Shared Gemini API call wrapper with rate-limit retry and model fallback."""

import logging
import time

from google.genai.errors import ClientError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.5-flash-lite"


def _call_with_backoff(client, model, contents, config, max_retries=3):
    """Try a single model with exponential backoff on 429s."""
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config,
            )
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
