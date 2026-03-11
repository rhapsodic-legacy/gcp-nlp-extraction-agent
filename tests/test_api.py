"""Tests for the streaming agent API.

Tests endpoint behaviour, SSE event format, and request validation
with fully mocked Gemini calls — no live API key needed.
"""

import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Mock all GCP dependencies before importing the app
with patch.dict("sys.modules", {
    "google": MagicMock(),
    "google.genai": MagicMock(),
    "google.genai.types": MagicMock(),
    "google.genai.errors": MagicMock(),
    "google.cloud": MagicMock(),
    "google.cloud.firestore": MagicMock(),
    "google.cloud.bigquery": MagicMock(),
    "google.cloud.language": MagicMock(),
    "google.cloud.language_v2": MagicMock(),
    "google.cloud.storage": MagicMock(),
    "spacy": MagicMock(),
    "spacy.language": MagicMock(),
    "presidio_analyzer": MagicMock(),
}):
    import os
    os.environ.setdefault("GOOGLE_API_KEY", "test-key-fake")

    from src.api.app import app, _sse, _parse_response, _execute_tool

from fastapi.testclient import TestClient

client = TestClient(app)


# ── Health endpoint ──────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_ok(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "api_key_set" in data
        assert "data_loaded" in data

    def test_health_shows_no_data_initially(self):
        resp = client.get("/api/health")
        data = resp.json()
        assert data["data_loaded"] is False


# ── SSE formatting ───────────────────────────────────────────────────

class TestSSEFormat:
    def test_sse_format(self):
        result = _sse("thought", {"content": "Let me search"})
        assert result.startswith("event: message\n")
        assert "data: " in result
        assert result.endswith("\n\n")

        payload = json.loads(result.split("data: ")[1].strip())
        assert payload["type"] == "thought"
        assert payload["content"] == "Let me search"

    def test_sse_answer_event(self):
        result = _sse("answer", {"content": "The top complaints are..."})
        payload = json.loads(result.split("data: ")[1].strip())
        assert payload["type"] == "answer"

    def test_sse_done_event(self):
        result = _sse("done", {})
        payload = json.loads(result.split("data: ")[1].strip())
        assert payload["type"] == "done"


# ── Response parsing ─────────────────────────────────────────────────

class TestParseResponse:
    def test_parse_thought_and_action(self):
        text = "THOUGHT: I need to search for battery issues\nACTION: SEARCH(battery)"
        thought, action, answer = _parse_response(text)
        assert thought == "I need to search for battery issues"
        assert action == "SEARCH(battery)"
        assert answer is None

    def test_parse_thought_and_answer(self):
        text = "THOUGHT: I have enough info\nANSWER: The top complaints are about batteries."
        thought, action, answer = _parse_response(text)
        assert thought == "I have enough info"
        assert action is None
        assert answer == "The top complaints are about batteries."

    def test_parse_stops_at_observation(self):
        text = "THOUGHT: thinking\nACTION: SEARCH(x)\nOBSERVATION: simulated"
        thought, action, answer = _parse_response(text)
        assert thought == "thinking"
        assert action == "SEARCH(x)"
        assert answer is None

    def test_parse_multiline_answer(self):
        text = "THOUGHT: done\nANSWER: Line 1\nLine 2\nLine 3"
        thought, action, answer = _parse_response(text)
        assert "Line 1" in answer
        assert "Line 3" in answer

    def test_parse_empty_input(self):
        thought, action, answer = _parse_response("")
        assert thought is None
        assert action is None
        assert answer is None


# ── Tool execution ───────────────────────────────────────────────────

class TestToolExecution:
    def test_unknown_tool(self):
        tools = {}
        result = _execute_tool(tools, "UNKNOWN_TOOL(args)")
        assert "Unknown tool" in result

    def test_malformed_action(self):
        tools = {}
        result = _execute_tool(tools, "no-parens")
        assert "Tool error" in result

    def test_search_tool_dispatch(self):
        mock_search = MagicMock()
        mock_search.search.return_value = [{"id": "1", "text": "battery issue"}]
        tools = {"SEARCH": mock_search}

        result = _execute_tool(tools, 'SEARCH("battery")')
        mock_search.search.assert_called_once()
        parsed = json.loads(result)
        assert len(parsed) == 1

    def test_search_with_source_type(self):
        mock_search = MagicMock()
        mock_search.search.return_value = []
        tools = {"SEARCH": mock_search}

        _execute_tool(tools, 'SEARCH("battery", "review")')
        mock_search.search.assert_called_once_with("battery", source_type="review")

    def test_sentiment_tool_dispatch(self):
        mock_sentiment = MagicMock()
        mock_sentiment.analyze.return_value = {"score": 0.5, "magnitude": 1.2}
        tools = {"ANALYZE_SENTIMENT": mock_sentiment}

        result = _execute_tool(tools, 'ANALYZE_SENTIMENT("great product")')
        parsed = json.loads(result)
        assert parsed["score"] == 0.5

    def test_extract_entities_dispatch(self):
        mock_extract = MagicMock()
        mock_extract.extract_entities.return_value = {"entities": []}
        tools = {"EXTRACT_ENTITIES": mock_extract}

        result = _execute_tool(tools, 'EXTRACT_ENTITIES("some text")')
        mock_extract.extract_entities.assert_called_once()

    def test_summarize_dispatch(self):
        mock_summarize = MagicMock()
        mock_summarize.summarize.return_value = "A concise summary."
        tools = {"SUMMARIZE": mock_summarize}

        result = _execute_tool(tools, 'SUMMARIZE("long text here")')
        assert result == "A concise summary."


# ── Frontend serving ─────────────────────────────────────────────────

class TestFrontend:
    def test_root_serves_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Customer Insight Agent" in resp.text


# ── Load data endpoint ───────────────────────────────────────────────

class TestLoadData:
    def test_load_nonexistent_file(self):
        resp = client.post("/api/load-data", json={"jsonl_path": "/tmp/nonexistent.jsonl"})
        assert resp.status_code == 404

    def test_load_data_success(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id":"1","text":"hello","source_type":"review"}\n')
            f.write('{"id":"2","text":"world","source_type":"news"}\n')
            path = f.name

        try:
            resp = client.post("/api/load-data", json={"jsonl_path": path})
            assert resp.status_code == 200
            data = resp.json()
            assert data["documents_loaded"] == 2
        finally:
            os.unlink(path)
