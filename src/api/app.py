"""Streaming Agent API — FastAPI + Server-Sent Events.

Exposes the CustomerInsightAgent as a REST API with real-time streaming
of THOUGHT/ACTION/OBSERVATION steps via SSE. Includes a built-in HTML
frontend for interactive demos.

Run:
    uvicorn src.api.app:app --reload --port 8000
"""

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google import genai
from google.genai import types

from ..agent.tools import SearchTool, ExtractTool, SentimentTool, SummarizeTool
from ..agent.memory import LocalMemory
from ..agent.critic import CriticAgent
from ..agent.planner import HierarchicalPlanner, StepStatus
from ..api_utils import generate_with_retry, DEFAULT_MODEL, usage, reset_usage

# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Insight Agent",
    description="ReAct agent with real-time streaming of reasoning steps",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Shared state ─────────────────────────────────────────────────────
memory = LocalMemory()
_documents_df: Optional[pd.DataFrame] = None


def _get_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(500, "GOOGLE_API_KEY not set")
    return key


# ── Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    enable_critic: bool = False


class PlannerQueryRequest(BaseModel):
    query: str
    enable_critic: bool = True
    enable_escalation: bool = True


class EscalationDecision(BaseModel):
    approved: bool
    reason: Optional[str] = None


class LoadDataRequest(BaseModel):
    jsonl_path: Optional[str] = None


# ── Agent system prompt (same as agent.py) ───────────────────────────

SYSTEM_PROMPT = """You are a Customer Insight Agent that helps analysts understand patterns
in customer feedback, news, and support tickets. You have access to the following tools:

1. SEARCH(query, source_type?) - Search documents by keywords. Optional source_type filter: "review", "support_ticket", "news", "reddit"
2. EXTRACT_ENTITIES(text) - Extract named entities (people, orgs, products, dates) from text
3. EXTRACT_STRUCTURED(text) - Extract core issues, key attributes, topics, and action items
4. ANALYZE_SENTIMENT(text) - Get sentiment score (-1 to 1) and magnitude
5. SUMMARIZE(text) - Summarize a single document in 2-3 sentences
6. SUMMARIZE_MULTIPLE(texts) - Synthesize multiple documents into one summary
7. COMPARE(summaries) - Compare and contrast multiple summaries

To use a tool, respond with EXACTLY two lines:
THOUGHT: <your reasoning about what to do next>
ACTION: <TOOL_NAME>(<arguments>)

CRITICAL: Output ONLY ONE THOUGHT and ONE ACTION per response. Then STOP.
Do NOT simulate the tool's output. Do NOT write OBSERVATION lines.
Do NOT chain multiple actions. Wait for the real tool result before continuing.

When you have enough information to answer, respond with EXACTLY:
THOUGHT: <final reasoning>
ANSWER: <your comprehensive answer to the user's query>

Important:
- Break complex queries into steps — one tool call at a time
- Use SEARCH first to find relevant documents
- Apply extraction/sentiment on specific documents, not vague queries
- Synthesize findings before answering
- Be specific — cite document sources and data points
"""


# ── Tool execution (mirrors agent.py) ────────────────────────────────

def _build_tools(api_key: str) -> dict:
    return {
        "SEARCH": SearchTool(documents_df=_documents_df),
        "EXTRACT_ENTITIES": ExtractTool(api_key=api_key),
        "EXTRACT_STRUCTURED": ExtractTool(api_key=api_key),
        "ANALYZE_SENTIMENT": SentimentTool(api_key=api_key),
        "SUMMARIZE": SummarizeTool(api_key=api_key),
        "SUMMARIZE_MULTIPLE": SummarizeTool(api_key=api_key),
        "COMPARE": SummarizeTool(api_key=api_key),
    }


def _execute_tool(tools: dict, action: str) -> str:
    """Parse TOOL_NAME(args) and dispatch."""
    try:
        paren_idx = action.index("(")
        tool_name = action[:paren_idx].strip()
        args_str = action[paren_idx + 1 : -1].strip()

        if tool_name == "SEARCH":
            parts = [p.strip().strip("\"'") for p in args_str.split(",")]
            query = parts[0]
            source_type = parts[1] if len(parts) > 1 else None
            if "=" in query:
                query = query.split("=", 1)[1].strip().strip("\"'")
            if source_type and "=" in source_type:
                source_type = source_type.split("=", 1)[1].strip().strip("\"'")
            results = tools["SEARCH"].search(query, source_type=source_type)
            return json.dumps(results[:5], indent=2, default=str)

        elif tool_name == "EXTRACT_ENTITIES":
            text = args_str.strip("\"'")
            result = tools["EXTRACT_ENTITIES"].extract_entities(text)
            return json.dumps(result, indent=2, default=str)

        elif tool_name == "EXTRACT_STRUCTURED":
            text = args_str.strip("\"'")
            result = tools["EXTRACT_STRUCTURED"].extract_structured(text)
            return json.dumps(result, indent=2, default=str)

        elif tool_name == "ANALYZE_SENTIMENT":
            text = args_str.strip("\"'")
            result = tools["ANALYZE_SENTIMENT"].analyze(text)
            return json.dumps(result, indent=2, default=str)

        elif tool_name == "SUMMARIZE":
            text = args_str.strip("\"'")
            return tools["SUMMARIZE"].summarize(text)

        elif tool_name == "SUMMARIZE_MULTIPLE":
            texts = json.loads(args_str)
            return tools["SUMMARIZE_MULTIPLE"].summarize_multiple(texts)

        elif tool_name == "COMPARE":
            summaries = json.loads(args_str)
            return tools["COMPARE"].compare(summaries)

        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {str(e)}"


def _parse_response(text: str):
    """Extract THOUGHT, ACTION, ANSWER from model output."""
    thought = action = answer = None
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("THOUGHT:") and thought is None:
            thought = line[len("THOUGHT:"):].strip()
        elif line.startswith("ACTION:") and action is None:
            action = line[len("ACTION:"):].strip()
        elif line.startswith("ANSWER:") and answer is None:
            answer_start = text.index("ANSWER:")
            answer = text[answer_start + len("ANSWER:"):].strip()
            break
        elif line.startswith("OBSERVATION:"):
            break
    return thought, action, answer


# ── SSE streaming endpoint ───────────────────────────────────────────

@app.post("/api/query/stream")
async def stream_query(req: QueryRequest):
    """Stream agent reasoning steps as Server-Sent Events.

    Each SSE event has a `type` field:
      - thought: Agent's reasoning
      - action: Tool being called
      - observation: Tool result
      - answer: Final answer
      - critic: Critic verdict (if enabled)
      - error: Error message
      - done: Stream complete
    """
    api_key = _get_api_key()

    async def event_stream():
        session_id = req.session_id or str(uuid.uuid4())

        # Build multi-turn history BEFORE adding new message
        history_msgs = memory.get_messages(session_id, limit=20)
        history_block = ""
        if history_msgs:
            lines = [f"{m['role'].upper()}: {m['content']}" for m in history_msgs]
            history_block = "CONVERSATION HISTORY:\n" + "\n".join(lines[-3000:])

        memory.add_message(session_id, "user", req.query)

        client = genai.Client(api_key=api_key)
        tools = _build_tools(api_key)
        conversation = []
        if history_block:
            conversation.append(history_block)
        conversation.append(f"User query: {req.query}")
        max_steps = 10

        yield _sse("session", {"session_id": session_id})

        for step_num in range(max_steps):
            try:
                response = await asyncio.to_thread(
                    generate_with_retry,
                    client,
                    DEFAULT_MODEL,
                    SYSTEM_PROMPT + "\n\n" + "\n\n".join(conversation),
                    types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=1024,
                        stop_sequences=["OBSERVATION:"],
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                response_text = response.text.strip()
            except Exception as e:
                yield _sse("error", {"message": str(e)})
                break

            thought, action, answer = _parse_response(response_text)

            if thought:
                yield _sse("thought", {
                    "step": step_num + 1,
                    "content": thought,
                })

            # Final answer
            if answer:
                final_answer = answer

                # Optional critic pass
                if req.enable_critic:
                    try:
                        critic = CriticAgent(api_key=api_key)
                        evidence = [
                            (c.split("ACTION: ")[-1] if "ACTION: " in c else "", "")
                            for c in conversation if c.startswith("ACTION:")
                        ]
                        verdict = await asyncio.to_thread(
                            critic.evaluate, req.query, answer, evidence,
                        )
                        yield _sse("critic", {
                            "verdict": verdict.verdict,
                            "scores": {
                                "completeness": verdict.completeness_score,
                                "grounding": verdict.grounding_score,
                                "coherence": verdict.coherence_score,
                                "overall": verdict.overall_score,
                            },
                        })
                        if verdict.verdict in ("REVISE", "FAIL") and verdict.revised_answer:
                            final_answer = verdict.revised_answer
                    except Exception:
                        pass

                yield _sse("answer", {"content": final_answer})
                memory.add_message(session_id, "assistant", final_answer)
                break

            # Tool call
            if action:
                yield _sse("action", {
                    "step": step_num + 1,
                    "tool_call": action,
                })

                observation = await asyncio.to_thread(_execute_tool, tools, action)

                yield _sse("observation", {
                    "step": step_num + 1,
                    "content": observation[:2000],  # cap for SSE payload
                })

                conversation.append(f"THOUGHT: {thought}")
                conversation.append(f"ACTION: {action}")
                conversation.append(f"OBSERVATION: {observation}")
            else:
                # Protocol violation — return raw response
                yield _sse("answer", {"content": response_text})
                break

        yield _sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event_type: str, data: dict) -> str:
    """Format a single SSE message."""
    payload = json.dumps({"type": event_type, **data})
    return f"event: message\ndata: {payload}\n\n"


# ── Hierarchical planner endpoint ────────────────────────────────────

@app.post("/api/planner/stream")
async def stream_planner_query(req: PlannerQueryRequest):
    """Stream hierarchical planner execution as Server-Sent Events.

    Event types:
      - plan: The decomposed execution plan (DAG of sub-tasks)
      - step_start: A sub-agent is starting execution
      - step_complete: A sub-agent finished (includes its answer)
      - parallel_batch: Group of steps executing in parallel
      - synthesis: Synthesizing sub-results into final answer
      - escalation: Query flagged for human review
      - answer: Final synthesized answer
      - error / done: Error or stream complete
    """
    api_key = _get_api_key()

    async def event_stream():
        planner = HierarchicalPlanner(
            api_key=api_key,
            documents_df=_documents_df,
            enable_critic=req.enable_critic,
            enable_escalation=req.enable_escalation,
        )

        # Step 1: Create the plan
        plan = planner._create_plan(req.query)

        yield _sse("plan", {
            "query": plan.query,
            "steps": [
                {"id": s.id, "task": s.task, "depends_on": s.depends_on}
                for s in plan.steps
            ],
            "synthesis_instruction": plan.synthesis_instruction,
        })

        # Step 2: Check escalation
        if plan.should_escalate and req.enable_escalation:
            yield _sse("escalation", {
                "reason": plan.escalation_reason,
                "message": "This query has been flagged for human review. "
                           "The agent will still attempt an answer, but results "
                           "should be verified by a human analyst.",
            })

        # Step 3: Execute DAG with streaming updates
        results: dict[str, str] = {}
        completed: set[str] = set()
        step_map = {s.id: s for s in plan.steps}

        while len(completed) < len(plan.steps):
            ready = [
                s for s in plan.steps
                if s.id not in completed
                and s.status == StepStatus.PENDING
                and all(d in completed for d in s.depends_on)
            ]

            if not ready:
                break

            # Emit parallel batch info
            if len(ready) > 1:
                yield _sse("parallel_batch", {
                    "step_ids": [s.id for s in ready],
                    "message": f"Executing {len(ready)} sub-tasks in parallel",
                })

            for s in ready:
                s.status = StepStatus.RUNNING

            async def _run_step(step):
                context_parts = []
                for dep_id in step.depends_on:
                    if dep_id in results:
                        dep = step_map[dep_id]
                        context_parts.append(f"[{dep_id}: {dep.task}]\n{results[dep_id]}")
                context = "\n\n".join(context_parts)
                response = await asyncio.to_thread(
                    planner._run_sub_agent, step.task, context
                )
                return step.id, response.answer

            # Emit step_start for each
            for s in ready:
                yield _sse("step_start", {"id": s.id, "task": s.task})

            # Execute in parallel
            tasks = [_run_step(s) for s in ready]
            step_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in step_results:
                if isinstance(result, Exception):
                    for s in ready:
                        if s.status == StepStatus.RUNNING:
                            s.status = StepStatus.FAILED
                            s.result = f"Error: {result}"
                            results[s.id] = s.result
                            completed.add(s.id)
                            yield _sse("step_complete", {
                                "id": s.id, "status": "failed",
                                "content": s.result[:2000],
                            })
                            break
                else:
                    step_id, answer = result
                    step_map[step_id].status = StepStatus.COMPLETED
                    step_map[step_id].result = answer
                    results[step_id] = answer
                    completed.add(step_id)
                    yield _sse("step_complete", {
                        "id": step_id, "status": "completed",
                        "content": answer[:2000],
                    })

        # Step 4: Synthesize
        yield _sse("synthesis", {"message": "Synthesizing sub-task results..."})

        try:
            final_answer = await asyncio.to_thread(
                planner._synthesize, req.query,
                plan.synthesis_instruction, results, plan,
            )
        except Exception as e:
            yield _sse("error", {"message": f"Synthesis failed: {e}"})
            yield _sse("done", {})
            return

        response_data = {"content": final_answer}
        if plan.should_escalate:
            response_data["escalated"] = True
            response_data["escalation_reason"] = plan.escalation_reason

        yield _sse("answer", response_data)
        yield _sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── REST endpoints ───────────────────────────────────────────────────

@app.post("/api/load-data")
async def load_data(req: LoadDataRequest):
    """Load documents into memory for the search tool."""
    global _documents_df
    path = req.jsonl_path or "data/documents.jsonl"

    if not Path(path).exists():
        raise HTTPException(404, f"File not found: {path}")

    _documents_df = pd.read_json(path, lines=True)
    return {"status": "ok", "documents_loaded": len(_documents_df)}


@app.get("/api/health")
async def health():
    has_key = bool(os.environ.get("GOOGLE_API_KEY"))
    has_data = _documents_df is not None
    return {
        "status": "ok",
        "api_key_set": has_key,
        "data_loaded": has_data,
        "documents_count": len(_documents_df) if has_data else 0,
    }


@app.get("/api/usage")
async def get_usage():
    """Return accumulated token usage and estimated cost."""
    return usage.to_dict()


@app.post("/api/usage/reset")
async def reset_usage_stats():
    """Reset the usage tracker."""
    reset_usage()
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the built-in frontend."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>Customer Insight Agent API</h1><p>Frontend not found. Visit /docs for API docs.</p>")
