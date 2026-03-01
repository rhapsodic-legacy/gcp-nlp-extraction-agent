"""Customer Insight Agent — a ReAct-style orchestrator.

This is the brain of the system, and I'm not going to lie, it's the part
I'm most proud of. The idea is simple but powerful: instead of hardcoding
a pipeline that does extraction-then-summarization-then-report, you build
an agent that can *reason* about what tools to use and in what order.

It works like this:
  1. User asks a complex question
  2. Agent THINKS about what it needs to know
  3. Agent picks a TOOL and calls it
  4. Agent OBSERVES the result
  5. Agent decides if it needs more info or can answer now
  6. Repeat steps 2-5 until it has a good answer

This is called a ReAct loop (Reasoning + Acting), and it's basically how
a good engineer troubleshoots a problem. You don't just run every test —
you think about what's likely wrong, check that specific thing, and let
the result guide your next move.

The architecture is:
  User Query -> Plan -> [Tool Call -> Observe -> Reason]* -> Answer

Gemini serves as the reasoning backbone. The tools (search, extract,
sentiment, summarize) are the hands. Memory (local or Firestore) is
the hippocampus. Together they can answer questions that no single API
call could handle on its own.
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from google import genai
from google.genai import types

from .tools import SearchTool, ExtractTool, SentimentTool, SummarizeTool
from .memory import LocalMemory, FirestoreMemory


# The system prompt is the agent's instruction manual. It tells Gemini what
# tools are available, how to format tool calls, and how to reason through
# multi-step problems. Treat this like firmware — it defines the behavior.
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


@dataclass
class AgentStep:
    """A single step in the agent's reasoning chain — thought, action, result."""

    thought: str
    action: str
    observation: str = ""


@dataclass
class AgentResponse:
    """The complete response: final answer plus the full reasoning trace.

    Keeping the trace is important — it lets you audit how the agent
    arrived at its answer. Debuggability is not optional.
    """

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    session_id: str = ""


class CustomerInsightAgent:
    """ReAct agent that orchestrates extraction and summarization tools.

    This is the conductor of the orchestra. It takes a complex analytical
    question, breaks it down into tool calls, reasons about intermediate
    results, and builds up a comprehensive answer. Think of it as a very
    methodical research assistant that shows its work.

    The agent is stateful via memory (LocalMemory for dev, Firestore for
    production). It remembers what it's been asked and what it's found,
    which means follow-up questions can build on previous context.

    Optionally, a critic pass validates the final answer for completeness
    and factual grounding — a lightweight actor-critic pattern that catches
    low-quality responses before they reach the user.
    """

    def __init__(
        self,
        api_key: str = None,
        documents_df: pd.DataFrame = None,
        model_name: str = "gemini-2.5-flash",
        use_firestore: bool = False,
        max_steps: int = 10,
        enable_critic: bool = False,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.max_steps = max_steps
        self.enable_critic = enable_critic

        # Wire up all the tools — each one gets the same API key
        self.tools = {
            "SEARCH": SearchTool(documents_df=documents_df),
            "EXTRACT_ENTITIES": ExtractTool(api_key=self.api_key),
            "EXTRACT_STRUCTURED": ExtractTool(api_key=self.api_key),
            "ANALYZE_SENTIMENT": SentimentTool(api_key=self.api_key),
            "SUMMARIZE": SummarizeTool(api_key=self.api_key),
            "SUMMARIZE_MULTIPLE": SummarizeTool(api_key=self.api_key),
            "COMPARE": SummarizeTool(api_key=self.api_key),
        }

        # Pick memory backend — Firestore for production persistence,
        # local dict for development. Same interface either way.
        self.memory = FirestoreMemory() if use_firestore else LocalMemory()

    def _execute_tool(self, action: str) -> str:
        """Parse a tool call string and execute the right tool.

        The agent outputs tool calls as "TOOL_NAME(args)" strings, and this
        method parses that format and dispatches to the right tool. It's
        essentially a tiny interpreter. Not the prettiest parser in the world,
        but it works reliably for the formats Gemini produces.
        """
        try:
            # Parse "TOOL_NAME(args)" format
            paren_idx = action.index("(")
            tool_name = action[:paren_idx].strip()
            args_str = action[paren_idx + 1 : -1].strip()

            if tool_name == "SEARCH":
                # Handle both positional and named parameter formats:
                #   SEARCH("battery", "review")       — positional
                #   SEARCH(query="battery", source_type="review") — named
                parts = [p.strip().strip("\"'") for p in args_str.split(",")]
                query = parts[0]
                source_type = parts[1] if len(parts) > 1 else None
                # Strip named param prefixes (query=, source_type=)
                if "=" in query:
                    query = query.split("=", 1)[1].strip().strip("\"'")
                if source_type and "=" in source_type:
                    source_type = source_type.split("=", 1)[1].strip().strip("\"'")
                results = self.tools["SEARCH"].search(query, source_type=source_type)
                return json.dumps(results[:5], indent=2, default=str)

            elif tool_name == "EXTRACT_ENTITIES":
                text = args_str.strip("\"'")
                result = self.tools["EXTRACT_ENTITIES"].extract_entities(text)
                return json.dumps(result, indent=2, default=str)

            elif tool_name == "EXTRACT_STRUCTURED":
                text = args_str.strip("\"'")
                result = self.tools["EXTRACT_STRUCTURED"].extract_structured(text)
                return json.dumps(result, indent=2, default=str)

            elif tool_name == "ANALYZE_SENTIMENT":
                text = args_str.strip("\"'")
                result = self.tools["ANALYZE_SENTIMENT"].analyze(text)
                return json.dumps(result, indent=2, default=str)

            elif tool_name == "SUMMARIZE":
                text = args_str.strip("\"'")
                return self.tools["SUMMARIZE"].summarize(text)

            elif tool_name == "SUMMARIZE_MULTIPLE":
                texts = json.loads(args_str)
                return self.tools["SUMMARIZE_MULTIPLE"].summarize_multiple(texts)

            elif tool_name == "COMPARE":
                summaries = json.loads(args_str)
                return self.tools["COMPARE"].compare(summaries)

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            # Don't let one bad tool call crash the whole reasoning loop.
            # Report the error and let the agent reason about what to do next.
            return f"Tool error: {type(e).__name__}: {str(e)}"

    def _parse_response(self, text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse THOUGHT/ACTION/ANSWER from the model's text output.

        Only extracts the FIRST thought, action, and answer. If the model
        outputs multiple THOUGHT/ACTION cycles (simulating the full loop),
        we take only the first one and ignore the rest. This forces one
        tool call per reasoning step, which is the whole point of ReAct.
        """
        thought = None
        action = None
        answer = None

        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("THOUGHT:") and thought is None:
                thought = line[len("THOUGHT:"):].strip()
            elif line.startswith("ACTION:") and action is None:
                action = line[len("ACTION:"):].strip()
            elif line.startswith("ANSWER:") and answer is None:
                # Collect everything after ANSWER: (may span multiple lines)
                answer_start = text.index("ANSWER:")
                answer = text[answer_start + len("ANSWER:"):].strip()
                break
            elif line.startswith("OBSERVATION:"):
                # Model is simulating tool output — stop parsing here.
                # Real observations come from actual tool execution.
                break

        return thought, action, answer

    def _critic_evaluate(self, user_query: str, answer: str, steps: list[AgentStep]) -> str:
        """Critic pass — delegates to the standalone CriticAgent.

        This is the integration point between actor and critic. The CriticAgent
        evaluates the answer on completeness, grounding, and coherence. If the
        verdict is REVISE or FAIL and the critic provides a revised answer,
        we use that instead. If PASS, the original answer goes through unchanged.

        The CriticVerdict is stored on self.last_critic_verdict for inspection.
        """
        from .critic import CriticAgent

        try:
            critic = CriticAgent(api_key=self.api_key, model_name=self.model_name)
            evidence = [
                (s.action, s.observation)
                for s in steps
                if s.observation and s.action != "ANSWER"
            ]
            verdict = critic.evaluate(
                query=user_query,
                answer=answer,
                evidence=evidence,
            )
            self.last_critic_verdict = verdict

            if verdict.verdict in ("REVISE", "FAIL") and verdict.revised_answer:
                return verdict.revised_answer
            return answer

        except Exception:
            # Critic failure should never block the answer
            self.last_critic_verdict = None
            return answer

    def query(self, user_query: str, session_id: str = None) -> AgentResponse:
        """Process a user query through the full ReAct loop.

        This is the main entry point. Give it a question, get back an answer
        with a full reasoning trace. The agent will call tools, reason about
        results, and keep going until it has a good answer or hits the step limit.

        The step limit (default 10) is a safety valve — it prevents infinite
        loops if the agent gets confused. In practice, most queries resolve
        in 3-5 steps.

        Args:
            user_query: Natural language question from the user.
            session_id: Optional session ID for memory persistence across queries.

        Returns:
            AgentResponse with the final answer and every reasoning step.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        self.memory.add_message(session_id, "user", user_query)

        # Build the conversation context for the model
        conversation = [f"User query: {user_query}"]
        steps = []

        for step_num in range(self.max_steps):
            # Ask Gemini what to do next. Stop sequences prevent the model
            # from simulating tool output — it must stop after ACTION or ANSWER
            # and let us execute the tool for real.
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=SYSTEM_PROMPT + "\n\n" + "\n\n".join(conversation),
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1024,
                    stop_sequences=["OBSERVATION:"],
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            response_text = response.text.strip()

            thought, action, answer = self._parse_response(response_text)

            # Got a final answer? Run it through the critic if enabled.
            if answer:
                if self.enable_critic:
                    answer = self._critic_evaluate(user_query, answer, steps)
                step = AgentStep(thought=thought or "", action="ANSWER", observation=answer)
                steps.append(step)
                self.memory.add_message(session_id, "assistant", answer)
                return AgentResponse(answer=answer, steps=steps, session_id=session_id)

            # Got a tool call? Execute it and feed the result back in.
            if action:
                observation = self._execute_tool(action)
                step = AgentStep(thought=thought or "", action=action, observation=observation)
                steps.append(step)

                # Append to the conversation so the model can see what happened
                conversation.append(f"THOUGHT: {thought}")
                conversation.append(f"ACTION: {action}")
                conversation.append(f"OBSERVATION: {observation}")
            else:
                # Model didn't follow the protocol — treat its entire response
                # as the answer. Graceful degradation beats crashing.
                self.memory.add_message(session_id, "assistant", response_text)
                return AgentResponse(answer=response_text, steps=steps, session_id=session_id)

        # Hit the step limit — give the user what we've got so far
        final = "I was unable to fully answer your query within the allowed reasoning steps. Here is what I found so far:\n"
        final += "\n".join(s.observation for s in steps if s.observation)
        return AgentResponse(answer=final, steps=steps, session_id=session_id)
