"""Customer Insight Agent -- ReAct-style orchestrator.

Implements a Reason+Act loop: the agent iteratively selects tools, observes
results, and reasons about next steps until it can answer the user's query.
Gemini provides the reasoning backbone; memory is pluggable (local or Firestore).
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
from ..api_utils import generate_with_retry, DEFAULT_MODEL


# System prompt defining available tools, call format, and reasoning protocol.
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
- This is a multi-turn conversation. Prior exchanges are shown as CONVERSATION HISTORY.
  Use that context to resolve references like "those", "the same", "drill into", etc.
"""

# Maximum characters of conversation history to include in the prompt.
MAX_HISTORY_CHARS = 3000


@dataclass
class AgentStep:
    """A single step in the agent's reasoning chain — thought, action, result."""

    thought: str
    action: str
    observation: str = ""


@dataclass
class AgentResponse:
    """Complete response containing the final answer and full reasoning trace."""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    session_id: str = ""


class CustomerInsightAgent:
    """ReAct agent that orchestrates extraction and summarization tools.

    Decomposes analytical questions into tool calls, reasons over intermediate
    results, and assembles a comprehensive answer. Stateful via pluggable
    memory (LocalMemory or FirestoreMemory). An optional critic pass validates
    the final answer for completeness and factual grounding before returning.
    """

    def __init__(
        self,
        api_key: str = None,
        documents_df: pd.DataFrame = None,
        model_name: str = DEFAULT_MODEL,
        use_firestore: bool = False,
        max_steps: int = 10,
        enable_critic: bool = False,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.max_steps = max_steps
        self.enable_critic = enable_critic

        # Initialize all tools with the shared API key
        self.tools = {
            "SEARCH": SearchTool(documents_df=documents_df),
            "EXTRACT_ENTITIES": ExtractTool(api_key=self.api_key),
            "EXTRACT_STRUCTURED": ExtractTool(api_key=self.api_key),
            "ANALYZE_SENTIMENT": SentimentTool(api_key=self.api_key),
            "SUMMARIZE": SummarizeTool(api_key=self.api_key),
            "SUMMARIZE_MULTIPLE": SummarizeTool(api_key=self.api_key),
            "COMPARE": SummarizeTool(api_key=self.api_key),
        }

        # Select memory backend (same interface either way)
        self.memory = FirestoreMemory() if use_firestore else LocalMemory()

    def _execute_tool(self, action: str) -> str:
        """Parse a ``TOOL_NAME(args)`` string and dispatch to the corresponding tool."""
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
            # Report the error so the agent can reason about recovery.
            return f"Tool error: {type(e).__name__}: {str(e)}"

    def _parse_response(self, text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse the first THOUGHT/ACTION/ANSWER from model output.

        Only the first occurrence of each tag is extracted. Subsequent
        THOUGHT/ACTION cycles are ignored to enforce one tool call per step.
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

    def _critic_evaluate(self, user_query: str, answer: str, steps: list[AgentStep],
                         conversation: list[str]) -> str:
        """Run the CriticAgent, then feed critique back to the agent for self-correction.

        Instead of returning the critic's revised_answer directly (which
        lacks access to the tools and evidence), we feed the critique back
        into the agent's reasoning loop so it can produce a better answer
        using its full context.

        Stores the CriticVerdict on ``self.last_critic_verdict`` for inspection.
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

            if verdict.verdict == "PASS":
                return answer

            # Self-correction: feed the critique back to the agent
            critique_feedback = self._format_critique(verdict)
            conversation.append(f"THOUGHT: Here is my proposed answer:\n{answer}")
            conversation.append(f"CRITIC FEEDBACK: {critique_feedback}")

            # One more generation pass with the critique context
            response = generate_with_retry(
                self.client,
                model=self.model_name,
                contents=SYSTEM_PROMPT + "\n\n" + "\n\n".join(conversation)
                    + "\n\nRevise your answer based on the critic's feedback. "
                    "Respond with:\nTHOUGHT: <reasoning about how to improve>\n"
                    "ANSWER: <your improved answer>",
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            _, _, revised = self._parse_response(response.text.strip())
            if revised:
                return revised

            # Fallback: use critic's revised answer if agent didn't produce one
            if verdict.revised_answer:
                return verdict.revised_answer
            return answer

        except Exception:
            # Critic failure should never block the answer
            self.last_critic_verdict = None
            return answer

    @staticmethod
    def _format_critique(verdict) -> str:
        """Format a CriticVerdict into a feedback string for the agent."""
        parts = [f"Verdict: {verdict.verdict} (overall: {verdict.overall_score}/5)"]
        if verdict.completeness_score < 4:
            parts.append(f"Completeness ({verdict.completeness_score}/5): {verdict.completeness_reason}")
        if verdict.grounding_score < 4:
            parts.append(f"Grounding ({verdict.grounding_score}/5): {verdict.grounding_reason}")
        if verdict.coherence_score < 4:
            parts.append(f"Coherence ({verdict.coherence_score}/5): {verdict.coherence_reason}")
        return " | ".join(parts)

    def _build_history_context(self, session_id: str) -> str:
        """Build a conversation history block from prior session messages.

        Includes up to MAX_HISTORY_CHARS of prior exchanges so the agent
        can resolve references like 'those', 'drill into', 'compare to last'.
        Returns an empty string for new sessions.
        """
        messages = self.memory.get_messages(session_id)
        if not messages:
            return ""

        lines = []
        total_chars = 0
        # Walk backwards from most recent, prepend to preserve order
        for msg in reversed(messages):
            role = msg["role"].upper()
            content = msg["content"]
            line = f"{role}: {content}"
            total_chars += len(line)
            if total_chars > MAX_HISTORY_CHARS:
                break
            lines.insert(0, line)

        if not lines:
            return ""
        return "CONVERSATION HISTORY:\n" + "\n".join(lines)

    def query(self, user_query: str, session_id: str = None) -> AgentResponse:
        """Run the ReAct loop for a user query.

        Supports multi-turn conversations: when a session_id is provided
        and has prior messages, the conversation history is included in
        the prompt so the agent can resolve references like 'those',
        'the shipping ones', 'compare to last time', etc.

        Args:
            user_query: Natural language question from the user.
            session_id: Optional session ID for memory persistence across queries.

        Returns:
            AgentResponse with the final answer and full reasoning trace.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Build history BEFORE adding the new message so we don't include
        # the current query twice.
        history_block = self._build_history_context(session_id)

        self.memory.add_message(session_id, "user", user_query)

        # Build the conversation context for the model
        conversation = []
        if history_block:
            conversation.append(history_block)
        conversation.append(f"User query: {user_query}")
        steps = []

        for step_num in range(self.max_steps):
            # Generate next reasoning step; stop sequence prevents simulated output.
            response = generate_with_retry(
                self.client,
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

            # Final answer — optionally validated by the critic.
            if answer:
                if self.enable_critic:
                    answer = self._critic_evaluate(user_query, answer, steps, conversation)
                step = AgentStep(thought=thought or "", action="ANSWER", observation=answer)
                steps.append(step)
                self.memory.add_message(session_id, "assistant", answer)
                return AgentResponse(answer=answer, steps=steps, session_id=session_id)

            # Tool call — execute and feed the observation back.
            if action:
                observation = self._execute_tool(action)
                step = AgentStep(thought=thought or "", action=action, observation=observation)
                steps.append(step)

                # Append to conversation context
                conversation.append(f"THOUGHT: {thought}")
                conversation.append(f"ACTION: {action}")
                conversation.append(f"OBSERVATION: {observation}")
            else:
                # Model did not follow the protocol — treat response as the answer.
                self.memory.add_message(session_id, "assistant", response_text)
                return AgentResponse(answer=response_text, steps=steps, session_id=session_id)

        # Step limit reached — return partial results.
        final = "I was unable to fully answer your query within the allowed reasoning steps. Here is what I found so far:\n"
        final += "\n".join(s.observation for s in steps if s.observation)
        return AgentResponse(answer=final, steps=steps, session_id=session_id)
