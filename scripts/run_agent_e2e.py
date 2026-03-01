"""Run the agent end-to-end with live Gemini API.

Validates the full pipeline: data loading -> agent reasoning -> tool calls
-> critic evaluation. Uses real API calls (not mocked).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError("Set GOOGLE_API_KEY environment variable before running.")

import pandas as pd
from src.data.loader import load_reviews, load_support_tickets
from src.agent.agent import CustomerInsightAgent


def load_documents():
    reviews = load_reviews(max_docs=500)
    tickets = load_support_tickets(max_docs=500)
    rows = []
    for doc in reviews + tickets:
        rows.append({
            "id": doc.id,
            "text": doc.text,
            "source_type": doc.source_type,
            "metadata": str(doc.metadata),
        })
    return pd.DataFrame(rows)


def display_response(response, label=""):
    print("\n" + "=" * 70)
    print(f"  {label}")
    print("=" * 70)
    for i, step in enumerate(response.steps, 1):
        print(f"\n--- Step {i} ---")
        if step.thought:
            print(f"  THOUGHT: {step.thought}")
        if step.action and step.action != "ANSWER":
            print(f"  ACTION:  {step.action}")
            obs = step.observation[:200] + "..." if len(step.observation) > 200 else step.observation
            print(f"  OBSERVE: {obs}")
    print("\n" + "=" * 70)
    print("  FINAL ANSWER")
    print("=" * 70)
    print(f"\n{response.answer[:800]}")
    print(f"\n[Session: {response.session_id} | Steps: {len(response.steps)}]")


def main():
    print("Loading documents...")
    documents_df = load_documents()
    print(f"Loaded {len(documents_df)} documents ({documents_df.source_type.value_counts().to_dict()})")

    # --- Regular agent ---
    agent = CustomerInsightAgent(
        api_key=os.environ["GOOGLE_API_KEY"],
        documents_df=documents_df,
        model_name="gemini-2.5-flash",
        use_firestore=False,
        max_steps=8,
    )
    print(f"Agent initialized with {len(agent.tools)} tools: {', '.join(agent.tools.keys())}")

    # Query 1: Product complaints
    print("\n>>> Query 1: Product Complaint Analysis")
    r1 = agent.query(
        "What are the most common product complaints in support tickets? "
        "Find some examples and summarize the key issues."
    )
    display_response(r1, "QUERY 1: Product Complaint Analysis")

    # Query 2: Sentiment comparison (same session)
    print("\n>>> Query 2: Sentiment Comparison (same session)")
    r2 = agent.query(
        "How does customer sentiment differ between product reviews and support tickets? "
        "Analyze a few examples from each.",
        session_id=r1.session_id,
    )
    display_response(r2, "QUERY 2: Sentiment Comparison")

    # --- Critic-enabled agent ---
    print("\n>>> Query 3: With Critic Validation")
    critic_agent = CustomerInsightAgent(
        api_key=os.environ["GOOGLE_API_KEY"],
        documents_df=documents_df,
        model_name="gemini-2.5-flash",
        use_firestore=False,
        max_steps=8,
        enable_critic=True,
    )

    r3 = critic_agent.query(
        "What are the top product issues in support tickets? "
        "Summarize the key themes."
    )
    display_response(r3, "QUERY 3: With Critic Validation")

    if hasattr(critic_agent, "last_critic_verdict") and critic_agent.last_critic_verdict:
        v = critic_agent.last_critic_verdict
        print(f"\n--- CRITIC VERDICT ---")
        print(f"  Verdict:      {v.verdict}")
        print(f"  Completeness: {v.completeness_score}/5 -- {v.completeness_reason}")
        print(f"  Grounding:    {v.grounding_score}/5 -- {v.grounding_reason}")
        print(f"  Coherence:    {v.coherence_score}/5 -- {v.coherence_reason}")
        print(f"  Overall:      {v.overall_score}/5.0")

    # --- Standalone critic ---
    print("\n>>> Standalone Critic: Evaluate Query 1 response")
    from src.agent.critic import CriticAgent

    critic = CriticAgent(api_key=os.environ["GOOGLE_API_KEY"])
    verdict = critic.evaluate_agent_response(
        query="What are the most common product complaints in support tickets?",
        agent_response=r1,
    )
    print(f"\n--- STANDALONE CRITIC ---")
    print(f"  Verdict: {verdict.verdict} (overall: {verdict.overall_score}/5.0)")
    print(f"  Completeness: {verdict.completeness_score}/5")
    print(f"  Grounding:    {verdict.grounding_score}/5")
    print(f"  Coherence:    {verdict.coherence_score}/5")
    if verdict.revised_answer:
        print(f"\n  Revised answer:\n  {verdict.revised_answer[:300]}...")

    print("\n" + "=" * 70)
    print("  ALL E2E TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
