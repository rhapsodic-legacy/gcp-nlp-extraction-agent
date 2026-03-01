"""BERTScore evaluation — semantic similarity metric for summarization.

ROUGE counts n-gram overlaps, which penalizes paraphrasing. BERTScore uses
contextual embeddings to measure semantic similarity, making it fairer for
abstractive summaries that capture meaning without copying exact phrases.
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError("Set GOOGLE_API_KEY environment variable before running.")


def load_articles(n=25):
    """Load n CNN/DailyMail articles via streaming."""
    from datasets import load_dataset

    print(f"Loading {n} CNN/DailyMail articles...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
    articles = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        articles.append(row)
    print(f"Loaded {len(articles)} articles")
    return articles


def extractive_baseline(article_text):
    """First 3 sentences as baseline summary."""
    sentences = article_text.replace("\n", " ").split(". ")
    return ". ".join(sentences[:3]) + "."


def main():
    articles = load_articles(25)

    references = [a["highlights"] for a in articles]
    baselines = [extractive_baseline(a["article"]) for a in articles]

    # Generate Gemini summaries
    from src.summarization.vertex_summarize import GeminiSummarizer

    summarizer = GeminiSummarizer(api_key=os.environ["GOOGLE_API_KEY"])
    gemini_summaries = []

    print("Generating Gemini summaries...")
    for i, article in enumerate(articles):
        try:
            summary = summarizer.summarize(article["article"])
            gemini_summaries.append(summary)
            print(f"  [{i+1:2d}] {summary[:80]}")
        except Exception as e:
            print(f"  [{i+1:2d}] ERROR: {e}")
            gemini_summaries.append(baselines[i])  # fallback
        time.sleep(2)

    # Compute BERTScore
    from bert_score import score as bert_score

    print("\nComputing BERTScore for extractive baseline...")
    P_base, R_base, F1_base = bert_score(baselines, references, lang="en", verbose=True)

    print("\nComputing BERTScore for Gemini summaries...")
    P_gem, R_gem, F1_gem = bert_score(gemini_summaries, references, lang="en", verbose=True)

    print("\n" + "=" * 60)
    print("  BERTSCORE EVALUATION — 25 CNN/DailyMail articles")
    print("=" * 60)
    print(f"  Extractive Baseline:")
    print(f"    Precision: {P_base.mean():.4f}")
    print(f"    Recall:    {R_base.mean():.4f}")
    print(f"    F1:        {F1_base.mean():.4f}")
    print(f"\n  Gemini 2.5 Flash:")
    print(f"    Precision: {P_gem.mean():.4f}")
    print(f"    Recall:    {R_gem.mean():.4f}")
    print(f"    F1:        {F1_gem.mean():.4f}")
    print(f"\n  Delta (Gemini - Baseline):")
    delta_p = (P_gem.mean() - P_base.mean()) / P_base.mean() * 100
    delta_r = (R_gem.mean() - R_base.mean()) / R_base.mean() * 100
    delta_f1 = (F1_gem.mean() - F1_base.mean()) / F1_base.mean() * 100
    print(f"    Precision: {delta_p:+.1f}%")
    print(f"    Recall:    {delta_r:+.1f}%")
    print(f"    F1:        {delta_f1:+.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
