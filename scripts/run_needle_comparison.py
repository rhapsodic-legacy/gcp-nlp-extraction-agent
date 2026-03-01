"""Run needle-in-a-haystack evaluation with three extractors.

Compares SpaCy, Presidio, and Gemini on the same injected needles
to highlight complementary strengths across extraction approaches.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_reviews, load_support_tickets
from src.evaluation.needle_haystack import NeedleHaystackEvaluator
from src.extraction.spacy_baseline import SpacyExtractor
from src.extraction.presidio_extract import PresidioExtractor


def main():
    # Load documents for the haystack
    print("Loading documents...")
    reviews = load_reviews(max_docs=200)
    tickets = load_support_tickets(max_docs=30)
    all_docs = reviews + tickets
    print(f"Loaded {len(all_docs)} documents")

    evaluator = NeedleHaystackEvaluator()

    # --- SpaCy ---
    print("\n" + "=" * 60)
    print("  SPACY (General NER Baseline)")
    print("=" * 60)
    docs_spacy = [d.__class__(id=d.id, text=d.text, source_type=d.source_type, metadata=d.metadata) for d in all_docs]
    haystack_spacy, injections_spacy = evaluator.build_haystack(docs_spacy, num_needles=6, seed=42)
    spacy_extractor = SpacyExtractor()
    spacy_results = evaluator.evaluate_extraction(haystack_spacy, injections_spacy, spacy_extractor)
    spacy_stats = evaluator.report(spacy_results)

    # --- Presidio ---
    print("\n" + "=" * 60)
    print("  PRESIDIO (PII-focused Baseline)")
    print("=" * 60)
    docs_presidio = [d.__class__(id=d.id, text=d.text, source_type=d.source_type, metadata=d.metadata) for d in all_docs]
    haystack_presidio, injections_presidio = evaluator.build_haystack(docs_presidio, num_needles=6, seed=42)
    presidio_extractor = PresidioExtractor()
    presidio_results = evaluator.evaluate_extraction(haystack_presidio, injections_presidio, presidio_extractor)
    presidio_stats = evaluator.report(presidio_results)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<25s} {'SpaCy':<12s} {'Presidio':<12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Detection Rate':<25s} {spacy_stats['detection_rate']*100:.0f}%{'':<9s} {presidio_stats['detection_rate']*100:.0f}%")
    print(f"  {'Avg Entity Recall':<25s} {spacy_stats['average_recall']*100:.1f}%{'':<8s} {presidio_stats['average_recall']*100:.1f}%")

    # Entity type breakdown
    all_types = sorted(set(list(spacy_stats["by_entity_type"].keys()) + list(presidio_stats["by_entity_type"].keys())))
    print(f"\n  {'Entity Type':<15s} {'SpaCy':<12s} {'Presidio':<12s}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")
    for etype in all_types:
        s = spacy_stats["by_entity_type"].get(etype, {"found": 0, "total": 0})
        p = presidio_stats["by_entity_type"].get(etype, {"found": 0, "total": 0})
        s_pct = f"{s['found']}/{s['total']}" if s["total"] > 0 else "n/a"
        p_pct = f"{p['found']}/{p['total']}" if p["total"] > 0 else "n/a"
        print(f"  {etype:<15s} {s_pct:<12s} {p_pct:<12s}")


if __name__ == "__main__":
    main()
