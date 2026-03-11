"""Summarization evaluation against CNN/DailyMail gold-standard highlights.

Provides ROUGE scoring (ROUGE-1, ROUGE-2, ROUGE-L), length/compression
analysis, and qualitative spot-check utilities for comparing generated
summaries against human-written references.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ROUGEScores:
    """ROUGE-1, ROUGE-2, and ROUGE-L precision/recall/F1 for a single summary."""

    rouge1_precision: float = 0.0
    rouge1_recall: float = 0.0
    rouge1_f1: float = 0.0
    rouge2_precision: float = 0.0
    rouge2_recall: float = 0.0
    rouge2_f1: float = 0.0
    rougeL_precision: float = 0.0
    rougeL_recall: float = 0.0
    rougeL_f1: float = 0.0

    def to_dict(self) -> dict:
        return {
            "rouge1": {"precision": self.rouge1_precision, "recall": self.rouge1_recall, "f1": self.rouge1_f1},
            "rouge2": {"precision": self.rouge2_precision, "recall": self.rouge2_recall, "f1": self.rouge2_f1},
            "rougeL": {"precision": self.rougeL_precision, "recall": self.rougeL_recall, "f1": self.rougeL_f1},
        }


@dataclass
class EvaluationResult:
    """Evaluation output for a single document: summaries, ROUGE scores, and length statistics."""

    doc_id: str
    generated_summary: str
    reference_summary: str
    rouge_scores: ROUGEScores
    source_length: int = 0
    generated_length: int = 0
    reference_length: int = 0
    compression_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "rouge": self.rouge_scores.to_dict(),
            "source_length": self.source_length,
            "generated_length": self.generated_length,
            "reference_length": self.reference_length,
            "compression_ratio": self.compression_ratio,
        }


class SummarizationEvaluator:
    """ROUGE-based evaluator for generated summaries against reference summaries.

    Uses Google's ``rouge_score`` library with stemming enabled.
    """

    def __init__(self):
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"],
                use_stemmer=True,
            )
        except ImportError:
            raise ImportError(
                "Install rouge-score: pip install rouge-score"
            )

    def score_single(
        self,
        generated: str,
        reference: str,
        doc_id: str = "",
        source_text: str = "",
    ) -> EvaluationResult:
        """Score a single generated summary against its reference."""
        scores = self.scorer.score(reference, generated)

        source_words = len(source_text.split()) if source_text else 0
        gen_words = len(generated.split())
        ref_words = len(reference.split())

        return EvaluationResult(
            doc_id=doc_id,
            generated_summary=generated,
            reference_summary=reference,
            rouge_scores=ROUGEScores(
                rouge1_precision=scores["rouge1"].precision,
                rouge1_recall=scores["rouge1"].recall,
                rouge1_f1=scores["rouge1"].fmeasure,
                rouge2_precision=scores["rouge2"].precision,
                rouge2_recall=scores["rouge2"].recall,
                rouge2_f1=scores["rouge2"].fmeasure,
                rougeL_precision=scores["rougeL"].precision,
                rougeL_recall=scores["rougeL"].recall,
                rougeL_f1=scores["rougeL"].fmeasure,
            ),
            source_length=source_words,
            generated_length=gen_words,
            reference_length=ref_words,
            compression_ratio=gen_words / max(source_words, 1),
        )

    def score_batch(
        self,
        generated_summaries: list[str],
        reference_summaries: list[str],
        doc_ids: list[str] = None,
        source_texts: list[str] = None,
    ) -> list[EvaluationResult]:
        """Score parallel lists of generated and reference summaries."""
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(generated_summaries))]
        if source_texts is None:
            source_texts = [""] * len(generated_summaries)

        results = []
        for gen, ref, doc_id, src in zip(generated_summaries, reference_summaries, doc_ids, source_texts):
            results.append(self.score_single(gen, ref, doc_id, src))
        return results

    @staticmethod
    def aggregate_scores(results: list[EvaluationResult]) -> dict:
        """Compute mean and std of ROUGE scores and length statistics across results."""
        if not results:
            return {}

        # F1 scores
        r1_f1 = [r.rouge_scores.rouge1_f1 for r in results]
        r2_f1 = [r.rouge_scores.rouge2_f1 for r in results]
        rL_f1 = [r.rouge_scores.rougeL_f1 for r in results]

        # Precision and recall
        r1_p = [r.rouge_scores.rouge1_precision for r in results]
        r1_r = [r.rouge_scores.rouge1_recall for r in results]
        r2_p = [r.rouge_scores.rouge2_precision for r in results]
        r2_r = [r.rouge_scores.rouge2_recall for r in results]
        rL_p = [r.rouge_scores.rougeL_precision for r in results]
        rL_r = [r.rouge_scores.rougeL_recall for r in results]

        gen_lens = [r.generated_length for r in results]
        ref_lens = [r.reference_length for r in results]
        compressions = [r.compression_ratio for r in results if r.compression_ratio > 0]

        return {
            "n_documents": len(results),
            "rouge1": {
                "f1_mean": float(np.mean(r1_f1)),
                "f1_std": float(np.std(r1_f1)),
                "precision_mean": float(np.mean(r1_p)),
                "recall_mean": float(np.mean(r1_r)),
            },
            "rouge2": {
                "f1_mean": float(np.mean(r2_f1)),
                "f1_std": float(np.std(r2_f1)),
                "precision_mean": float(np.mean(r2_p)),
                "recall_mean": float(np.mean(r2_r)),
            },
            "rougeL": {
                "f1_mean": float(np.mean(rL_f1)),
                "f1_std": float(np.std(rL_f1)),
                "precision_mean": float(np.mean(rL_p)),
                "recall_mean": float(np.mean(rL_r)),
            },
            "length_stats": {
                "avg_generated_words": float(np.mean(gen_lens)),
                "avg_reference_words": float(np.mean(ref_lens)),
                "avg_compression_ratio": float(np.mean(compressions)) if compressions else 0.0,
            },
        }

    @staticmethod
    def print_report(aggregate: dict, title: str = "Summarization Evaluation Report"):
        """Pretty-print an aggregate evaluation report to the console."""
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(f"  Documents evaluated: {aggregate.get('n_documents', 0)}")
        print(f"{'─' * 60}")

        for metric in ["rouge1", "rouge2", "rougeL"]:
            data = aggregate.get(metric, {})
            print(f"  {metric.upper()}")
            print(f"    F1:        {data.get('f1_mean', 0):.4f} ± {data.get('f1_std', 0):.4f}")
            print(f"    Precision: {data.get('precision_mean', 0):.4f}")
            print(f"    Recall:    {data.get('recall_mean', 0):.4f}")

        lens = aggregate.get("length_stats", {})
        print(f"{'─' * 60}")
        print(f"  LENGTH STATS")
        print(f"    Avg generated length: {lens.get('avg_generated_words', 0):.1f} words")
        print(f"    Avg reference length:  {lens.get('avg_reference_words', 0):.1f} words")
        print(f"    Avg compression ratio: {lens.get('avg_compression_ratio', 0):.4f}")
        print(f"{'=' * 60}\n")

    @staticmethod
    def qualitative_spot_check(results: list[EvaluationResult], n: int = 3) -> list[dict]:
        """Return the worst, best, and median examples by ROUGE-1 F1 for manual review."""
        sorted_results = sorted(results, key=lambda r: r.rouge_scores.rouge1_f1)

        spot_checks = []

        # Worst performer
        if sorted_results:
            worst = sorted_results[0]
            spot_checks.append({
                "label": "LOWEST ROUGE-1 F1",
                "doc_id": worst.doc_id,
                "rouge1_f1": worst.rouge_scores.rouge1_f1,
                "generated": worst.generated_summary,
                "reference": worst.reference_summary,
            })

        # Best performer
        if sorted_results:
            best = sorted_results[-1]
            spot_checks.append({
                "label": "HIGHEST ROUGE-1 F1",
                "doc_id": best.doc_id,
                "rouge1_f1": best.rouge_scores.rouge1_f1,
                "generated": best.generated_summary,
                "reference": best.reference_summary,
            })

        # Median performer
        if len(sorted_results) >= 3:
            mid = sorted_results[len(sorted_results) // 2]
            spot_checks.append({
                "label": "MEDIAN ROUGE-1 F1",
                "doc_id": mid.doc_id,
                "rouge1_f1": mid.rouge_scores.rouge1_f1,
                "generated": mid.generated_summary,
                "reference": mid.reference_summary,
            })

        return spot_checks
