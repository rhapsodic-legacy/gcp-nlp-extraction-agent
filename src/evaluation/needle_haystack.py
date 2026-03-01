"""Needle-in-a-haystack evaluation for extraction and agent retrieval.

The idea is simple but powerful: take a large corpus, inject known
"needles" (specific entities, facts, or statements) at controlled
positions, then test whether our extraction pipeline can find them.

This gives us something most NLP prototypes lack: an objective,
quantifiable recall metric. Instead of "the summary looks reasonable,"
we get "the system found 47 out of 50 injected entities." That's the
kind of number you can put in a report and defend.

It's also directly relevant to the agent scenario — if a user asks
"did anyone mention a product recall in Q3?", the agent needs to find
that needle in a haystack of thousands of reviews and tickets. This
module tests exactly that capability.
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..data.loader import Document


# Needles — synthetic but realistic facts that are distinctive enough
# to unambiguously detect in extraction output
DEFAULT_NEEDLES = [
    {
        "text": "Dr. Evelyn Thorncastle reported a critical malfunction in the XR-7 stabilizer unit on March 14th, 2024.",
        "entities": {"PERSON": "Dr. Evelyn Thorncastle", "PRODUCT": "XR-7 stabilizer unit", "DATE": "March 14th, 2024"},
        "topic": "product defect",
    },
    {
        "text": "Zenithra Corp announced a voluntary recall of 12,000 units from their Brisbane warehouse due to overheating batteries.",
        "entities": {"ORG": "Zenithra Corp", "CARDINAL": "12,000", "GPE": "Brisbane", "ISSUE": "overheating batteries"},
        "topic": "recall",
    },
    {
        "text": "Customer satisfaction in the Nordic region dropped to 34% after the firmware update released by Kelvoran Systems on July 9th.",
        "entities": {"LOC": "Nordic region", "PERCENT": "34%", "ORG": "Kelvoran Systems", "DATE": "July 9th"},
        "topic": "customer satisfaction",
    },
    {
        "text": "The internal audit by Priya Ramanathan found that shipping delays from the Mombasa facility averaged 11.3 days in Q4.",
        "entities": {"PERSON": "Priya Ramanathan", "GPE": "Mombasa", "CARDINAL": "11.3 days", "DATE": "Q4"},
        "topic": "shipping delays",
    },
    {
        "text": "Arcturus Medical's CEO, Jonah Whitfield, confirmed that the Helix-9 diagnostic tool passed FDA approval on November 2nd, 2025.",
        "entities": {"ORG": "Arcturus Medical", "PERSON": "Jonah Whitfield", "PRODUCT": "Helix-9 diagnostic tool", "DATE": "November 2nd, 2025"},
        "topic": "regulatory approval",
    },
    {
        "text": "A whistleblower at Ferrovian Industries leaked documents showing $4.2 million in undisclosed expenses at the São Paulo office.",
        "entities": {"ORG": "Ferrovian Industries", "MONEY": "$4.2 million", "GPE": "São Paulo"},
        "topic": "financial fraud",
    },
    {
        "text": "Professor Liang Wei published findings that the Caspian-3 algorithm reduces processing latency by 67% compared to baseline.",
        "entities": {"PERSON": "Professor Liang Wei", "PRODUCT": "Caspian-3 algorithm", "PERCENT": "67%"},
        "topic": "performance improvement",
    },
    {
        "text": "Orinoco Logistics reported that their partnership with TransNordic Shipping saved €8.5 million annually since the Rotterdam hub opened.",
        "entities": {"ORG": "Orinoco Logistics", "ORG2": "TransNordic Shipping", "MONEY": "€8.5 million", "GPE": "Rotterdam"},
        "topic": "cost savings",
    },
]


@dataclass
class NeedleResult:
    """Result of searching for a single needle in the haystack."""

    needle_id: str
    needle_text: str
    position: str  # "early", "middle", "deep"
    expected_entities: dict
    found_entities: dict = field(default_factory=dict)
    recall: float = 0.0
    found: bool = False

    def to_dict(self) -> dict:
        return {
            "needle_id": self.needle_id,
            "position": self.position,
            "expected": self.expected_entities,
            "found": self.found_entities,
            "recall": self.recall,
            "detected": self.found,
        }


class NeedleHaystackEvaluator:
    """Inject known needles into a document corpus and measure extraction recall.

    Workflow:
    1. build_haystack() — combine real documents with injected needles
    2. evaluate_extraction() — run an extractor and measure what it found
    3. report() — print a summary of recall by position and entity type

    The needles are designed to be unambiguously synthetic — names like
    "Zenithra Corp" and "Dr. Evelyn Thorncastle" won't collide with real
    data, so we can confidently attribute any detection to the needle.
    """

    def __init__(self, needles: list[dict] = None):
        self.needles = needles or DEFAULT_NEEDLES

    def build_haystack(
        self,
        documents: list[Document],
        num_needles: int = 5,
        seed: int = 42,
    ) -> tuple[list[Document], list[dict]]:
        """Build a haystack by injecting needles into real documents.

        Distributes needles across three position bands:
        - early: first 20% of documents
        - middle: 40-60% range
        - deep: last 20% of documents

        Args:
            documents: Real documents to use as the haystack.
            num_needles: How many needles to inject.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (modified document list, list of injection records).
        """
        rng = random.Random(seed)
        num_needles = min(num_needles, len(self.needles))
        selected_needles = rng.sample(self.needles, num_needles)

        # Define position bands
        n = len(documents)
        bands = {
            "early": (0, int(n * 0.2)),
            "middle": (int(n * 0.4), int(n * 0.6)),
            "deep": (int(n * 0.8), n),
        }

        injections = []
        band_names = list(bands.keys())

        for i, needle in enumerate(selected_needles):
            band = band_names[i % len(band_names)]
            lo, hi = bands[band]
            if hi <= lo:
                hi = lo + 1

            insert_idx = rng.randint(lo, min(hi - 1, n - 1))
            needle_id = f"needle_{uuid.uuid4().hex[:8]}"

            # Inject the needle text into the document
            target_doc = documents[insert_idx]
            original_text = target_doc.text
            target_doc.text = f"{original_text}\n\n{needle['text']}"

            injection = {
                "needle_id": needle_id,
                "needle_text": needle["text"],
                "position": band,
                "doc_index": insert_idx,
                "doc_id": target_doc.id,
                "entities": needle["entities"],
                "topic": needle.get("topic", ""),
            }
            injections.append(injection)

        return documents, injections

    def evaluate_extraction(
        self,
        documents: list[Document],
        injections: list[dict],
        extractor,
    ) -> list[NeedleResult]:
        """Run extraction on injected documents and measure needle recall.

        For each needle, checks whether the extractor found the expected
        entities. A needle is "found" if at least one of its key entities
        appears in the extraction output.

        Args:
            documents: Documents with injected needles.
            injections: Injection records from build_haystack().
            extractor: Any extractor with an extract(text) method.

        Returns:
            List of NeedleResult objects with per-needle recall scores.
        """
        results = []

        for injection in injections:
            doc_idx = injection["doc_index"]
            doc = documents[doc_idx]

            # Run extraction
            extraction = extractor.extract(doc.text)

            # Collect all extracted entity texts (lowercased for matching)
            if hasattr(extraction, "entities"):
                extracted_texts = {e.text.lower() for e in extraction.entities}
            else:
                extracted_texts = set()

            # Check which expected entities were found
            expected = injection["entities"]
            found = {}
            for etype, etext in expected.items():
                # Bidirectional substring matching: the expected text may be
                # inside an extracted entity, or vice versa (e.g. Presidio
                # finds "Evelyn Thorncastle" while expected is "Dr. Evelyn
                # Thorncastle")
                etext_lower = etext.lower()
                is_found = any(
                    etext_lower in ext or ext in etext_lower
                    for ext in extracted_texts
                )
                found[etype] = is_found

            total_expected = len(expected)
            total_found = sum(1 for v in found.values() if v)
            recall = total_found / max(total_expected, 1)

            results.append(NeedleResult(
                needle_id=injection["needle_id"],
                needle_text=injection["needle_text"],
                position=injection["position"],
                expected_entities=expected,
                found_entities={k: v for k, v in found.items()},
                recall=recall,
                found=total_found > 0,
            ))

        return results

    @staticmethod
    def report(results: list[NeedleResult]) -> dict:
        """Aggregate needle results and print a report.

        Returns aggregate stats and prints a human-readable summary.
        """
        if not results:
            print("No needle results to report.")
            return {}

        total = len(results)
        detected = sum(1 for r in results if r.found)
        avg_recall = sum(r.recall for r in results) / total

        # By position
        by_position = {}
        for r in results:
            if r.position not in by_position:
                by_position[r.position] = {"total": 0, "detected": 0, "recall_sum": 0.0}
            by_position[r.position]["total"] += 1
            by_position[r.position]["detected"] += 1 if r.found else 0
            by_position[r.position]["recall_sum"] += r.recall

        # By entity type
        entity_hits = {}
        entity_total = {}
        for r in results:
            for etype, was_found in r.found_entities.items():
                entity_total[etype] = entity_total.get(etype, 0) + 1
                entity_hits[etype] = entity_hits.get(etype, 0) + (1 if was_found else 0)

        # Print report
        print(f"\n{'=' * 60}")
        print(f"  Needle-in-a-Haystack Evaluation")
        print(f"{'=' * 60}")
        print(f"  Needles injected:  {total}")
        print(f"  Needles detected:  {detected} ({detected/total*100:.0f}%)")
        print(f"  Average recall:    {avg_recall:.3f}")
        print(f"{'─' * 60}")
        print(f"  BY POSITION")
        for pos in ["early", "middle", "deep"]:
            data = by_position.get(pos, {"total": 0, "detected": 0, "recall_sum": 0})
            if data["total"] > 0:
                avg = data["recall_sum"] / data["total"]
                print(f"    {pos:8s}: {data['detected']}/{data['total']} detected, avg recall {avg:.3f}")
        print(f"{'─' * 60}")
        print(f"  BY ENTITY TYPE")
        for etype in sorted(entity_total.keys()):
            hits = entity_hits.get(etype, 0)
            tot = entity_total[etype]
            print(f"    {etype:12s}: {hits}/{tot} found ({hits/tot*100:.0f}%)")
        print(f"{'=' * 60}\n")

        # Per-needle detail
        print("  INDIVIDUAL NEEDLE RESULTS")
        for r in results:
            status = "FOUND" if r.found else "MISSED"
            print(f"    [{status:6s}] pos={r.position:6s} recall={r.recall:.2f} | {r.needle_text[:70]}...")
            for etype, was_found in r.found_entities.items():
                marker = "+" if was_found else "-"
                print(f"      {marker} {etype}: {r.expected_entities[etype]}")

        return {
            "total_needles": total,
            "detected": detected,
            "detection_rate": detected / total,
            "average_recall": avg_recall,
            "by_position": {
                pos: {
                    "detected": data["detected"],
                    "total": data["total"],
                    "avg_recall": data["recall_sum"] / max(data["total"], 1),
                }
                for pos, data in by_position.items()
            },
            "by_entity_type": {
                etype: {"found": entity_hits.get(etype, 0), "total": entity_total[etype]}
                for etype in entity_total
            },
        }
