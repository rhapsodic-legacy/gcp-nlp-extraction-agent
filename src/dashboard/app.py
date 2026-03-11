"""Evaluation Dashboard — Streamlit app for visualising extraction and
summarisation metrics.

Panels:
  1. Ensemble Extraction Comparison (per-extractor entity counts, agreement)
  2. Needle-in-a-Haystack heatmaps (entity type x position x extractor)
  3. ROUGE / BERTScore distribution charts
  4. Interactive document explorer

Run:
    streamlit run src/dashboard/app.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure the project root is on the path so `src` imports work.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Extraction & Summarisation Dashboard",
    page_icon="::bar_chart::",
    layout="wide",
)

st.title("NLP Extraction & Summarisation Dashboard")


# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.header("Configuration")

data_path = st.sidebar.text_input(
    "Documents JSONL",
    value="data/documents.jsonl",
)

num_docs = st.sidebar.slider("Documents to evaluate", 3, 50, 10)
num_needles = st.sidebar.slider("Needles to inject", 3, 8, 5)
run_gemini = st.sidebar.checkbox("Include Gemini extractor", value=False)
api_key = st.sidebar.text_input("Google API Key", type="password",
                                  value=os.environ.get("GOOGLE_API_KEY", ""))

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key


# ── Data loading ─────────────────────────────────────────────────────

@st.cache_data
def load_documents(path: str, limit: int) -> list:
    """Load documents from JSONL."""
    from src.data.loader import Document

    docs = []
    p = Path(path)
    if not p.exists():
        return docs
    with open(p) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            obj = json.loads(line)
            docs.append(Document(
                id=obj.get("id", f"doc_{i}"),
                text=obj.get("text", ""),
                source_type=obj.get("source_type", "unknown"),
                metadata=obj.get("metadata", {}),
            ))
    return docs


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Ensemble Extraction
# ═══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "Ensemble Extraction",
    "Needle-in-a-Haystack",
    "Summarisation Metrics",
    "Document Explorer",
])


with tab1:
    st.header("Ensemble Extraction Comparison")
    st.markdown("Run all extractors on a document and compare entity coverage and agreement.")

    docs = load_documents(data_path, num_docs)

    if not docs:
        st.warning(f"No documents found at `{data_path}`. Load data first.")
    else:
        doc_options = {f"{d.id} ({d.source_type})": i for i, d in enumerate(docs)}
        selected_doc_label = st.selectbox("Select document", list(doc_options.keys()))
        selected_doc = docs[doc_options[selected_doc_label]]

        with st.expander("Document text", expanded=False):
            st.text(selected_doc.text[:3000])

        if st.button("Run Ensemble Extraction", key="ensemble_btn"):
            with st.spinner("Running extractors..."):
                from src.extraction.ensemble import EnsembleExtractor

                extractor = EnsembleExtractor(
                    api_key=api_key if run_gemini else None,
                    enable_gemini=run_gemini and bool(api_key),
                )
                result = extractor.extract(selected_doc.text)

            # ── Summary metrics ─────────────────────
            col1, col2, col3 = st.columns(3)
            col1.metric("Total entities", len(result.entities))
            col2.metric(
                "High confidence (2+ sources)",
                sum(1 for e in result.entities if e.confidence >= 0.66),
            )
            col3.metric("Extractors used", len(result.per_extractor))

            # ── Per-extractor entity counts ─────────
            st.subheader("Entities per extractor")
            counts = {
                name: len(getattr(res, "entities", []))
                for name, res in result.per_extractor.items()
            }
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(counts.keys(), counts.values(), color=["#6366f1", "#f59e0b", "#22c55e"][:len(counts)])
            ax.set_ylabel("Entity count")
            ax.set_title("Raw entity count by extractor")
            for bar, val in zip(bars, counts.values()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(val), ha="center", fontsize=11, fontweight="bold")
            st.pyplot(fig)
            plt.close()

            # ── Agreement chart ─────────────────────
            st.subheader("Cross-extractor agreement")
            agreement_counts = {1: 0, 2: 0, 3: 0}
            for e in result.entities:
                n = len(e.sources)
                agreement_counts[min(n, 3)] += 1

            fig2, ax2 = plt.subplots(figsize=(5, 3))
            labels = ["1 extractor\n(low)", "2 extractors\n(medium)", "3 extractors\n(high)"]
            values = [agreement_counts.get(i, 0) for i in [1, 2, 3]]
            colours = ["#ef4444", "#f59e0b", "#22c55e"]
            ax2.bar(labels[:len(result.per_extractor)], values[:len(result.per_extractor)],
                    color=colours[:len(result.per_extractor)])
            ax2.set_ylabel("Entity count")
            ax2.set_title("Entity agreement distribution")
            st.pyplot(fig2)
            plt.close()

            # ── Entity table ────────────────────────
            st.subheader("Merged entities")
            df = pd.DataFrame([e.to_dict() for e in result.entities])
            if not df.empty:
                df["sources"] = df["sources"].apply(lambda x: ", ".join(x))
                df["variants"] = df["variants"].apply(lambda x: ", ".join(x))
                st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Needle-in-a-Haystack
# ═══════════════════════════════════════════════════════════════════

with tab2:
    st.header("Needle-in-a-Haystack Evaluation")
    st.markdown(
        "Inject synthetic facts with known entities into the corpus, "
        "then measure extraction recall across positions and entity types."
    )

    docs2 = load_documents(data_path, num_docs)

    if not docs2:
        st.warning("No documents loaded.")
    elif st.button("Run Needle-in-a-Haystack", key="needle_btn"):
        import copy
        from src.evaluation.needle_haystack import NeedleHaystackEvaluator
        from src.extraction.spacy_baseline import SpacyExtractor
        from src.extraction.presidio_extract import PresidioExtractor

        evaluator = NeedleHaystackEvaluator()

        extractors_to_run = {
            "SpaCy": SpacyExtractor(),
            "Presidio": PresidioExtractor(),
        }

        if run_gemini and api_key:
            from src.extraction.vertex_extract import GeminiExtractor
            extractors_to_run["Gemini"] = GeminiExtractor(api_key=api_key)

        # Also run ensemble
        from src.extraction.ensemble import EnsembleExtractor
        extractors_to_run["Ensemble"] = EnsembleExtractor(
            api_key=api_key if (run_gemini and api_key) else None,
            enable_gemini=run_gemini and bool(api_key),
        )

        all_results = {}
        progress = st.progress(0)
        total = len(extractors_to_run)

        for i, (name, ext) in enumerate(extractors_to_run.items()):
            with st.spinner(f"Running {name}..."):
                # Deep copy docs so each extractor gets fresh haystack
                docs_copy = copy.deepcopy(docs2)
                hay_docs, injections = evaluator.build_haystack(docs_copy, num_needles=num_needles)
                results = evaluator.evaluate_extraction(hay_docs, injections, ext)
                all_results[name] = {
                    "results": results,
                    "stats": evaluator.report(results),
                }
            progress.progress((i + 1) / total)

        progress.empty()

        # ── Summary table ───────────────────────
        st.subheader("Overall recall by extractor")
        summary_rows = []
        for name, data in all_results.items():
            stats = data["stats"]
            summary_rows.append({
                "Extractor": name,
                "Detection rate": f"{stats['detection_rate'] * 100:.0f}%",
                "Avg recall": f"{stats['average_recall']:.3f}",
                "Needles found": f"{stats['detected']}/{stats['total_needles']}",
            })
        st.table(pd.DataFrame(summary_rows))

        # ── Heatmap: entity type x extractor ────
        st.subheader("Entity recall by type and extractor")

        # Collect all entity types
        all_types = set()
        for data in all_results.values():
            all_types.update(data["stats"].get("by_entity_type", {}).keys())
        all_types = sorted(all_types)
        extractor_names = list(all_results.keys())

        if all_types:
            heatmap_data = np.zeros((len(all_types), len(extractor_names)))
            for j, name in enumerate(extractor_names):
                by_type = all_results[name]["stats"].get("by_entity_type", {})
                for i, etype in enumerate(all_types):
                    info = by_type.get(etype, {"found": 0, "total": 1})
                    heatmap_data[i, j] = info["found"] / max(info["total"], 1)

            fig3, ax3 = plt.subplots(figsize=(max(6, len(extractor_names) * 2), max(4, len(all_types) * 0.6)))
            im = ax3.imshow(heatmap_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax3.set_xticks(range(len(extractor_names)))
            ax3.set_xticklabels(extractor_names, fontsize=11)
            ax3.set_yticks(range(len(all_types)))
            ax3.set_yticklabels(all_types, fontsize=11)
            ax3.set_title("Entity type recall heatmap", fontsize=13)
            plt.colorbar(im, ax=ax3, label="Recall")

            # Annotate cells
            for i in range(len(all_types)):
                for j in range(len(extractor_names)):
                    val = heatmap_data[i, j]
                    color = "white" if val < 0.5 else "black"
                    ax3.text(j, i, f"{val:.0%}", ha="center", va="center",
                             fontsize=10, fontweight="bold", color=color)

            st.pyplot(fig3)
            plt.close()

        # ── Heatmap: position x extractor ───────
        st.subheader("Recall by position band")
        positions = ["early", "middle", "deep"]
        pos_data = np.zeros((len(positions), len(extractor_names)))
        for j, name in enumerate(extractor_names):
            by_pos = all_results[name]["stats"].get("by_position", {})
            for i, pos in enumerate(positions):
                info = by_pos.get(pos, {"avg_recall": 0})
                pos_data[i, j] = info.get("avg_recall", 0)

        fig4, ax4 = plt.subplots(figsize=(max(6, len(extractor_names) * 2), 3))
        im2 = ax4.imshow(pos_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax4.set_xticks(range(len(extractor_names)))
        ax4.set_xticklabels(extractor_names, fontsize=11)
        ax4.set_yticks(range(len(positions)))
        ax4.set_yticklabels([p.title() for p in positions], fontsize=11)
        ax4.set_title("Recall by document position", fontsize=13)
        plt.colorbar(im2, ax=ax4, label="Recall")

        for i in range(len(positions)):
            for j in range(len(extractor_names)):
                val = pos_data[i, j]
                color = "white" if val < 0.5 else "black"
                ax4.text(j, i, f"{val:.0%}", ha="center", va="center",
                         fontsize=10, fontweight="bold", color=color)

        st.pyplot(fig4)
        plt.close()

        # ── Per-needle breakdown ────────────────
        st.subheader("Per-needle detail")
        for name, data in all_results.items():
            with st.expander(f"{name} — individual needle results"):
                for r in data["results"]:
                    status = "FOUND" if r.found else "MISSED"
                    st.markdown(f"**[{status}]** pos={r.position} recall={r.recall:.2f}")
                    st.text(r.needle_text[:100])
                    for etype, was_found in r.found_entities.items():
                        marker = "+" if was_found else "-"
                        st.text(f"  {marker} {etype}: {r.expected_entities[etype]}")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — Summarisation Metrics
# ═══════════════════════════════════════════════════════════════════

with tab3:
    st.header("Summarisation Evaluation")
    st.markdown("ROUGE and BERTScore metrics for generated summaries against gold-standard references.")

    # Allow uploading pre-computed results or running live
    results_file = st.file_uploader("Upload evaluation results JSON (optional)", type=["json"])

    if results_file:
        eval_data = json.loads(results_file.read())
        st.json(eval_data)

    else:
        st.info(
            "To run live summarisation evaluation, provide a Google API key "
            "in the sidebar and ensure CNN/DailyMail data is available."
        )

        if st.button("Run ROUGE Evaluation (local extractive baseline)", key="rouge_btn"):
            with st.spinner("Computing ROUGE scores..."):
                from src.summarization.evaluation import SummarizationEvaluator

                evaluator = SummarizationEvaluator()

                # Generate simple extractive baselines for demo
                docs3 = load_documents(data_path, min(num_docs, 20))
                if not docs3:
                    st.warning("No documents available.")
                else:
                    # Use first sentence as extractive summary, full text as reference
                    generated = []
                    references = []
                    doc_ids = []
                    for doc in docs3:
                        sentences = doc.text.split(".")
                        if len(sentences) >= 2:
                            gen = ". ".join(sentences[:2]) + "."
                            ref = doc.text
                            generated.append(gen)
                            references.append(ref)
                            doc_ids.append(doc.id)

                    if generated:
                        results = evaluator.score_batch(generated, references, doc_ids)
                        agg = evaluator.aggregate_scores(results)

                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("ROUGE-1 F1", f"{agg['rouge1']['f1_mean']:.4f}")
                        col2.metric("ROUGE-2 F1", f"{agg['rouge2']['f1_mean']:.4f}")
                        col3.metric("ROUGE-L F1", f"{agg['rougeL']['f1_mean']:.4f}")

                        # Distribution chart
                        fig5, axes = plt.subplots(1, 3, figsize=(14, 4))

                        for ax, (metric_name, key) in zip(axes, [
                            ("ROUGE-1 F1", "rouge1_f1"),
                            ("ROUGE-2 F1", "rouge2_f1"),
                            ("ROUGE-L F1", "rougeL_f1"),
                        ]):
                            scores = [getattr(r.rouge_scores, key) for r in results]
                            ax.hist(scores, bins=10, color="#6366f1", alpha=0.8, edgecolor="white")
                            ax.axvline(np.mean(scores), color="#ef4444", linestyle="--", label=f"Mean: {np.mean(scores):.3f}")
                            ax.set_title(metric_name, fontsize=12)
                            ax.set_xlabel("Score")
                            ax.legend(fontsize=9)

                        plt.tight_layout()
                        st.pyplot(fig5)
                        plt.close()

                        # Spot check table
                        spots = evaluator.qualitative_spot_check(results)
                        if spots:
                            st.subheader("Spot check (best / worst / median)")
                            st.table(pd.DataFrame(spots)[["label", "doc_id", "rouge1_f1"]])


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — Document Explorer
# ═══════════════════════════════════════════════════════════════════

with tab4:
    st.header("Document Explorer")

    docs4 = load_documents(data_path, num_docs)

    if not docs4:
        st.warning("No documents loaded.")
    else:
        # Source type breakdown
        types_count = {}
        for d in docs4:
            types_count[d.source_type] = types_count.get(d.source_type, 0) + 1

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Source distribution")
            fig6, ax6 = plt.subplots(figsize=(4, 4))
            ax6.pie(types_count.values(), labels=types_count.keys(),
                    autopct="%1.0f%%", colors=["#6366f1", "#f59e0b", "#22c55e", "#3b82f6"])
            st.pyplot(fig6)
            plt.close()

        with col2:
            st.subheader("Text length distribution")
            lengths = [len(d.text.split()) for d in docs4]
            fig7, ax7 = plt.subplots(figsize=(6, 4))
            ax7.hist(lengths, bins=15, color="#6366f1", alpha=0.8, edgecolor="white")
            ax7.set_xlabel("Word count")
            ax7.set_ylabel("Documents")
            ax7.axvline(np.mean(lengths), color="#ef4444", linestyle="--",
                        label=f"Mean: {np.mean(lengths):.0f}")
            ax7.legend()
            st.pyplot(fig7)
            plt.close()

        # Document viewer
        st.subheader("Browse documents")
        for i, doc in enumerate(docs4):
            with st.expander(f"{doc.id} ({doc.source_type}) — {len(doc.text.split())} words"):
                st.text(doc.text[:2000])
                if doc.metadata:
                    st.json(doc.metadata)
