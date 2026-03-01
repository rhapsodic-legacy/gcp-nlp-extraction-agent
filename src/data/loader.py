"""Unified data loading for heterogeneous text corpora.

Here's a thing I learned building the Apple II: if you want your system to handle
different kinds of input gracefully, you need a common interface. A floppy disk
and a cassette tape are totally different hardware, but the computer shouldn't
have to care — it just wants bytes.

Same idea here. Reviews, support tickets, news articles, Reddit posts — they all
come in different formats with different column names and different quirks. This
module normalizes everything into a single Document dataclass so the rest of the
pipeline never has to think about where the data came from. One connector per
source, one universal output format. Clean wiring.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


@dataclass
class Document:
    """Universal document representation — the common bus for all source types.

    Think of this as the data bus in a computer. Every peripheral speaks a
    different protocol, but they all put their bits on the same bus. Same deal:
    reviews have ratings, tickets have priorities, news has highlights — but
    they all have text, an ID, and a source label.
    """

    id: str
    text: str
    source_type: str  # "review", "support_ticket", "news", "reddit"
    metadata: dict = field(default_factory=dict)
    created_at: Optional[str] = None

    def __repr__(self):
        preview = self.text[:80] + "..." if len(self.text) > 80 else self.text
        return f"Document(id={self.id!r}, source={self.source_type}, text={preview!r})"


def _load_config(config_path: str = "config/gcp_config.yaml") -> dict:
    """Grab settings from the config file. Falls back to sane defaults if missing."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f)
    return {"max_documents_per_source": 5000}


def load_reviews(data_dir: str = "data/raw/reviews/archive", max_docs: int = None) -> list[Document]:
    """Load Amazon product reviews into Document format.

    These reviews are messy in the best way — real humans typing real opinions.
    The Datafiniti CSV format has some column name variations across files,
    so we sniff for the right text column and handle it. The metadata carries
    along product names, brands, ratings, and dates because that context is
    gold for downstream extraction.
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Sniff for the text column — Datafiniti files aren't consistent here
    text_col = "reviews.text" if "reviews.text" in combined.columns else "text"
    combined = combined.dropna(subset=[text_col])
    combined = combined[combined[text_col].str.strip().str.len() > 0]

    if max_docs:
        combined = combined.head(max_docs)

    documents = []
    for idx, row in combined.iterrows():
        doc = Document(
            id=f"review_{idx}",
            text=str(row[text_col]),
            source_type="review",
            metadata={
                "product_name": row.get("name", ""),
                "brand": row.get("brand", ""),
                "rating": row.get("reviews.rating", row.get("rating", None)),
                "title": row.get("reviews.title", row.get("title", "")),
                "categories": row.get("categories", ""),
            },
            created_at=str(row.get("reviews.date", row.get("dateAdded", ""))),
        )
        documents.append(doc)
    return documents


def load_support_tickets(
    filepath: str = "data/raw/support_tickets/customer_support_tickets.csv",
    max_docs: int = None,
) -> list[Document]:
    """Load customer support tickets into Document format.

    Fair warning: these tickets are noisy. Lots of template placeholders like
    {product_purchased} that the preprocessing step will strip out, plus some
    real gems of auto-generated gibberish. But that's realistic data for you —
    you don't get to pick your input in production. The metadata is rich though:
    ticket type, priority, product, resolution status, satisfaction rating.
    """
    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    df = df.dropna(subset=["Ticket Description"])
    df = df[df["Ticket Description"].str.strip().str.len() > 0]

    if max_docs:
        df = df.head(max_docs)

    documents = []
    for idx, row in df.iterrows():
        doc = Document(
            id=f"ticket_{row.get('Ticket ID', idx)}",
            text=str(row["Ticket Description"]),
            source_type="support_ticket",
            metadata={
                "subject": row.get("Ticket Subject", ""),
                "type": row.get("Ticket Type", ""),
                "priority": row.get("Ticket Priority", ""),
                "product": row.get("Product Purchased", ""),
                "resolution": row.get("Resolution", ""),
                "status": row.get("Ticket Status", ""),
                "satisfaction": row.get("Customer Satisfaction Rating", None),
                "channel": row.get("Ticket Channel", ""),
            },
            created_at=str(row.get("Date of Purchase", "")),
        )
        documents.append(doc)
    return documents


def load_news(data_dir: str = "data/raw/news", max_docs: int = None) -> list[Document]:
    """Load CNN/DailyMail articles via HuggingFace datasets library.

    This is the crown jewel of the corpus — real journalism with human-written
    summary highlights. That means we've got a gold standard to measure our
    summarization against, which is rare and wonderful. It's like having a
    known-good reference signal to calibrate your instruments.

    Uses streaming mode to avoid downloading the full ~1.3GB dataset to disk.
    Only pulls the articles we actually need. We use the test split (~11.5k
    articles) which is plenty for prototyping.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install `datasets` package: pip install datasets")

    # Stream instead of downloading the full dataset — saves ~1.3GB of disk
    ds = load_dataset(
        "abisee/cnn_dailymail",
        "3.0.0",
        split="test",
        streaming=True,
    )

    documents = []
    for idx, item in enumerate(ds):
        if max_docs and idx >= max_docs:
            break
        doc = Document(
            id=f"news_{item['id']}",
            text=item["article"],
            source_type="news",
            metadata={
                "highlights": item["highlights"],  # the gold standard summaries!
            },
        )
        documents.append(doc)
    return documents


def load_reddit(
    filepath: str = "data/raw/reddit/reddit_wsb.csv",
    max_docs: int = None,
) -> list[Document]:
    """Load Reddit WSB posts into Document format.

    This one's reserved for a future iteration — it's a big, noisy dataset
    full of memes and financial hot takes. Great stress test for extraction
    robustness, but not essential for the MVP. Like overclocking: fun, but
    do the stable build first.
    """
    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    df = df.dropna(subset=["body"])
    df = df[df["body"].str.strip().str.len() > 10]

    if max_docs:
        df = df.head(max_docs)

    documents = []
    for idx, row in df.iterrows():
        text = str(row["body"])
        title = str(row.get("title", ""))
        if title:
            text = f"{title}\n\n{text}"

        doc = Document(
            id=f"reddit_{row.get('id', idx)}",
            text=text,
            source_type="reddit",
            metadata={
                "title": title,
                "score": row.get("score", 0),
                "url": row.get("url", ""),
                "num_comments": row.get("comms_num", 0),
            },
            created_at=str(row.get("timestamp", "")),
        )
        documents.append(doc)
    return documents


def load_all_datasets(
    data_root: str = "data",
    max_per_source: int = None,
    include_reddit: bool = False,
    include_news: bool = True,
) -> list[Document]:
    """Load all datasets into a unified document list.

    This is the one-call-does-it-all function. It grabs every source,
    normalizes them into Documents, and hands you back one big list.
    Like plugging all your peripherals into a USB hub — one cable to
    the computer, everything just works.

    Args:
        data_root: Root data directory.
        max_per_source: Cap per source type (None = load all, up to config limit).
        include_reddit: Include Reddit WSB data (large/noisy, off by default).
        include_news: Include CNN/DailyMail (requires download on first run).

    Returns:
        Combined list of Document objects from all sources.
    """
    if max_per_source is None:
        config = _load_config()
        max_per_source = config.get("max_documents_per_source", 5000)

    all_docs = []

    print(f"Loading reviews (max {max_per_source})...")
    reviews = load_reviews(f"{data_root}/raw/reviews/archive", max_docs=max_per_source)
    all_docs.extend(reviews)
    print(f"  -> {len(reviews)} review documents loaded")

    print(f"Loading support tickets (max {max_per_source})...")
    tickets = load_support_tickets(
        f"{data_root}/raw/support_tickets/customer_support_tickets.csv",
        max_docs=max_per_source,
    )
    all_docs.extend(tickets)
    print(f"  -> {len(tickets)} ticket documents loaded")

    if include_news:
        print(f"Loading news articles (max {max_per_source})...")
        news = load_news(f"{data_root}/raw/news", max_docs=max_per_source)
        all_docs.extend(news)
        print(f"  -> {len(news)} news documents loaded")

    if include_reddit:
        print(f"Loading Reddit posts (max {max_per_source})...")
        reddit = load_reddit(f"{data_root}/raw/reddit/reddit_wsb.csv", max_docs=max_per_source)
        all_docs.extend(reddit)
        print(f"  -> {len(reddit)} Reddit documents loaded")

    print(f"\nTotal: {len(all_docs)} documents loaded")
    return all_docs
