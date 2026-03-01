"""Text preprocessing pipeline for heterogeneous document corpora.

There's an art to preprocessing NLP data, and the art is knowing when to
stop. I've seen people normalize their text into oblivion — lowercase
everything, strip all punctuation, stem every word — and then wonder why
their NER can't find "Apple Inc." anymore. The signal was in the casing
and the punctuation. You destroyed it!

So this pipeline is deliberately light-touch. We clean up the obvious junk
(HTML tags, busted URLs, template placeholders from those support tickets)
and normalize whitespace, but we keep sentence structure and casing intact.
The downstream models — both GCP's NL API and Gemini — are smart enough
to handle real text. Let them do their job.

Think of it like preparing components before soldering: you clean the leads
and tin them, but you don't file them down to nothing.
"""

import re
import string
from typing import Optional

import nltk

from .loader import Document

# Make sure NLTK has what it needs — download quietly if missing
for resource in ["punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Strip out the noise while preserving the signal.

    This is the core cleaning function. It handles the crud that shows up
    across all our source types — garbled unicode, stray HTML, broken URLs,
    and those obnoxious {placeholder} templates from the support ticket
    system. But it deliberately leaves sentence boundaries and entity casing
    alone because NER and summarization need those.
    """
    # Normalize unicode — some of these reviews have wild character encoding
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()

    # Nuke URLs — they're noise for extraction and summarization
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Strip HTML tags (surprisingly common in product reviews)
    text = re.sub(r"<[^>]+>", "", text)

    # Calm down the excited punctuation ("!!!" -> "!")
    text = re.sub(r"([!?.]){2,}", r"\1", text)

    # Remove template placeholders from support tickets (e.g., {product_purchased})
    text = re.sub(r"\{[^}]+\}", "", text)

    return text.strip()


def tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences. NLTK's Punkt tokenizer handles this well."""
    return sent_tokenize(text)


def tokenize_words(text: str, remove_stopwords: bool = False, lowercase: bool = True) -> list[str]:
    """Break text into individual words with optional filtering.

    The stopword removal is useful for EDA and vocabulary analysis, but
    you'd never want it for text going into the extraction pipeline —
    "not working" becomes just "working" if you strip stopwords, and
    that's a pretty important distinction!
    """
    if lowercase:
        text = text.lower()
    tokens = word_tokenize(text)
    # Toss bare punctuation tokens
    tokens = [t for t in tokens if t not in string.punctuation]
    if remove_stopwords:
        tokens = [t for t in tokens if t.lower() not in STOP_WORDS]
    return tokens


def get_text_stats(text: str) -> dict:
    """Compute basic text statistics for exploratory analysis.

    These numbers tell you a lot about your corpus at a glance. Short
    sentences + high stopword ratio = conversational text (reviews).
    Long sentences + low stopword ratio = information-dense text (news).
    It's like reading the specs of a component before you design around it.
    """
    sentences = tokenize_sentences(text)
    words = tokenize_words(text)
    words_no_stop = tokenize_words(text, remove_stopwords=True)

    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "stopword_ratio": 1 - len(words_no_stop) / max(len(words), 1),
    }


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline for a single text string.

    One function, one job: take raw text in, hand clean text out.
    That's it. No surprises.
    """
    return clean_text(text)


def preprocess_documents(
    documents: list[Document],
    compute_stats: bool = False,
) -> list[Document]:
    """Preprocess a list of documents in place.

    Runs clean_text on every document and optionally computes text
    statistics for EDA. Modifies documents in place (returns the
    same list for convenience, but it's not a copy).

    Args:
        documents: List of Document objects to clean up.
        compute_stats: If True, attach text stats to each doc's metadata.
            Handy for exploration, but skip it in production — it's extra cycles.

    Returns:
        The same list with cleaned text and optionally enriched metadata.
    """
    for doc in documents:
        doc.text = preprocess_text(doc.text)

        if compute_stats:
            doc.metadata["text_stats"] = get_text_stats(doc.text)

    return documents
