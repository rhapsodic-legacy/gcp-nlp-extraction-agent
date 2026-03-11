"""Text preprocessing pipeline for heterogeneous document corpora.

Light-touch cleaning that removes noise (HTML tags, URLs, template
placeholders, encoding artifacts) and normalizes whitespace while
preserving sentence structure and casing for downstream NER and
summarization.
"""

import re
import string
from typing import Optional

import nltk

from .loader import Document

# Download required NLTK resources if missing
for resource in ["punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Remove encoding artifacts, HTML, URLs, and template placeholders.

    Preserves sentence boundaries and entity casing for downstream
    NER and summarization.
    """
    # Normalize unicode to ASCII
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Collapse repeated punctuation ("!!!" -> "!")
    text = re.sub(r"([!?.]){2,}", r"\1", text)

    # Remove template placeholders (e.g., {product_purchased})
    text = re.sub(r"\{[^}]+\}", "", text)

    return text.strip()


def tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK's Punkt tokenizer."""
    return sent_tokenize(text)


def tokenize_words(text: str, remove_stopwords: bool = False, lowercase: bool = True) -> list[str]:
    """Tokenize text into words with optional stopword removal and lowercasing."""
    if lowercase:
        text = text.lower()
    tokens = word_tokenize(text)
    # Remove bare punctuation tokens
    tokens = [t for t in tokens if t not in string.punctuation]
    if remove_stopwords:
        tokens = [t for t in tokens if t.lower() not in STOP_WORDS]
    return tokens


def get_text_stats(text: str) -> dict:
    """Compute basic text statistics for exploratory analysis.

    Returns character count, word count, sentence count, average word
    length, average sentence length, and stopword ratio.
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
    """Run the full cleaning pipeline on a single text string."""
    return clean_text(text)


def preprocess_documents(
    documents: list[Document],
    compute_stats: bool = False,
) -> list[Document]:
    """Preprocess a list of documents in place.

    Runs clean_text on every document and optionally computes text
    statistics. Modifies documents in place and returns the same list.

    Args:
        documents: List of Document objects to clean.
        compute_stats: If True, attach text stats to each doc's metadata.

    Returns:
        The same list with cleaned text and optionally enriched metadata.
    """
    for doc in documents:
        doc.text = preprocess_text(doc.text)

        if compute_stats:
            doc.metadata["text_stats"] = get_text_stats(doc.text)

    return documents
