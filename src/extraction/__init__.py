# SpaCy baseline is always available (local, no GCP required)
from .spacy_baseline import SpacyExtractor

# GCP-native extractors require google-cloud packages — import on demand
def __getattr__(name):
    if name == "GCPEntityExtractor":
        from .gcp_nlp import GCPEntityExtractor
        return GCPEntityExtractor
    if name == "GeminiExtractor":
        from .vertex_extract import GeminiExtractor
        return GeminiExtractor
    if name == "EnsembleExtractor":
        from .ensemble import EnsembleExtractor
        return EnsembleExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
