# Evaluation is always available (local, no GCP required)
from .evaluation import SummarizationEvaluator, EvaluationResult, ROUGEScores

# Gemini summarizer requires vertexai — import on demand
def __getattr__(name):
    if name == "GeminiSummarizer":
        from .vertex_summarize import GeminiSummarizer
        return GeminiSummarizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
