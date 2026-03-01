# Agent components require GCP packages; import on demand
def __getattr__(name):
    if name == "CustomerInsightAgent":
        from .agent import CustomerInsightAgent
        return CustomerInsightAgent
    if name == "CriticAgent":
        from .critic import CriticAgent
        return CriticAgent
    if name in ("SearchTool", "ExtractTool", "SentimentTool", "SummarizeTool"):
        from . import tools
        return getattr(tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
