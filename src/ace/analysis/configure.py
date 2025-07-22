from .analyzer import Analyzer
from .validators import GrammarValidator


def configure_analyzer(
    grammar_check: bool = True,
) -> Analyzer:
    """
    Configure the analyzer with default validators.
    """
    analyzer = Analyzer()

    # Register default validators
    if grammar_check:
        analyzer.register(GrammarValidator())

    return analyzer
