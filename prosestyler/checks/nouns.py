
"""
Check for clunky noun phrases.

Functions:
    big_noun_phrases - detect clunky noun phrases
"""


def _check_consecutive(span):
    """Check for noun phrases with 3+ nouns."""
    nouns = [t for t in span if t.tag_.startswith('NN')]
    return len(nouns) >= 3


def _check_long(span):
    """Check for noun phrases with 5+ non-trivial tokens."""
    nontrivial = [t for t in span if not t.is_stop and not t.is_punct]
    return len(nontrivial) >= 5


def big_noun_phrases(nodes):
    """
    Detect clunky noun phrases.

    Arguments:
        nodes (spacy.tokens.Span) - sentence to check

    Returns:
        noun_phrases (list of `Span`s) - clunky noun phrases
    """
    noun_phrases = [n for n in nodes.noun_chunks
            if _check_long(n)
            or _check_consecutive(n)]
    return noun_phrases
