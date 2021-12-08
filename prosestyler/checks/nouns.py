
"""
Check for clunky noun phrases.

Classes:
    Nouns - said noun phrase checker

Functions:
    big_noun_phrases - detect clunky noun phrases
"""


from .base_check import BaseCheck


def _check_consecutive(span):
    """Check for noun phrases with 4+ nouns."""
    nouns = [t for t in span if t.tag_.startswith('NN')]
    return len(nouns) >= 4


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


class Nouns(BaseCheck):
    """
    Check a text's use of big noun phrases.

    Arguments:
        text (Text) - the text to check

    Iterates over each Sentence and applies an excessive noun phrase check.
    Text is saved and cleaned after each iteration.
    """

    _description = (
        'Excessive noun phrases are exactly what they sound like... '
        'phrases composed of far too many nouns. They are difficult to '
        'parse and slow down your reader. Try breaking the phrase into '
        'smaller chunks and spreading them throughout your sentence.')

    def __repr__(self):
        """Represent Nouns with a string."""
        return 'Excessive Noun Phrases'

    def _check_sent(self, sentence, ignore_list=None):
        errors, suggests, ignore_list, messages = super()._check_sent(
            sentence, ignore_list)

        span_start = sentence.nodes[:].start
        for err in big_noun_phrases(sentence.nodes):
            toks = list(err)
            ids = sentence.inds[err.start-span_start:err.end-span_start]
            tup = (toks, ids)
            errors += [tup]
        suggests = [[]] * len(errors)
        messages = [None] * len(errors)

        return errors, suggests, ignore_list, messages
