"""
Provide a spell-checker.

Classes:
    Speller - said spell-checker
"""


from string import punctuation

from .base_check import BaseCheck
from ..tools.spellcheck import SpellCheck


class Speller(BaseCheck):
    """
    Check a text's spelling.

    Arguments:
        text (Text) - the text to check

    Iterates over each Sentence and applies a spellcheck.
    Text is saved and cleaned after each iteration.
    """

    def __init__(self, lang="en_US"):
        """
        Initialize Speller.

        Optional Arguments:
            lang (str) - language to check (default: 'en_US')
        """
        super().__init__()
        self._dict = SpellCheck(lang)

    def __repr__(self):
        """Represent Speller with a string."""
        return "Spelling"

    def _check_sent(self, sentence, ignore_list=None):
        errors, suggests, ignore_list, messages = super()._check_sent(
            sentence, ignore_list
        )
        nodes = sentence.nodes
        for tok in nodes:
            if tok.ent_iob != 2:
                # If token is part of a named entity, don't spellcheck.
                continue
            if (
                tok.text == " "
                or tok.text == "\n"
                or tok.text == "\n\n"
                or tok.text in punctuation
            ):
                continue
            tup = ([tok.text], [sentence.inds[tok.i - nodes[:].start]])
            if self._dict.check(tok.text) is False and tup not in ignore_list:
                errors += [tup]
        suggests = [self._dict.suggest(err[0][0]) for err in errors]
        messages = [None] * len(errors)
        return errors, suggests, ignore_list, messages
