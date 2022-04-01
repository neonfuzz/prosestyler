"""
Check for weak words that could probably use stronger synonyms.

Classes:
    Weak - said weak word checker
"""


from .base_check import BaseCheck
from ..resources.weak_lists import (
    WEAK_ADJS,
    WEAK_MODALS,
    WEAK_NOUNS,
    WEAK_VERBS,
)
from ..tools.thesaurus import THESAURUS


class Weak(BaseCheck):
    """
    Check a text's use of weak words.

    Arguments:
        text (Text) - the text to check

    Iterates over each Sentence and applies a homophone check.
    Text is saved and cleaned after each iteration.
    """

    _description = (
        "Some words are just weak and boring, and make your writing "
        "sound uninspired. Make heavy use of the thesaurus and bring "
        "your writing to life. Tip: These suggestions are just that. "
        "Don't go overboard on fancy jargon or your writing will sound "
        'pompous. If the original "weak" word is the best fit, by all '
        "means stick with that one."
    )

    def __repr__(self):
        """Represent Weak with a string."""
        return "Weak Words"

    def _check_sent(self, sentence, ignore_list=None):
        errors, suggests, ignore_list, messages = super()._check_sent(
            sentence, ignore_list
        )

        for node in sentence.nodes:
            text = node.text
            lemma = node.lemma_
            pos = node.tag_
            try:
                idx = sentence.inds[node.i - sentence.nodes.start]
            except AttributeError:
                idx = sentence.inds[node.i]
            tup = ([text], [idx])

            if tup not in ignore_list:
                if (
                    pos.startswith("V")
                    and not node.dep_.startswith("aux")
                    and lemma in WEAK_VERBS
                ):
                    errors += [tup]
                    suggests += [THESAURUS.get_synonyms(text)]
                elif (
                    lemma in WEAK_ADJS
                    or lemma in WEAK_MODALS
                    or lemma in WEAK_NOUNS
                ):
                    errors += [tup]
                    suggests += [THESAURUS.get_synonyms(text)]
        messages = [None] * len(errors)

        return errors, suggests, ignore_list, messages
