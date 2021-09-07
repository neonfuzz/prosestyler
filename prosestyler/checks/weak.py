

"""
Check for weak words that could probably use stronger synonyms.

Classes:
    Weak - said weak word checker
"""


from .base_check import BaseCheck
from ..resources.weak_lists import WEAK_ADJS, WEAK_MODALS, WEAK_NOUNS, \
    WEAK_VERBS
from ..tools.thesaurus import THESAURUS


class Weak(BaseCheck):
    """
    Check a text's use of weak words.

    Arguments:
        text (Text) - the text to check

    Iterates over each Sentence and applies a homophone check.
    Text is saved and cleaned after each iteration.
    """

    def __repr__(self):
        """Represent Weak with a string."""
        return 'Weak Words'

    def _check_sent(self, sentence, ignore_list=None):
        errors, suggests, ignore_list, messages = super()._check_sent(
            sentence, ignore_list)

        for node in sentence.nodes:
            text = node.text
            lemma = node.lemma_
            pos = node.tag_
            try:
                idx = sentence.inds[node.i-sentence.nodes.start]
            except AttributeError:
                idx = sentence.inds[node.i]
            tup = ([text], [idx])

            if tup not in ignore_list:
                if pos.startswith('V') \
                        and node.dep_ != 'aux' \
                        and lemma in WEAK_VERBS:
                    errors += [tup]
                    suggests += [THESAURUS.get_synonyms(text)]
                elif lemma in WEAK_ADJS \
                        or lemma in WEAK_MODALS \
                        or lemma in WEAK_NOUNS:
                    errors += [tup]
                    suggests += [THESAURUS.get_synonyms(text)]
        messages = [None] * len(errors)

        return errors, suggests, ignore_list, messages
