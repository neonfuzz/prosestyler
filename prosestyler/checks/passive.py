

"""
Provide a checker for passive voice.

Classes:
    Passive - said passive voice checker
"""


# Many thanks to powerthesaurus.org and proselint.com

# pylint: disable=too-many-lines
# Yes, it's a lot, but it's mostly just dictionary.


from .base_check import BaseCheck


class Passive(BaseCheck):
    """
    Check a text's use of passive voice.

    Arguments:
        text (Text) - the text to check

    Iterates over each Sentence and applies a homophone check.
    Text is saved and cleaned after each iteration.
    """

    def __repr__(self):
        """Represent Passive with a string."""
        return 'Passive Voice'

    def _check_sent(self, sentence, ignore_list=None):
        errors, suggests, ignore_list, messages = super()._check_sent(
            sentence, ignore_list)

        nodes = sentence.nodes
        vbns = [n for n in nodes if 'auxpass' in [c.dep_ for c in n.children]]
        for word in vbns:
            children = list(word.children)
            be_verbs = [c for c in children if c.dep_ == 'auxpass']
            try:
                ids = [sentence.inds[bv.i-nodes.start] for bv in be_verbs]
                ids += [sentence.inds[word.i-nodes.start]]
            except AttributeError:
                ids = [sentence.inds[bv.i] for bv in be_verbs]
                ids += [sentence.inds[word.i]]
            ids.sort()
            toks = [sentence.tokens[i] for i in ids]
            tup = (toks, ids)
            if tup not in ignore_list:
                errors += [tup]
                suggests += [[]]
        messages = [None] * len(errors)

        return errors, suggests, ignore_list, messages
