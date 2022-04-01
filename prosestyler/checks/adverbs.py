"""
Check for adverbs.

Classes:
    Adverbs - said adverb checker
"""


from .base_check import BaseCheck
from ..tools.thesaurus import THESAURUS


class Adverbs(BaseCheck):
    """
    Check a text's use of adverbs.

    Arguments:
        text (Text) - the text to check

    Iterates over each Sentence and applies a homophone check.
    Text is saved and cleaned after each iteration.
    """

    _description = (
        "Adverbs signal a weak verb. Try replacing both with a more "
        'descriptive verb. e.g., "He ran really fast" could become "he '
        'sprinted" or "he dashed." "They walked easily" could become '
        '"they ambled" or "they glided."'
    )

    def __repr__(self):
        """Represent Adverbs with a string."""
        return "Adverbs"

    def _check_sent(self, sentence, ignore_list=None):
        errors, suggests, ignore_list, messages = super()._check_sent(
            sentence, ignore_list
        )

        adv_modified = sorted(
            [
                n
                for n in sentence.nodes
                if "advmod" in [c.dep_ for c in n.children]
            ]
        )
        first_i = sentence.nodes[0].i
        for node in adv_modified:
            adv_node_ids = [
                c.i - first_i
                for c in node.children
                if c.dep_ == "advmod" and c.text.endswith("ly")
            ]
            ids = [sentence.inds[i] for i in adv_node_ids]
            ids += [sentence.inds[node.i - first_i]]
            ids.sort()
            toks = [sentence.tokens[i] for i in ids]
            tup = (toks, ids)
            if (
                adv_node_ids
                and ids[1] - ids[0] <= 5
                and node.tag_ is not None
                and tup not in ignore_list
            ):
                errors += [tup]
                suggests += [THESAURUS.get_synonyms(node.text)]
        messages = [None] * len(errors)

        return errors, suggests, ignore_list, messages
