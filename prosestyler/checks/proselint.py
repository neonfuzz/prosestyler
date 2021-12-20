"""
Run a ProseLint on text.

Classes:
    ProseLinter - said ProseLint checker
"""


import proselint

from .base_check import BaseCheck
from ..tools.helper_functions import fromx_to_id


class ProseLinter(BaseCheck):
    """
    Check a text with ProseLint.

    Arguments:
        text (Text) - the text to check

    Iterates over each Sentence and applies a homophone check.
    Text is saved and cleaned after each iteration.
    """

    _description = 'Some checks from our friends at ProseLint.'

    def __repr__(self):
        """Represent ProseLinter with a string."""
        return 'ProseLint'

    def _check_sent(self, sentence, ignore_list=None):
        errors, suggests, ignore_list, messages = super()._check_sent(
            sentence, ignore_list
        )

        linted = proselint.tools.lint(sentence.string)
        for _, message, _, _, fromx, tox, _, _, replacements in linted:
            ids = fromx_to_id(fromx, tox, sentence.tokens)
            toks = [sentence.tokens[i] for i in ids]
            try:
                if toks[-1] == ' ':
                    ids = ids[:-1]
                    toks = toks[:-1]
            except IndexError:
                pass
            else:
                errors += [(toks, ids)]
                errors = [e for e in errors if e not in ignore_list]
                if isinstance(replacements, list):
                    suggests += [replacements]
                elif replacements:
                    suggests += [[replacements]]
                else:
                    suggests += [[]]
                messages += [message]

        return errors, suggests, ignore_list, messages
