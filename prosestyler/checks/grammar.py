"""
Provide a grammar checker.

Classes:
    Grammar - said grammar checker
"""

import language_tool_python  # Grammar Check

from .base_check import BaseCheck
from ..tools.helper_functions import fromx_to_id


class Grammar(BaseCheck):
    """
    Check a text's grammar.

    Arguments:
        text (Text) - the text to check

    Iterates over each Sentence and applies a grammar check.
    Text is saved and cleaned after each iteration.
    """

    def __init__(self, lang="en_US"):
        """
        Initialize Grammar.

        Optional Arguments:
            lang (str) - language to check (default: 'en_US')
        """
        super().__init__()
        self._gram = language_tool_python.LanguageTool(lang)

    def __repr__(self):
        """Represent Grammar with a string."""
        return "Grammar"

    def _check_sent(self, sentence, ignore_list=None):
        errors, suggests, ignore_list, messages = super()._check_sent(
            sentence, ignore_list
        )

        errors_gram = self._gram.check(sentence.string)
        # Don't check for smart quotes
        errors_gram = [
            err
            for err in errors_gram
            if err.ruleId != "EN_QUOTES"  # No smartquotes.
            and not err.ruleId.startswith("MORFOLOGIK")  # No spellcheck.
        ]
        for err in errors_gram:
            fromx = err.offset
            tox = fromx + err.errorLength
            ids = fromx_to_id(fromx, tox, sentence.tokens)
            toks = [sentence.tokens[i] for i in ids]
            errors += [(toks, ids)]
            # TODO: I think this would mess up suggestion/message
            #       order if errors wind up in the ignore list.
            errors = [e for e in errors if e not in ignore_list]
            suggests += [err.replacements]
            messages += [err.message]

        return errors, suggests, ignore_list, messages
