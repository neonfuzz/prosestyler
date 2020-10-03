

"""
Wrapper around enchant spellcheck for faster intial loading.

Classes:
    SpellCheck - wraps enchant spellcheck with lazy loading
"""


import enchant

from .. import resources


RESOURCE_PATH = resources.__path__[0]


class SpellCheck():
    """
    Wrap enchant spell check with lazy load.

    Instance Attributes:
        checker (enchant.DictWithPWL) - spell check tool

    Methods:
        check - check word spelling
        suggest - provide spelling suggestions
    """
    def __init__(self, lang):
        """
        Initialize SpellCheck.

        Arguments:
            lang (str) - language to check spelling.
        """
        self._lang = lang
        self._checker = None

    @property
    def checker(self):
        """Get Enchant checker, lazy loaded."""
        if self._checker is None:
            self._checker = enchant.DictWithPWL(
                self._lang, RESOURCE_PATH + '/scientific_word_list.txt')
        return self._checker

    def check(self, *args, **kwargs):
        """Check spelling of word. See Enchant documentation."""
        return self.checker.check(*args, **kwargs)

    def suggest(self, *args, **kwargs):
        """Suggest alternate spellings of word. See Enchant documentation."""
        return self.checker.suggest(*args, **kwargs)
