

"""
Tools for parsing information at the sentence level.

Classes:
    Sentence - hold a lot of information about a sentence

Functions:
    gen_tokens - generate tokens from a string
    gen_sent - split a string into sentences

Variables:
    NLP - spaCy parser
    TOKENIZER - custom tokenizer for `NLP`
        that leaves contractions as a single word/token
    PREFIX_RE - prefix regex for `TOKENIZER`
    SUFFIX_RE - suffix regex for `TOKENIZER`
    INFIX_RE - infix regex for `TOKENIZER`
"""


# pylint: disable=no-name-in-module
import spacy
from spacy.lang.punctuation import TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES, \
                                   TOKENIZER_INFIXES
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex, \
                       compile_infix_regex


NLP = spacy.load('en_core_web_sm')

# Custom tokenizer that doesn't use the
# built-in exceptions list. This has
# the effect of keeping e.g. contractions
# together as one token.
PREFIX_RE = compile_prefix_regex(TOKENIZER_PREFIXES).search
SUFFIX_RE = compile_suffix_regex(TOKENIZER_SUFFIXES).search
INFIX_RE = compile_infix_regex(TOKENIZER_INFIXES).finditer

TOKENIZER = Tokenizer(
    NLP.vocab,
    prefix_search=PREFIX_RE,
    suffix_search=SUFFIX_RE,
    infix_finditer=INFIX_RE,
    )
NLP.tokenizer = TOKENIZER


def gen_tokens(string=None, doc=None):
    """
    Generate tokens from a string or spaCy doc.

    Arguments; must provide one of:
        string (str) - the string to parse
        doc (spacy.tokens.Span) - a parsed string

    Returns:
        tokens (list of strs) - all tokens including whitespace
    """
    if doc is None:
        doc = NLP(string)
    tokens = []
    for tok in doc:
        tokens.append(tok.text)
        if tok.whitespace_:
            tokens.append(tok.whitespace_)
    return tokens


def gen_sent(string):
    """Generate a list of sentences from a string."""
    doc = NLP(string)
    return list(doc.sents)


class Sentence():
    """
    A fancy text object for holding one sentence at a time.

    Instance variables:
        tokens (list) - word, punctuation, and whitespace tokens
        words (list) - words in the sentence
        inds (list) - indices of each word in `tokens`
        tags (list) - tuples containing each word and its POS tag
        lemmas (list) - like `tags`, but with lemmatized words
        nodes (spacy.tokens.Span) - spaCy dependency graph

    Methods:
        clean - remove unnecessary whitespace
    """

    def __init__(self, string):
        """
        Initialize Sentence.

        Arguments:
            string (str or spacy.tokens.Span) - text to parse
        """
        if isinstance(string, spacy.tokens.Span):
            self._string = string.text
            self._doc = string
        else:
            self._string = string
            self._doc = NLP(self._string)
        self.clean()

    def __repr__(self):
        """Provide string representation."""
        return self._string

    def __getitem__(self, idx):
        """Return when indexed."""
        return self.words[idx]

    @property
    def string(self):
        """Raw text of the sentence."""
        return self._string

    @string.setter
    def string(self, string):
        self._string = string
        self._doc = NLP(self._string)

    @property
    def tokens(self):
        """All tokens including words, punctuation, and whitespace."""
        return gen_tokens(doc=self._doc)

    @tokens.setter
    def tokens(self, tokens):
        self._string = ''.join(tokens)
        self._doc = NLP(self._string)

    @property
    def words(self):
        """Only the words of the sentence."""
        return [tok.text for tok in self._doc if not tok.is_punct]

    @property
    def inds(self):
        """Get indices of each word in `tokens`."""
        return [i for i, tok in enumerate(self.tokens) if tok in self.words]

    @property
    def tags(self):
        """Tuples of each word and its POS tag."""
        return [(tok.text, tok.tag_) for tok in self._doc if not tok.is_punct]

    @property
    def lemmas(self):
        """Tuples of each word lemma and its POS tag."""
        return [(tok.lemma_, tok.tag_) for tok in self._doc
                if not tok.is_punct]

    @property
    def nodes(self):
        """Get the spaCy dependency graph."""
        return self._doc

    def clean(self):
        """Remove unnecessary whitespace."""
        new_string = self._string.strip()
        for i in ',:;.?! ':
            new_string = new_string.replace(' %s' % i, i)
        if new_string != self._string:
            self._string = new_string
            self._doc = NLP(self._string)
        return self
