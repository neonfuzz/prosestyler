

"""
Tools for parsing information at the sentence level.

Classes:
    TokenizeAndParse - wrapper around spaCy
    Sentence - hold a lot of information about a sentence
    Text - hold an entire text (multiple sentences)

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
from datetime import datetime
import re
from string import punctuation

import spacy
from spacy.lang.punctuation import TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES, \
                                   TOKENIZER_INFIXES
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex, \
                       compile_infix_regex, filter_spans


def _custom_sent_boundaries(doc):
    """
    Create custom sentence tokenizing boundaries for spaCy.

    By default, sentences could be splitting with e.g. a comma at the
    beginning of a sentence. This function forces all punctuation to
    not be allowed to start a sentence.

    Additionally, 'et. al.', 'i.e.', and 'e.g.' will not split a sentence.

    Arguments:
        doc (spacy.tokens.Span) - the spacy doc to process

    Returns:
        doc (spacy.tokens.Span) - processed doc
    """
    for token in doc[1:-1]:
        if token.text in punctuation:
            doc[token.i].is_sent_start = False
        if token.text == 'al' \
                and doc[token.i+1].text == '.' \
                and doc[token.i-1].text == '.' \
                and doc[token.i-2].text == 'et':
            doc[token.i].is_sent_start = False
        if token.text == 'i.e' \
                and doc[token.i+1].text == '.':
            doc[token.i+2].is_sent_start = False
    return doc


def _custom_token_boundaries(doc):
    """
    Create custom token boundaries that keep abbreviations together.

    Merges 'e.g.', 'i.e.', 'et.', and 'al.' into a single token each.

    Arguments:
        doc (spacy.tokens.Span) - the spaCy doc to process

    Returns:
        doc (spacy.tokens.Span) - processed doc

    """
    merge_list = []
    for token in doc:
        if token.text in ['et', 'al', 'i.e', 'e.g'] \
                and doc[token.i+1].text == '.':
            merge_list.append(doc[token.i:token.i+2])
    merge_list = filter_spans(merge_list)
    with doc.retokenize() as retokenizer:
        for span in merge_list:
            retokenizer.merge(span)
    return doc


class TokenizeAndParse():
    """
    Lazy-loads the spaCy model and treats contractions as one token.

    Instance Attributes:
        nlp (spacy.lang model) - the raw spaCy model

    When Called:
        pass arguments to `nlp`
    """

    def __init__(self, loadfile='en_core_web_sm'):
        """
        Initialize TokenizeAndParse.

        Arguments:
            loadfile (str) - spaCy model to load
        """
        self._loadfile = loadfile
        self._nlp = None

    def __call__(self, *args, **kwargs):
        """Wrap spaCy model."""
        return self.nlp.__call__(*args, **kwargs)

    def _gen_nlp(self):
        """Lazy load the spaCy model, with custom tokenizer."""
        self._nlp = spacy.load(self._loadfile)

        # Custom tokenizer that doesn't use the
        # built-in exceptions list. This has
        # the effect of keeping e.g. contractions
        # together as one token.
        suffixes = TOKENIZER_SUFFIXES
        for suf in ["'s", "'S", '’s', '’S']:
            # Possessives are handled in suffixes.
            # We will not split on these.
            suffixes.remove(suf)
        prefix_re = compile_prefix_regex(TOKENIZER_PREFIXES).search
        suffix_re = compile_suffix_regex(suffixes).search
        infix_re = compile_infix_regex(TOKENIZER_INFIXES).finditer
        tokenizer = Tokenizer(
            self._nlp.vocab,
            prefix_search=prefix_re,
            suffix_search=suffix_re,
            infix_finditer=infix_re,
            )
        self._nlp.tokenizer = tokenizer

        # Custom sentence and token boundaries.
        self._nlp.add_pipe(_custom_sent_boundaries, before='parser')
        self._nlp.add_pipe(_custom_token_boundaries, before='parser')

    @property
    def nlp(self):
        """Get the spaCy model. Load if not already."""
        if self._nlp is None:
            self._gen_nlp()
        return self._nlp


NLP = TokenizeAndParse()


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

    def __setitem__(self, idx, value):
        """Set a word in place."""
        self.words[idx] = value

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
        return [tok.text for tok in self._doc]

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
        return [(tok.lemma_, tok.tag_) for tok in self._doc]

    @property
    def nodes(self):
        """Get the spaCy dependency graph."""
        return self._doc

    def clean(self):
        """Remove unnecessary whitespace."""
        new_string = self._string.strip(' ')
        for i in ',:;.?! ':
            new_string = new_string.replace(' %s' % i, i)
        if new_string != self._string:
            self._string = new_string
            self._doc = NLP(self._string)
        return self


class Text():
    """
    A fancy text object for holding multiple sentences.

    Instance variables:
        save_file - the file to be saved as the checks are performed
        sentences - a list of sententces within the text
        string - a string of the entire text
        tags - a list of words and their parts of speech tags
        tokens - a list of tokens
        words - a list of words

    Methods:
        save - save the text to a file
    """

    def __repr__(self):
        """Represent self as string."""
        return self._string

    def __init__(self, string, save_file=None):
        """
        Initialize `Text`.

        Arguments:
            string (str) - the text string to be parsed

        Optional arguments:
            save_file (str) - the output file to be used between each step
        """
        self._string = string.replace('“', '"').replace('”', '"')
        self._string = self._string.replace('‘', "'").replace('’', "'")
        self._sentences = [Sentence(x) for x in gen_sent(self._string)]
        self._tokens = None
        self._words = None
        self._tags = None
        self.clean()  # Also makes tokens, words, tags.

        # Save for the very first time.
        if save_file is None:
            save_file = ''.join(self._words[:3]) + \
                        ' ' + str(datetime.now()) + '.txt'
        self.save_file = save_file
        self.save()

    def __getitem__(self, idx):
        """Return sentence when indexed."""
        return self.sentences[idx]

    def __setitem__(self, idx, value):
        """Set a sentence in place."""
        self.sentences[idx] = value

    def __len__(self):
        """Return number of sentences."""
        return len(self.sentences)

    def save(self):
        """Save the object to file."""
        with open(self.save_file, 'w') as myfile:
            myfile.write(self._string)

    def clean(self):
        """Remove unneccesary whitespace."""
        sents = [s.clean() for s in self._sentences]

        self._string = ' '.join([str(s) for s in sents])
        self._string = re.sub(r'\n+\s+', r'\n\n', self._string)
        self._sentences = sents
        self._tokens = [t for s in self._sentences for t in s.tokens]
        self._words = [w for s in self._sentences for w in s.words]
        self._tags = [t for s in self._sentences for t in s.tags]

    @property
    def string(self):
        """
        Get/set the text string.

        Setting will automatically set sentences/tokens/etc.
        """
        return self._string

    @string.setter
    def string(self, string):
        self._string = string
        self._string = self._string.replace('“', '"').replace('”', '"')
        self._string = self._string.replace('‘', "'").replace('’', "'")
        self._sentences = gen_sent(self._string)
        self.clean()

    @property
    def sentences(self):
        """Get the sentences. sentences cannot be set."""
        return self._sentences

    @property
    def tokens(self):
        """Get the tokens. tokens cannot be set."""
        return [s.tokens for s in self._sentences]

    @property
    def words(self):
        """Get the words. words cannot be set."""
        return [s.words for s in self._sentences]

    @property
    def tags(self):
        """Get the tags. tags cannot be set."""
        return [s.tags for s in self._sentences]
