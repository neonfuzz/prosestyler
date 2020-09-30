

"""
Tools for parsing information at the sentence level.

Classes:
    Parser - wrapper around NLTK's CoreNLPDependencyParser
    Sentence - hold a lot of information about a sentence

Variables:
    PATH (str) - location of Stanford Core NLP model
    DEP_PARSER - instantiated `Parser` instance
"""


import os

import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPServer, \
    CoreNLPServerError

from helper_functions import penn2morphy, gen_tags, gen_words, gen_tokens


PATH = '/home/addie/opt/stanford-corenlp-4.1.0/'


class Parser():
    """Wrapper around NLTK's `CoreNLPDependencyParser.`"""
    def __init__(self, path=PATH):
        os.environ['CLASSPATH'] = os.environ.get('CLASSPATH', path)
        self.server = CoreNLPServer()
        self.parser = CoreNLPDependencyParser()

    def __del__(self):
        self.server.stop()

    def parse(self, *args, **kwargs):
        """Wrap NLTK's `CoreNLPDependencyParser.parse`"""
        try:
            self.server.start()
        except CoreNLPServerError:
            pass
        return self.parser.parse(*args, **kwargs)


DEP_PARSER = Parser()


class Sentence():
    """
    A fancy text object for holding one sentence at a time.

    Instance variables:
        tokens - list of word and punctuation tokens
        words - a list of words in the sentence
        inds - list of indices of each word in the token list
        tags - list of tuples contain word and its POS tag
        nodes - Stanford Dependency Parser nodes
                NOTE: only created if called.
        lemmas - like tags, but with lemmatized words.
                 NOTE: only created if called.

    Methods:
        clean - remove unnecessary whitespace
    """

    def __init__(self, string):
        """
        Arguments:
            string - the raw text string
        """
        self._lemmatizer = nltk.stem.WordNetLemmatizer()

        self._string = string
        self._tokens = gen_tokens(self._string)
        self._words, self._inds = gen_words(self._tokens)
        self._tags = gen_tags(self._words)
        self._lemmas = None
        self._nodes = None

        self.clean()

    def __repr__(self):
        return self._string

    def __getitem__(self, idx):
        return self.words[idx]

    def __getitem__(self, index):
        return self.words[index]

    @property
    def tokens(self):
        """
        Get/set the sentence tokens.

        Setting automatically re-calculates:
            string, words, inds, tags, nodes
        """
        return self._tokens

    @tokens.setter
    def tokens(self, tokens):
        self._string = ''.join(tokens)
        self._tokens = tokens
        self._words, self._inds = gen_words(self._tokens)
        self._tags = gen_tags(self._words)
        self._lemmas = None
        self._nodes = None

    @property
    def words(self):
        """Get the sentence words. words cannot be set."""
        return self._words

    @property
    def inds(self):
        """Get the sentence inds. inds cannot be set."""
        return self._inds

    @property
    def tags(self):
        """Get the sentence tags. tags cannot be set."""
        return self._tags

    @property
    def lemmas(self):
        """Get the sentence lemmas. lemmas cannot be set."""
        if self._lemmas is None:
            self._lemmas = []
            for word, tag in self._tags:
                new_tag = penn2morphy(tag)
                if new_tag is not None:
                    lemma = self._lemmatizer.lemmatize(word, new_tag)
                else:
                    lemma = word
                self._lemmas.append((lemma.lower(), tag))
        return self._lemmas

    @property
    def nodes(self):
        """Get the sentence nodes. nodes cannot be set."""
        if self._nodes is None:
            parse = list(DEP_PARSER.parse(self._words))[0]
            self._nodes = parse.nodes
        return self._nodes

    def clean(self):
        """Remove unneccesary whitespace."""
        new_string = self._string
        for i in ',:;.?! ':
            new_string = new_string.replace(' %s' % i, i)

        if new_string != self._string:
            self._string = new_string
            self._tokens = gen_tokens(self._string)
            self._words, self._inds = gen_words(self._tokens)

        return self
