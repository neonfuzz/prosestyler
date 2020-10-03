

r"""
A simple thesaurus using word embeddings fine-tuned on synonym/antonym tasks.

Variables:
    THESAURUS_FILE (str) - location of the word embeddings

Classes:
    Thesaurus - look up synonyms

Provided Courtesy of:
@InProceedings{mrksic:2016:naacl,
  author = {Nikola Mrk\v{s}i\'c and Diarmuid {\'O S\'eaghdha} and
            Blaise Thomson and Milica Ga\v{s}i\'c and Lina Rojas-Barahona and
            Pei-Hao Su and David Vandyke and Tsung-Hsien Wen and Steve Young},
  title = {Counter-fitting Word Vectors to Linguistic Constraints},
  booktitle = {Proceedings of HLT-NAACL},
  year = {2016},
}
"""

from numpy.linalg import norm
import pandas as pd

from .spellcheck import SpellCheck
from .. import resources
from ..checks.weak import WEAK_ADJS, WEAK_NOUNS, WEAK_VERBS


RESOURCE_PATH = resources.__path__[0]
THESAURUS_FILE = RESOURCE_PATH + '/counter-fitted-vectors.en.pkl.gz'
WEAK_WORDS = WEAK_ADJS + WEAK_NOUNS + WEAK_VERBS


class Thesaurus():
    """
    Look up synonyms.

    Instance attributes:
        vecs (pd.DataFrame) - word encoding vectors

    Methods:
        get_synonyms - look up synonyms for a word
    """

    def __init__(self, lang='en'):
        """
        Initialize Thesaurus.

        Optional Arguments:
            lang (str) - language; words not in this
                language will be pruned from results; default 'en'
        """
        self._spellcheck = SpellCheck(lang)
        self._vecs = None

    @property
    def vecs(self):
        """Get word vectors."""
        if self._vecs is None:
            self._vecs = pd.read_pickle(THESAURUS_FILE)
        return self._vecs

    def _is_word_good(self, word):
        """Filter non-dictionary and weak words."""
        if word in WEAK_WORDS:
            return False
        if self._spellcheck.check(word) is False:
            return False
        return True

    def get_synonyms(self, word, thresh=1.1):
        """
        Look up synonyms for a word.

        Arguments:
            word (str) - word to query

        Optional Arguments:
            thresh (float) - distance allowed for synonyms;
                higher values will bring more results, but of lower quality;
                lower values will bring fewer results, but of higher quality;
                default 1.1

        Returns:
            synonyms (np.array) - synonyms for `word`, sorted by relevance
        """
        word_vec = self.vecs.loc[word]
        dists = norm(self.vecs - word_vec, axis=1)
        dists = pd.Series(dists, index=self.vecs.index)
        dists = dists[dists < thresh]
        dists.sort_values(inplace=True)
        dists = dists[1:]  # Delete self-lookup
        # Remove weak words and non-dictionary words.
        good_words = pd.Series(
            dists.index, index=dists.index).apply(self._is_word_good)
        dists = dists[good_words]
        return dists.index.values
