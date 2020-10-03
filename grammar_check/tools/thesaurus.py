

r"""
A simple thesaurus using word embeddings fine-tuned on synonym/antonym tasks.

Variables:
    RESOURCE_PATH (str) - path to resource directory
    THESAURUS_FILE (str) - location of the word embeddings
    VECS (pd.DataFrame) - the file, loaded

Functions:
    get_synonyms - query the thesaurus

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

import enchant
from numpy.linalg import norm
import pandas as pd

from .. import resources
from ..checks.weak import WEAK_ADJS, WEAK_NOUNS, WEAK_VERBS


RESOURCE_PATH = resources.__path__[0]
LANG = 'en'
THESAURUS_FILE = './counter-fitted-vectors.en.pkl.gz'
VECS = pd.read_pickle(THESAURUS_FILE)
WEAK_WORDS = WEAK_ADJS + WEAK_NOUNS + WEAK_VERBS
DICT = enchant.DictWithPWL(LANG, RESOURCE_PATH + '/scientific_word_list.txt')


def _is_word_good(word):
    if word in WEAK_WORDS:
        return False
    if DICT.check(word) is False:
        return False
    return True


def get_synonyms(word, thresh=1.):
    """
    Query the thesaurus for the given word.

    Arguments:
        word (str) - word to look up

    Optional Arguments:
        thresh (float) - distance cutoff for synonyms
            default 1.0,
            maximum is often ~1.5,
            0.005 quantile is often ~1.2
            0.001 quantile is often ~0.9

    Returns:
        synonyms (list of strs) - synonyms, listed in order of closeness
    """
    word_vec = VECS.loc[word]
    dists = norm(VECS - word_vec, axis=1)
    dists = pd.Series(dists, index=VECS.index)
    dists = dists[dists < thresh]
    dists.sort_values(inplace=True)
    dists = dists[1:]  # Delete self-lookup
    # Remove weak words and non-dictionary words.
    good_words = pd.Series(
        dists.index, index=dists.index).apply(_is_word_good)
    dists = dists[good_words]
    return dists.index.values
