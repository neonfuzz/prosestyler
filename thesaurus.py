

"""
A simple thesaurus using word embeddings fine-tuned on synonym/antonym tasks.

Variables:
    THESAURUS_FILE - location of the word embeddings
    VECS - the file, loaded

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
import nltk
from numpy.linalg import norm
import pandas as pd

from weak_words import WEAK_ADJS, WEAK_NOUNS, WEAK_VERBS


LANG = 'en'
THESAURUS_FILE = './counter-fitted-vectors.en.pkl.gz'
VECS = pd.read_pickle(THESAURUS_FILE)
WEAK_WORDS = WEAK_ADJS + WEAK_NOUNS + WEAK_VERBS
DICT = enchant.DictWithPWL(LANG, 'scientific_word_list.txt')


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
        dists.index, index=dists.index).apply(lambda w: _is_word_good(w))
    dists = dists[good_words]
    return dists.index.values
