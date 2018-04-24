

from bs4 import BeautifulSoup
from lxml import html
import requests

from weak_words import weak_adjs, weak_nouns, weak_verbs


class Thesaurus(object):
    """
    Hold information about a word and its synonyms.

    Instance variables:
        word - the word you've looked up
        synonyms - a dictionary with keys
                    'noun', 'verb', 'adj', and 'adv'
                   where each value is a list of synonyms for that
                   part of speech
        n_defs - the number of definitions for the word
    """
    def __init__(self, word):
        """Initialize the class with 'word'."""
        # Download thesaurus information from thesaurus.com
        self._word = word
        self._address = 'http://www.thesaurus.com/browse/%s?s=t' % self._word
        self._searchpage = requests.get(self._address)
        self._soup = BeautifulSoup(self._searchpage.content, 'html.parser')

        # Initialize synonyms lists.
        self._syns = {
            'noun': [],
            'verb': [],
            'adj': [],
            'adv': [],
            'as in': [],
        }
        # Find parts of speech for each definition.
        pos_tags = self._soup.select('div.mask a.pos-tab')
        self._pos = [[z.text for z in x.select('em')][0] for x in pos_tags]

        # Number of definitons
        self._n_defs = len(self._pos)
        if self._n_defs == 0:
            return

        # For each definition, append each synonym to the proper pos.
        max_syns = 70
        max_per_list = int(max_syns/self._n_defs)
        for defn in range(self._n_defs):
            data = self._soup.select('div#synonyms-%s li a' % defn)
            pos_word = self._pos[defn]
            defn_syns = []
            for d in data:
                word = d.find('span').string
                if word not in weak_verbs \
                   and word not in weak_adjs \
                   and word not in weak_nouns:
                    defn_syns += [word]
            self._syns[pos_word] += defn_syns[:max_per_list]

        # Remove duplicates and sort alphebetically.
        for k, v in self._syns.items():
            self._syns[k] = list(set(v))
            self._syns[k].sort()

    @property
    def word(self):
        return self._word

    @word.setter
    def word(self, word):
        if word == self._word:
            return
        self.__init__(word)

    @property
    def synonyms(self):
        return self._syns

    @property
    def n_defs(self):
        return self._n_defs
