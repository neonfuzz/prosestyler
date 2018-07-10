
import json
import re

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
    def __init__(self, word, max_syns=70):
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

        # Locate script which has json for all synonyms.
        scripts = self._soup.find_all('script')
        pattern = re.compile('window.INITIAL_STATE = {(.*?)};')
        for script in scripts:
            match = pattern.match(script.string)
            if match:
                json_string = '{%s}' % match.groups()[0]
                json_dict = json.loads(json_string)
                break

        # Find tabs for each definiton.
        try:
            self._defs = json_dict['searchData']['tunaApiData']['posTabs']
        except TypeError:
            self._defs = []

        # Number of definitions.
        self._n_defs = len(self._defs)
        if self._n_defs == 0:
            return

        # For each definition, append some synonyms to the proper pos.
        max_per_list = int(max_syns/self._n_defs)
        for defn in self._defs:
            defn_syns = [
                x['term'] for x in defn['synonyms']
                if x['term'] not in weak_verbs \
                and x['term'] not in weak_adjs \
                and x['term'] not in weak_nouns]
            self._syns[defn['pos']] += defn_syns[:max_per_list]

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
