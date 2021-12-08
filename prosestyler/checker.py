

"""
The main script.

Classes:
    TextCheck - fancy text object with loads of stlye checks

Variables:
    PARSER (argparse.ArgumentParser) - parse command line arguments

Functions:
    check - execute the main program
"""


from datetime import datetime

import argparse
from argparse import BooleanOptionalAction

from . import resources
from .checks.adverbs import Adverbs
from .checks.cliches import Cliches
from .checks.fillers import Filler
from .checks.grammar import Grammar
from .checks.homophones import Homophones
from .checks.nominalizations import Nominalizations
from .checks.nouns import Nouns
from .checks.passive import Passive
from .checks.proselint import ProseLinter
from .checks.spelling import Speller
from .checks.weak import Weak
from .sentence import Text
from .tools.thesaurus import Thesaurus


RESOURCE_PATH = resources.__path__[0]


PARSER = argparse.ArgumentParser(
    description='Perform a deep grammar and style check.')
PARSER.add_argument('file', help='The file to be analyzed.')
PARSER.add_argument('-o', type=str, metavar='outfile',
                    help='Name of output file ' \
                         '(default: <filename>_out_<datetime>)')
PARSER.add_argument(
    '-d', default='en_US', type=str, metavar='dictionary',
    help='Which dictionary to use (default: en_US)')
PARSER.add_argument(
    '-l', type=str, nargs='+', metavar='check_name',
    help='List of checks to use (overrides all other options, except --all).'
    )
PARSER.add_argument(
    '--all', action='store_true',
    help='Use ALL checks (overrides all other options, including -l).')
PARSER.add_argument(
    '--spelling', action=BooleanOptionalAction, default=True,
    help='Run a spellcheck')
PARSER.add_argument(
    '--grammar', action=BooleanOptionalAction, default=True,
    help='Run a grammar check')
PARSER.add_argument(
    '--cliches', action=BooleanOptionalAction, default=True,
    help='Check for cliches')
PARSER.add_argument(
    '--passive', action=BooleanOptionalAction, default=True,
    help='Check for passive voice')
PARSER.add_argument(
    '--nominalizations', action=BooleanOptionalAction, default=True,
    help='Check for nominalizations')
PARSER.add_argument(
    '--filler', action=BooleanOptionalAction, default=True,
    help='Check for filler words')
PARSER.add_argument(
    '--adverbs', action=BooleanOptionalAction, default=True,
    help='Check for adverbs')
PARSER.add_argument(
    '--noun_phrases', action=BooleanOptionalAction, default=True,
    help='Check for adverbs')
PARSER.add_argument(
    '--homophones', action=BooleanOptionalAction, default=False,
    help='Show every detected homophone')
PARSER.add_argument(
    '--weak', action=BooleanOptionalAction, default=False,
    help='Check for weak words')
PARSER.add_argument(
    '--lint', action=BooleanOptionalAction, default=False,
    help='Run Proselint on the text')


class TextCheck(Text):
    """
    A fancy text object which can provide style suggestions.

    Sub-classed from `sentence.Text`.

    Instance variables:
        save_file - the file to be saved as the checks are performed
        sentences - a list of sentences within the text
        string - a string of the entire text
        tags - a list of words and their parts of speech tags
        tokens - a list of tokens
        words - a list of words

    Methods:
        save - save the text to a file
        spelling - check spelling
        grammar - check grammar
        homophone_check - highlight homophones
        cliches - point out overused phrases
        passive_voice - check for passive voice
        nominalizations - point out nominalizations
        weak_words - highlight weak words
        filler_words - point out words that may be unneccesary
        adverbs - highlight adverbs
        noun_phrases - show clunky noun phrases
        proselint - ask Proselint for advice
    """

    def __init__(self, string, save_file=None, lang='en_US'):
        """
        Initialize `Text`.

        Arguments:
            string (str) - the text string to be parsed

        Optional arguments:
            save_file (str) - the output file to be used between each step
            lang (str) - the language to be used
                (not fully implemented, default en_US)
        """
        # Define checks.
        self._adverbs = Adverbs()
        self._cliches = Cliches()
        self._filler = Filler()
        self._grammar = Grammar(lang)
        self._homophones = Homophones()
        self._nominalizations = Nominalizations()
        self._nouns = Nouns()
        self._passive = Passive()
        self._proselint = ProseLinter()
        self._speller = Speller(lang)
        self._weak = Weak()

        # Define dictionaries etc.
        self._thesaurus = Thesaurus()

        super().__init__(string, save_file)

    def spelling(self):
        """Run a spell check on the text."""
        # pylint: disable=line-too-long
        # Courtesy of http://www.jpetrie.net/scientific-word-list-for-spell-checkersspelling-dictionaries/
        self._speller(self)

    def grammar(self):
        """Run a grammar check on the text."""
        self._grammar(self)

    def homophone_check(self):
        """Point out every single homophone, for good measure."""
        self._homophones(self)

    def cliches(self):
        """Highlight cliches and offer suggestions."""
        self._cliches(self)

    def passive_voice(self):
        """Point out instances of passive voice."""
        self._passive(self)

    def nominalizations(self):
        """Find many nominalizations and suggest stronger verbs."""
        self._nominalizations(self)

    def weak_words(self):
        """Find weak words and suggest stronger ones."""
        self._weak(self)

    def filler_words(self):
        """Point out filler words and offer to delete them."""
        self._filler(self)

    def adverbs(self):
        """Find adverbs and verbs, offer better verbs."""
        self._adverbs(self)

    def noun_phrases(self):
        """Detect clunky noun phrases."""
        self._nouns(self)

    def proselint(self):
        """Ask Proselint for advice."""
        self._proselint(self)

    def _ask_user(self, word, freq, close):
        """Ask user if they want to view words in close proximity."""
        nwords = len([w for sentence in self.words for w in sentence])
        print("'%s' appeard %s times (%.02f%%)." % (
            word, freq, freq/nwords*100))
        ans = input(
            'Would you like to view occurances in proximity? (%s) ' % close)
        while not ans:
            ans = input('Sorry, try again: ')
        return ans[0].lower()


def _reset_args_with_list(args):
    if args.l is not None:
        args.spelling = 'spelling' in args.l
        args.grammar = 'grammar' in args.l
        args.homophones = 'homophones' in args.l
        args.cliches = 'cliches' in args.l
        args.passive = 'passive' in args.l
        args.nominalizations = 'nominalizations' in args.l
        args.weak = 'weak' in args.l
        args.filler = 'filler' in args.l
        args.adverbs = 'adverbs' in args.l
        args.noun_phrases = 'noun_phrases' in args.l
        args.lint = 'lint' in args.l
    return args


def check():
    """Run the program with given arguments."""
    # Import text
    args = PARSER.parse_args()
    args = _reset_args_with_list(args)
    if args.o is None:
        args.o = '%s_out_%s.txt' % (args.file, datetime.now())
    with open(args.file) as myfile:
        text = TextCheck(
            ''.join(myfile.readlines()), save_file=args.o, lang=args.d)

    # Check everything.
    if args.spelling or args.all:
        text.spelling()
    if args.grammar or args.all:
        text.grammar()
    if args.homophones or args.all:
        text.homophone_check()
    if args.cliches or args.all:
        text.cliches()
    if args.passive or args.all:
        text.passive_voice()
    if args.nominalizations or args.all:
        text.nominalizations()
    if args.weak or args.all:
        text.weak_words()
    if args.filler or args.all:
        text.filler_words()
    if args.adverbs or args.all:
        text.adverbs()
    if args.noun_phrases or args.all:
        text.noun_phrases()
    if args.lint or args.all:
        text.proselint()

    # Final result
    print('\n\n%s' % text.string)

    text.save()
