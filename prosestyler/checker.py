

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
from string import punctuation

import argparse
import language_tool_python  # Grammar Check
import numpy as np
import proselint

from .checks.cliches import CLICHES
from .checks.fillers import FILLER_WORDS
from .checks.homophones import HOMOPHONE_LIST
from .checks.nominalizations import nominalize_check, filter_syn_verbs
from .checks.nouns import big_noun_phrases
from .checks.weak import WEAK_ADJS, WEAK_MODALS, WEAK_NOUNS, WEAK_VERBS
from . import resources
from .sentence import Sentence, Text, gen_sent, gen_tokens
from .tools import colors
from .tools.extended_argparse import BooleanOptionalAction
from .tools.gui import visual_edit
from .tools.helper_functions import fromx_to_id, now_checking_banner, \
    print_rows
from .tools.thesaurus import Thesaurus
from .tools.spellcheck import SpellCheck


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
PARSER.add_argument(
    '--frequent', action=BooleanOptionalAction, default=False,
    help='Show the most frequently used words')
PARSER.add_argument(
    '--vis-length', action=BooleanOptionalAction, default=False,
    help='Visualize sentence lengths')


class TextCheck(Text):
    """
    A fancy text object which can provide style suggestions.

    Sub-classed from `sentence.Text`.

    Instance variables:
        save_file - the file to be saved as the checks are performed
        sentences - a list of sententces within the text
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
        frequent_words - list the most-used words
        visualize_length - provide visual cues for sentence length
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
        # Define dictionaries etc.
        self._dict = SpellCheck(lang)
        self._gram = language_tool_python.LanguageTool(lang)
        self._thesaurus = Thesaurus(lang)

        super().__init__(string, save_file)

    def _suggest_toks(self, tokens, indices, suggestions, message,
                      can_replace_sent=False):
        """
        Ask the user to provide input on errors or style suggestions.

        Arguments:
            tokens (list) - tokens of the sentence in question
            indices (list) - indices of tokens to be replaced
            suggestions (list) - possible suggestions
            message (str) - error message, if any

        Optional arguments:
            can_replace_sent (bool) - should the user have the explicit option
                of replacing the entire sentence? default `False`
        """
        # Print the sentence with the desired token underlined.
        print()
        inds = range(indices[0], indices[-1]+1)
        colors.tokenprint(tokens, inds)
        phrase = ''.join([tokens[i] for i in inds])
        if message is not None:
            print()
            print('REASON:', message)
        print('Possible suggestions for "%s":' % phrase)

        # Print list of suggestions, as well as custom options.
        print_rows(suggestions)
        print(' (0) Leave be.')
        if can_replace_sent is True:
            print('(ss) Edit entire sentence.')
        print(' (?) Input your own.')

        # Get user input.
        # If a number, replace with suggestion.
        # If 0, return sentence as-is.
        # If 'ss', ask user to replace entire sentence.
        # Else: return user-input.
        user_input = input('Your choice: ')
        try:
            user_choice = int(user_input)
            if len(suggestions) >= user_choice > 0:
                ans = suggestions[user_choice-1]
                # Replace everything between the first and last tokens.
                tokens = tokens[:indices[0]] + [ans] + tokens[indices[-1]+1:]
            elif user_choice != 0:
                print('\n\n-------------\nINVALID VALUE\n-------------')
                tokens = self._suggest_toks(
                    tokens, indices, suggestions, can_replace_sent)
        except ValueError:
            if user_input == 'ss':
                sent = visual_edit(''.join(tokens))
                tokens = gen_tokens(sent)
            else:
                ans = user_input
                tokens = tokens[:indices[0]] + [ans] + tokens[indices[-1]+1:]
        return tokens

    def _synonyms(self, word):
        """Provide a list of synonyms for word."""
        synonyms = self._thesaurus.get_synonyms(word)
        return synonyms

    def _check_loop(self, error_method):
        for i, sent in enumerate(self._sentences):
            errors, suggests, ignore_list, messages = error_method(sent)
            tmp_sent = sent
            while errors:
                err = errors[0]
                new_tokens = self._suggest_toks(
                    # TODO: why is this 1 and not 0??
                    tmp_sent.tokens, err[1], suggests[0], messages[0], True)
                if new_tokens == tmp_sent.tokens:
                    ignore_list += [err]
                    errors = errors[1:]
                    suggests = suggests[1:]
                    messages = messages[1:]
                else:
                    tmp_sent = Sentence(''.join(new_tokens))
                    errors, suggests, ignore_list, messages = error_method(
                        tmp_sent, ignore_list)
            self._sentences[i] = tmp_sent
            self._clean()
            self.save()

    def _spelling_errors(self, sentence, ignore_list=None):
        errors = []
        if ignore_list is None:
            ignore_list = []
        nodes = sentence.nodes
        for tok in nodes:
            if tok.ent_iob != 2:
                # If token is part of a named entity, don't spellcheck.
                continue
            if tok.text == ' ' or tok.text == '\n' or tok.text in punctuation:
                continue
            tup = ([tok.text], [sentence.inds[tok.i-nodes[:].start]])
            if self._dict.check(tok.text) is False and tup not in ignore_list:
                errors += [tup]
        suggests = [self._dict.suggest(err[0][0]) for err in errors]
        messages = [None] * len(errors)
        return errors, suggests, ignore_list, messages

    def _grammar_errors(self, sentence, ignore_list=None):
        errors_gram = self._gram.check(sentence.string)
        # Don't check for smart quotes
        errors_gram = [
            err for err in errors_gram
            if err.ruleId != 'EN_QUOTES'  # No smartquotes.
            and not err.ruleId.startswith('MORFOLOGIK')  # No spellcheck.
            ]
        errors = []
        suggests = []
        messages = []
        if ignore_list is None:
            ignore_list = []
        for err in errors_gram:
            fromx = err.offset
            tox = fromx + err.errorLength
            ids = fromx_to_id(fromx, tox, sentence.tokens)
            toks = [sentence.tokens[i] for i in ids]
            errors += [(toks, ids)]
            # TODO: I think this would mess up suggestion/message
            #       order if errors wind up in the ignore list.
            errors = [e for e in errors if e not in ignore_list]
            suggests += [err.replacements]
            messages += [err.message]
        return errors, suggests, ignore_list, messages

    def _homophone_errors(self, sentence, ignore_list=None):
        errors = []
        suggests = []
        if ignore_list is None:
            ignore_list = []
        for i, tok in enumerate(sentence.tokens):
            for homophones in HOMOPHONE_LIST:
                for hom in homophones:
                    if hom == tok.lower() and ([tok], [i]) not in ignore_list:
                        other_homs = [h for h in homophones if h != hom]
                        errors += [([tok], [i])]
                        suggests += [homophones]
                        ignore_list += [([h], [i]) for h in other_homs]
        messages = [None] * len(errors)
        return errors, suggests, ignore_list, messages

    def _cliche_errors(self, sentence, ignore_list=None):
        errors = []
        suggests = []
        if ignore_list is None:
            ignore_list = []
        lem = ' '.join([x[0] if not x[1].startswith('PRP') else 'prp'
                        for x in sentence.lemmas
                        ]).lower()

        for k in CLICHES:
            if k in lem:
                fromx = lem.find(k)
                tox = fromx + len(k)
                ids = fromx_to_id(fromx, tox, gen_tokens(lem))
                toks = [sentence.tokens[i] for i in ids]
                if (toks, ids) not in ignore_list:
                    errors += [(toks, ids)]
                    suggests += [CLICHES[k]]
        messages = [None] * len(errors)
        return errors, suggests, ignore_list, messages

    def _passive_errors(self, sentence, ignore_list=None):
        errors = []
        suggests = []
        if ignore_list is None:
            ignore_list = []
        nodes = sentence.nodes
        vbns = [n for n in nodes if 'auxpass' in [c.dep_ for c in n.children]]
        for word in vbns:
            children = list(word.children)
            be_verbs = [c for c in children if c.dep_ == 'auxpass']
            try:
                ids = [sentence.inds[bv.i-nodes.start] for bv in be_verbs]
                ids += [sentence.inds[word.i-nodes.start]]
            except AttributeError:
                ids = [sentence.inds[bv.i] for bv in be_verbs]
                ids += [sentence.inds[word.i]]
            ids.sort()
            toks = [sentence.tokens[i] for i in ids]
            tup = (toks, ids)
            if tup not in ignore_list:
                errors += [tup]
                suggests += [[]]
        messages = [None] * len(errors)
        return errors, suggests, ignore_list, messages

    def _nominalization_errors(self, sentence, ignore_list=None):
        errors = []
        suggests = []
        if ignore_list is None:
            ignore_list = []
        nouns_lemmas = [
            (w[0], w[1], sentence.inds[i])
            for i, w in enumerate(sentence.lemmas) if w[1].startswith('NN')]
        for noun in nouns_lemmas:
            should_denom = nominalize_check(noun[0])
            if should_denom:
                syns = self._thesaurus.get_synonyms(noun[0])
                denoms = filter_syn_verbs(syns)
            else:
                denoms = []
            tup = ([noun[0]], [noun[2]])
            if denoms and tup not in ignore_list:
                errors += [tup]
                suggests += [denoms]
        messages = [None] * len(errors)
        return errors, suggests, ignore_list, messages

    def _weak_words_errors(self, sentence, ignore_list=None):
        errors = []
        suggests = []
        if ignore_list is None:
            ignore_list = []

        for node in sentence.nodes:
            text = node.text
            lemma = node.lemma_
            pos = node.tag_
            try:
                idx = sentence.inds[node.i-sentence.nodes.start]
            except AttributeError:
                idx = sentence.inds[node.i]
            tup = ([text], [idx])

            if tup not in ignore_list:
                if pos.startswith('V') \
                        and node.dep_ != 'aux' \
                        and lemma in WEAK_VERBS:
                    errors += [tup]
                    suggests += [self._synonyms(text)]
                elif lemma in WEAK_ADJS \
                        or lemma in WEAK_MODALS \
                        or lemma in WEAK_NOUNS:
                    errors += [tup]
                    suggests += [self._synonyms(text)]
        messages = [None] * len(errors)
        return errors, suggests, ignore_list, messages

    def _filler_errors(self, sentence, ignore_list=None):
        errors = []
        suggests = []
        if ignore_list is None:
            ignore_list = []
        for i, tok in enumerate(sentence.tokens):
            tup = ([tok], [i])
            if tok.lower() in FILLER_WORDS and tup not in ignore_list:
                errors += [tup]
                suggests += [['']]
        messages = [None] * len(errors)
        return errors, suggests, ignore_list, messages

    def _adverb_errors(self, sentence, ignore_list=None):
        errors = []
        suggests = []
        if ignore_list is None:
            ignore_list = []
        adv_modified = sorted([n for n in sentence.nodes
                               if 'advmod' in [c.dep_ for c in n.children]])
        for node in adv_modified:
            adv_node_ids = [c.i for c in node.children
                            if c.dep_ == 'advmod'
                            and c.text.endswith('ly')]
            ids = [sentence.inds[i] for i in adv_node_ids]
            ids += [sentence.inds[node.i]]
            ids.sort()
            toks = [sentence.tokens[i] for i in ids]
            tup = (toks, ids)
            if adv_node_ids \
                    and ids[1] - ids[0] <= 5 \
                    and node.tag_ is not None \
                    and tup not in ignore_list:
                errors += [tup]
                suggests += [self._synonyms(node.text)]
        messages = [None] * len(errors)
        return errors, suggests, ignore_list, messages

    def _noun_phrase_errors(self, sentence, ignore_list=None):
        """Detect clunky noun phrases."""
        errors = []
        suggests = []
        if ignore_list is None:
            ignore_list = []
        span_start = sentence.nodes[:].start
        for err in big_noun_phrases(sentence.nodes):
            toks = list(err)
            ids = sentence.inds[err.start-span_start:err.end-span_start]
            tup = (toks, ids)
            errors += [tup]
        suggests = [[]] * len(errors)
        messages = [None] * len(errors)
        return errors, suggests, ignore_list, messages

    def _proselint_errors(self, sentence, ignore_list=None):
        """Ask Proselint for advice."""
        errors = []
        suggests = []
        messages = []
        if ignore_list is None:
            ignore_list = []
        linted = proselint.tools.lint(sentence.string)
        for _, message, _, _, fromx, tox, _, _, replacements in linted:
            ids = fromx_to_id(fromx, tox, sentence.tokens)
            toks = [sentence.tokens[i] for i in ids]
            try:
                if toks[-1] == ' ':
                    ids = ids[:-1]
                    toks = toks[:-1]
            except IndexError:
                pass
            else:
                errors += [(toks, ids)]
                errors = [e for e in errors if e not in ignore_list]
                suggests += [replacements or []]
                messages += [message]
        return errors, suggests, ignore_list, messages

    def spelling(self):
        """Run a spell check on the text."""
        # pylint: disable=line-too-long
        # Courtesy of http://www.jpetrie.net/scientific-word-list-for-spell-checkersspelling-dictionaries/
        now_checking_banner('spelling')
        self._check_loop(self._spelling_errors)

    def grammar(self):
        """Run a grammar check on the text."""
        now_checking_banner('grammar')
        self._check_loop(self._grammar_errors)

    def homophone_check(self):
        """Point out every single homophone, for good measure."""
        now_checking_banner('homophones')
        self._check_loop(self._homophone_errors)

    def cliches(self):
        """Highlight cliches and offer suggestions."""
        now_checking_banner('clichés')
        self._check_loop(self._cliche_errors)

    def passive_voice(self):
        """Point out instances of passive voice."""
        now_checking_banner('passive voice')
        self._check_loop(self._passive_errors)

    def nominalizations(self):
        """Find many nominalizations and suggest stronger verbs."""
        now_checking_banner('nominalizations')
        self._check_loop(self._nominalization_errors)

    def weak_words(self):
        """Find weak words and suggest stronger ones."""
        now_checking_banner('weak words')
        self._check_loop(self._weak_words_errors)

    def filler_words(self):
        """Point out filler words and offer to delete them."""
        now_checking_banner('filler words')
        self._check_loop(self._filler_errors)

    def adverbs(self):
        """Find adverbs and verbs, offer better verbs."""
        now_checking_banner('adverbs')
        self._check_loop(self._adverb_errors)

    def noun_phrases(self):
        """Detect clunky noun phrases."""
        now_checking_banner('noun-phrases')
        self._check_loop(self._noun_phrase_errors)

    def proselint(self):
        """Ask Proselint for advice."""
        now_checking_banner('Proselint')
        self._check_loop(self._proselint_errors)

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

    def frequent_words(self, num=10):
        """
        Print a list of the most commonly used words.

        Ask user word-per-word if they'd like to view occurances
        in close proximity.
        """
        # TODO: We want the lemmas to include spaces somehow?
        #       lemmatized tokens?
        #       The end result should be printed out as proper string
        #       (currently no spaces nor punctuation)
        now_checking_banner('frequent words')
        lemmas = [t.lemma_ for s in self.sentences for t in s.nodes
                  if not t.is_punct and not t.is_stop]

        distr = zip(*np.unique(lemmas, return_counts=True))
        distr = [x for x in distr if x[1] > 1]
        distr.sort(key=lambda x: x[1], reverse=True)

        # Print 'num' most frequent words.
        for i, j in distr[:num]:
            print('%s: %.02f%%' % (i, j/len(lemmas)*100))
        print()

        # Ask if user wants to see words in proximity.
        print('-----')
        for word, freq in distr:
            occurs = np.array([
                i for i, lem in enumerate(lemmas) if lem == word])
            dists = occurs[1:] - occurs[:-1]
            # dist_thresh can be less than 30 if the word occurs a lot.
            dist_thresh = min(30, int(len(lemmas)/freq/3))
            to_print = occurs[np.where(dists < dist_thresh)]
            if to_print.shape[0]:
                print('-----')
                yes_no = self._ask_user(word, freq, len(to_print))
                print('-----')
                if yes_no == 'y':
                    for i in to_print:
                        start = max(0, i-int(dist_thresh/2))
                        stop = min(i+int(1.5*dist_thresh), len(lemmas))
                        tokens = lemmas[start:stop]
                        indices = np.where(tokens == word)[0]
                        colors.tokenprint(tokens, indices)
                        input('Enter to continue. ')
                        print('-----')

    def visualize_length(self, char='X'):
        """Produce a visualization of sentence length."""
        now_checking_banner('sentence length')
        for i, sent in enumerate(self._sentences):
            if sent == '\n\n':
                print()
                continue
            num = len([x for x in sent if x != ' ' and x not in punctuation])
            print('{: >6}'.format('(%s)' % (i+1)), char*num)


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
        args.frequent = 'frequent' in args.l
        args.vis_length = 'vis_length' in args.l
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
    if args.frequent or args.all:
        text.frequent_words()
    if args.vis_length or args.all:
        text.visualize_length()

    # Final result
    print('\n\n%s' % text.string)

    text.save()
