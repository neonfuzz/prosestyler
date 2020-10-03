

"""
The main script.

Classes:
    Text - fancy text object with loads of stlye checks

Variables:
    PARSER (argparse.ArgumentParser) - parse command line arguments

Functions:
    main - execute the main program
"""


from datetime import datetime
from string import punctuation

import argparse
import enchant  # Spell Check
import language_tool_python  # Grammar Check
import numpy as np

from .checks.cliches import CLICHES
from .checks.fillers import FILLER_WORDS
from .checks.homophones import HOMOPHONE_LIST
from .checks.nominalizations import denominalize
from .checks.weak import WEAK_ADJS, WEAK_MODALS, WEAK_NOUNS, WEAK_VERBS
from .sentence import Sentence, gen_sent, gen_tokens
from .tools import colors
from .tools.gui import visual_edit
from .tools.helper_functions import fromx_to_id, now_checking_banner, print_rows
from .tools.thesaurus import get_synonyms, RESOURCE_PATH


PARSER = argparse.ArgumentParser(description='Perform a deep grammar check.')
PARSER.add_argument('file', help='The file to be analyzed.')
PARSER.add_argument('-o', type=str, metavar='outfile',
                    help='Name of output file ' \
                         '(default: <filename>_out_<datetime>)')
PARSER.add_argument(
    '-d', default='en_US', type=str, metavar='dictionary',
    help='Which dictionary to use (default: en_US)')


class Text():
    """
    A fancy text object which can provide style suggestions.

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
    frequent_words - list the most-used words
    visualize_length - provide visual cues for sentence length
    polish - run all checks in order
    quick_check - run some of the checks
    """

    def __repr__(self):
        """Represent self as string."""
        return self._string

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
        self._dict = enchant.DictWithPWL(
            lang, RESOURCE_PATH + 'scientific_word_list.txt')
        self._gram = language_tool_python.LanguageTool(lang)
        self._thesaurus_thresh = 1.1

        # Make all the things.
        self._string = string.replace('“', '"').replace('”', '"')
        self._string = string.replace('‘', "'").replace('’', "'")
        self._sentences = [Sentence(x) for x in gen_sent(self._string)]
        self._tokens = None
        self._words = None
        self._tags = None
        self._clean()  # Also makes tokens, words, tags.

        # Save for the very first time.
        if save_file is None:
            save_file = ''.join(self._words[:3]) + \
                        ' ' + str(datetime.now()) + '.txt'
        self.save_file = save_file
        self.save()

    def __getitem__(self, idx):
        """Return sentence when indexed."""
        return self.sentences[idx]

    def save(self):
        """Save the object to file."""
        with open(self.save_file, 'w') as myfile:
            myfile.write(self._string)

    def _suggest_toks(self, tokens, indices, suggestions,
                      can_replace_sent=False):
        """
        Ask the user to provide input on errors or style suggestions.

        Arguments:
            tokens (list) - tokens of the sentence in question
            indices (list) - indices of tokens to be replaced
            suggestions (list) - possible suggestions

        Optional arguments:
            can_replace_sent (bool) - should the user have the explicit option
                of replacing the entire sentence? default `False`
        """
        # Print the sentence with the desired token underlined.
        print()
        inds = range(indices[0], indices[-1]+1)
        colors.tokenprint(tokens, inds)
        phrase = ''.join([tokens[i] for i in indices])
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

    def _thesaurus(self, word, pos):
        """Provide a list of synonyms for word."""
        # TODO: Look up with word instead of lemma in all cases.
        synonyms = get_synonyms(word, self._thesaurus_thresh)
        return synonyms

    def _check_loop(self, error_method):
        for i, sent in enumerate(self._sentences):
            errors, suggests, ignore_list = error_method(sent)
            tmp_sent = sent
            while errors:
                err = errors[0]
                new_tokens = self._suggest_toks(
                    tmp_sent.tokens, err[1], suggests[0], True)
                if new_tokens == tmp_sent.tokens:
                    ignore_list += [err]
                    errors = errors[1:]
                    suggests = suggests[1:]
                else:
                    tmp_sent = Sentence(''.join(new_tokens))
                    errors, suggests, ignore_list = error_method(
                        tmp_sent, ignore_list)
            self._sentences[i] = tmp_sent
            self._clean()
            self.save()

    def _spelling_errors(self, sentence, ignore_list=None):
        errors = []
        if ignore_list is None:
            ignore_list = []
        for j, tok in enumerate(sentence.tokens):
            if tok == ' ' or tok == '\n' or tok in punctuation:
                continue
            tup = ([tok], [j])
            if self._dict.check(tok) is False and tup not in ignore_list:
                errors += [tup]
        suggests = [self._dict.suggest(err[0][0]) for err in errors]
        return errors, suggests, ignore_list

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
        if ignore_list is None:
            ignore_list = []
        for err in errors_gram:
            fromx = err.offset
            tox = fromx + err.errorLength
            ids = fromx_to_id(fromx, tox, sentence.tokens)
            toks = [sentence.tokens[i] for i in ids]
            errors += [(toks, ids)]
            errors = [e for e in errors if e not in ignore_list]
            suggests += [err.replacements]
        return errors, suggests, ignore_list

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
        return errors, suggests, ignore_list

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
        return errors, suggests, ignore_list

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
            ids = [sentence.inds[bv.i] for bv in be_verbs]
            ids += [sentence.inds[word.i]]
            ids.sort()
            toks = [sentence.tokens[i] for i in ids]
            tup = (toks, ids)
            if tup not in ignore_list:
                errors += [tup]
                suggests += [[]]
        return errors, suggests, ignore_list

    def _nominalization_errors(self, sentence, ignore_list=None):
        errors = []
        suggests = []
        if ignore_list is None:
            ignore_list = []
        nouns_lemmas = [
            (w[0], w[1], sentence.inds[i])
            for i, w in enumerate(sentence.lemmas) if w[1].startswith('NN')]
        for noun in nouns_lemmas:
            denoms = denominalize(noun[0])
            tup = ([noun[0]], [noun[2]])
            if denoms and tup not in ignore_list:
                errors += [tup]
                suggests += [denoms]
        return errors, suggests, ignore_list

    def _weak_words_errors(self, sentence, ignore_list=None):
        errors = []
        suggests = []
        if ignore_list is None:
            ignore_list = []

        def check_verbs(errors, suggests):
            """
            Check for weak verbs.

            Ignores "helper verbs" e.g. "have" in the perfect tense.

            Arguments:
                errors - current list of errors
                suggests - current list of suggestions

            Returns:
                errors - updated list of errors
                suggests - updated list of suggestions
            """
            nodes = sentence.nodes
            verbs_lemmas = [
                (w[0], w[1], sentence.inds[i])
                for i, w in enumerate(sentence.lemmas)
                if w[1].startswith('V') and nodes[i].dep_ != 'aux']
            for lemma, pos, index in verbs_lemmas:
                tup = ([lemma], [index])
                if lemma in WEAK_VERBS and tup not in ignore_list:
                    errors += [tup]
                    suggests += [self._thesaurus(lemma, pos)]
            return errors, suggests

        for i, lempair in enumerate(sentence.lemmas):
            if lempair[1].startswith('V'):
                continue
            tup = ([lempair[0]], [sentence.inds[i]])
            if (lempair[0] in WEAK_ADJS
                    or lempair[0] in WEAK_MODALS
                    or lempair[0] in WEAK_NOUNS
               ) and tup not in ignore_list:
                errors += [tup]
                suggests += [self._thesaurus(lempair[0], lempair[1])]
        errors, suggests = check_verbs(errors, suggests)

        return errors, suggests, ignore_list

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
        return errors, suggests, ignore_list

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
                suggests += [self._thesaurus(node.text, node.tag_)]
        return errors, suggests, ignore_list

    def spelling(self):
        """Run a spell check on the text!"""
        # pylint: disable=line-too-long
        # Courtesy of http://www.jpetrie.net/scientific-word-list-for-spell-checkersspelling-dictionaries/
        self._check_loop(self._spelling_errors)

    def grammar(self):
        """Run a grammar check on the text!"""
        self._check_loop(self._grammar_errors)

    def homophone_check(self):
        """Point out every single homophone, for good measure."""
        self._check_loop(self._homophone_errors)

    def cliches(self):
        """Highlight cliches and offer suggestions."""
        self._check_loop(self._cliche_errors)

    def passive_voice(self):
        """Point out instances of passive voice."""
        self._check_loop(self._passive_errors)

    def nominalizations(self):
        """Find many nominalizations and suggest stronger verbs."""
        self._check_loop(self._nominalization_errors)

    def weak_words(self):
        """Find weak words and suggest stronger ones."""
        self._check_loop(self._weak_words_errors)

    def filler_words(self):
        """Point out filler words and offer to delete them."""
        self._check_loop(self._filler_errors)

    def adverbs(self):
        """Find adverbs and verbs, offer better verbs."""
        self._check_loop(self._adverb_errors)

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
        for i, sent in enumerate(self._sentences):
            if sent == '\n\n':
                print()
                continue
            num = len([x for x in sent if x != ' ' and x not in punctuation])
            print('{: >6}'.format('(%s)' % (i+1)), char*num)

    def polish(self):
        """Run many of the default checks in order."""
        now_checking_banner('spelling')
        self.spelling()

        now_checking_banner('grammar')
        self.grammar()

        now_checking_banner('homophones')
        self.homophone_check()

        now_checking_banner('clichés')
        self.cliches()

        now_checking_banner('passive voice')
        self.passive_voice()

        now_checking_banner('nominalizations')
        self.nominalizations()

        now_checking_banner('weak words')
        self.weak_words()

        now_checking_banner('filler words')
        self.filler_words()

        now_checking_banner('adverbs')
        self.adverbs()

        now_checking_banner('frequent words')
        self.frequent_words()

        now_checking_banner('variation in sentence length')
        self.visualize_length()

    def quick_check(self):
        """Run some quick checks in order."""
        now_checking_banner('spelling')
        self.spelling()

        now_checking_banner('grammar')
        self.grammar()

        now_checking_banner('clichés')
        self.cliches()

        now_checking_banner('passive voice')
        self.passive_voice()

        now_checking_banner('nominalizations')
        self.nominalizations()

        now_checking_banner('filler words')
        self.filler_words()

        now_checking_banner('adverbs')
        self.adverbs()

    def _clean(self):
        """Remove unneccesary whitespace."""
        sents = [s.clean() for s in self._sentences]

        self._string = ' '.join([str(s) for s in sents])
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
        self._clean()

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


def main():
    """Run the program with given arguments."""
    # Import text
    global TEXT  # NOTE: for debugging
    args = PARSER.parse_args()
    if args.o is None:
        args.o = '%s_out_%s.txt' % (args.file, datetime.now())
    with open(args.file) as myfile:
        TEXT = Text(''.join(myfile.readlines()), save_file=args.o, lang=args.d)

    # Check that stuff
    TEXT.quick_check()

    # Final result
    print('\n\n%s' % TEXT.string)

    TEXT.save()

if __name__ == '__main__':
    TEXT = None  # NOTE: for debugging
    main()