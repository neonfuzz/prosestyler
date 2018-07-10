#!/bin/python3


from datetime import datetime
from math import ceil
import os
from string import punctuation

import argparse
import enchant  # Spell Check
import language_check  # Grammar Check
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.parse.stanford import StanfordDependencyParser
import numpy as np
from pattern.en import conjugate
from pattern.en import pluralize
from scipy import optimize

from cliches import cliches
import colors
from filler_words import filler_words
from gui import visual_edit
from homophone_list import homophone_list
from nominalizations import denominalize
from thesaurus import Thesaurus
from weak_words import weak_adjs, weak_modals, weak_nouns, weak_verbs


PARSER = argparse.ArgumentParser(description='Perform a deep grammar check.')
PARSER.add_argument('file', help='The file to be analyzed.')
PARSER.add_argument('-o', type=str, metavar='outfile',
                    help='Name of output file ' \
                         '(default: <filename>_out_<datetime>)')
PARSER.add_argument(
    '-d', default='en_US', type=str, metavar='dictionary',
    help='Which dictionary to use (default: en_US)')
PARSER.add_argument(
    '-t', action='store_true',
    help='Train the sentence tokenizer on the text instead of using '
         'the default training set; this takes longer but is useful '
         'for e.g. scientic papers which have a lot of atypical '
         'punctuation. (default: False)')


os.environ['STANFORD_PARSER'] = '/home/addie/opt/stanford/stanford-parser-' \
                                'full-2018-02-27/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/home/addie/opt/stanford/stanford-parser-' \
                                'full-2018-02-27/stanford-parser-3.9.1-' \
                                'models.jar'

dep_parser = StanfordDependencyParser(
    model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')


def now_checking_banner(word):
    """Print a pretty banner on the output."""
    mystr = '---  Now Checking %s!  ---' % word.title()
    dashstr = '-' * len(mystr)
    print('\n\n')
    print(dashstr)
    print(mystr)
    print(dashstr)


def print_rows(lis, max_rows=21, cols=3, item_width=18):
    """
    Given a list of items, print them, numbered, in columns and rows.
    """
    if len(lis) == 1 and lis[0] == '':
        print(' (1) <delete>')
        return

    max_items = max_rows * cols
    if len(lis) > max_items:
        lis = lis[:max_items]
    if len(lis) < 2*cols:
        cols = 1

    # Make a string template holding each column.
    mystr = '{: >4} {: <%s}' % (item_width) * cols
    nrows = ceil(len(lis)/cols)
    rows = [[]] * nrows
    r = 0
    # Order stuff to read down each column.
    # (rather than across each row).
    for i, j in enumerate(lis):
        rows[r] = rows[r] + ['(%s)' % (i+1), j]
        r = (r+1) % nrows
    while r != 0:
        rows[r] = rows[r] + ['', '']
        r = (r+1) % nrows
    for row in rows:
        print(mystr.format(*row))


class Text(object):
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

    def __init__(self, string, save_file=None, lang='en_US',
                 train_sents=False):
        """
        Arguments:
        string - the text string to be parsed

        Optional arguments:
        save_file - the output file to be used between each step
        lang - the language to be used
            (not fully implemented, default en_US)
        train_sents - train the sentence tokenizer on the text
            instead of using the default training set; takes longer,
            but useful for e.g. scientific papers.
            (default False)
        """
        # Define dictionaries etc.
        if train_sents is True:
            self._tokenizer = nltk.tokenize.PunktSentenceTokenizer(
                string).sentences_from_text
        else:
            self._tokenizer = nltk.data.load(
                'tokenizers/punkt/english.pickle').tokenize
        self._dict = enchant.DictWithPWL(lang, 'scientific_word_list.txt')
        self._gram = language_check.LanguageTool(lang)
        self._gram.disable_spellchecking()
        self._lemmatizer = nltk.stem.WordNetLemmatizer()

        # Make all the things.
        self._string = string.replace('“', '"').replace('”', '"')
        self._string = string.replace('‘', "'").replace('’', "'")
        self._sentences = self._gen_sent(self._string)
        self._clean()  # Also makes tokens, words, tags.

        # Save for the very first time.
        if save_file is None:
            save_file = ''.join(self._words[:3]) + \
                        ' ' + str(datetime.now()) + '.txt'
        self.save_file = save_file
        self.save()

    def save(self):
        """Save the object to file."""
        with open(self.save_file, 'w') as myfile:
            myfile.write(self._string)

    def _suggest_toks(self, tokens, indices, suggestions,
                     can_replace_sent=False):
        """
        Ask the user to provide input on errors or style suggestions.

        Arguments:
        tokens - tokens of the sentence in question
        indices - indices of tokens to be replaced
        suggestions - a list of possible suggestions

        Optional arguments:
        can_replace_sent - should the user have the explicit option
            of replacing the entire sentence?
        """

        # Print the sentence with the desired token underlined.
        print()
        inds = range(indices[0], indices[-1]+1)
        colors.tokenprint(tokens, inds)
        phrase = ' '.join([tokens[i] for i in indices]).replace('  ', ' ')
        print('Possible suggestions for "%s":' % phrase)

        # Print list of suggestions, as well as custom options.
        if len(suggestions) == 0:
            suggestions += ['']  # Play nicely with print_rows.
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
            n = int(user_input)
            if n > 0 and n <= len(suggestions):
                ans = suggestions[n-1]
                # Replace everything between the first and last tokens.
                tokens = tokens[:indices[0]] + [ans] + tokens[indices[-1]+1:]
            elif n != 0:
                print('\n\n-------------\nINVALID VALUE\n-------------')
                tokens = self._suggest_toks(
                    tokens, indices, suggestions, can_replace_sent)
        except ValueError:
            if user_input == 'ss':
                sent = visual_edit(''.join(tokens))
                tokens = self._gen_tokens(sent)
            else:
                ans = user_input
                tokens = tokens[:indices[0]] + [ans] + tokens[indices[-1]+1:]
        return tokens

    def _thesaurus(self, word, pos):
        """Provide a list of synonyms for word."""
        new_pos = self._penn2morphy(pos)
        if new_pos is not None:
            lemma = self._lemmatizer.lemmatize(word, new_pos)
            gen_pos = self._penn2gen(pos)
            syns = Thesaurus(lemma).synonyms[gen_pos]
            if gen_pos == 'verb':
                for i, syn in enumerate(syns):
                    syn = syn.split(' ')
                    syn = ' '.join([conjugate(syn[0], pos)] + syn[1:])
                    syns[i] = syn
            if pos == 'NNS':
                for i, syn in enumerate(syns):
                    syn = syn.split(' ')
                    syn = ' '.join([pluralize(syn[0], pos)] + syn[1:])
                    syns[i] = syn
            return syns
        return []

    def _check_loop(self, error_method):
        for i, sent in enumerate(self.sentences):
            tokens = self._gen_tokens(sent)
            errors, suggests, ignore_list = error_method(tokens)
            while len(errors) > 0:
                err = errors[0]
                new_tokens = self._suggest_toks(
                    tokens, err[1], suggests[0], True)
                if new_tokens == tokens:
                    ignore_list += [err]
                    errors = errors[1:]
                    suggests = suggests[1:]
                else:
                    new_sent = ''.join(new_tokens)
                    for j in ',:;.?! ':
                        new_sent = new_sent.replace(' %s' % j, j)
                    tokens = self._gen_tokens(new_sent)
                    errors, suggests, ignore_list = error_method(
                        tokens, ignore_list)
            new_sent = ''.join(tokens)
            for j in ',:;.?! ':
                new_sent = new_sent.replace(' %s' % j, j)
            self._sentences[i] = new_sent
            self.save()
        self._clean()
        self.save()

    def _fromx_to_id(self, fromx, tox, tokens):
        i = 0
        x = 0
        ids = []
        while x < tox:
            if x >= fromx:
                ids += [i]
            x += len(tokens[i])
            i += 1
        return ids

    def _spelling_errors(self, tokens, ignore_list=[]):
        errors = []
        for j, tok in enumerate(tokens):
            if tok == ' ' or tok == '\n' or tok in punctuation:
                continue
            tup = ([tok], [j])
            if self._dict.check(tok) is False and tup not in ignore_list:
                errors += [tup]
        suggests = [self._dict.suggest(err[0][0]) for err in errors]
        return errors, suggests, ignore_list

    def _grammar_errors(self, tokens, ignore_list=[]):
        errors_gram = self._gram.check(''.join(tokens))
        # Don't check for smart quotes
        errors_gram = [err for err in errors_gram if err.ruleId != 'EN_QUOTES']

        errors = []
        suggests = []
        for err in errors_gram:
            ids = self._fromx_to_id(err.fromx, err.tox, tokens)
            toks = [tokens[i] for i in ids]
            errors += [(toks, ids)]
            errors = [err for err in errors if err not in ignore_list]
            suggests += [err.replacements]
        return errors, suggests, ignore_list

    def _homophone_errors(self, tokens, ignore_list=[]):
        errors = []
        suggests = []
        for i, tok in enumerate(tokens):
            for homophones in homophone_list:
                for h in homophones:
                    if h == tok.lower() and ([tok], [i]) not in ignore_list:
                        other_homs = [hom for hom in homophones if hom != h]
                        errors += [([tok], [i])]
                        suggests += [homophones]
                        ignore_list += [([hom], [i]) for hom in other_homs]
        return errors, suggests, ignore_list

    def _cliche_errors(self, tokens, ignore_list=[]):
        errors = []
        suggests = []
        lem = ''.join([self._lemmatizer.lemmatize(t) for t in tokens]).lower()
        for k in cliches.keys():
            if k in lem:
                fromx = lem.find(k)
                tox = fromx + len(k)
                ids = self._fromx_to_id(fromx, tox, tokens)
                toks = [tokens[i] for i in ids]
                if (toks, ids) not in ignore_list:
                    errors += [(toks, ids)]
                    suggests += [cliches[k]]
        return errors, suggests, ignore_list

    def _passive_errors(self, tokens, ignore_list=[]):
        errors = []
        suggests = []
        words, inds = self._gen_words(tokens)
        tags = self._gen_tags(words)
        verbs = [(x[0], x[1], inds[i])
                 for i, x in enumerate(tags)
                 if x[1].startswith('V')]
        verbs_lemmas = [
            self._lemmatizer.lemmatize(v[0], wn.VERB) for v in verbs]
        if 'be' in verbs_lemmas and 'VBN' in [v[1] for v in verbs]:
            # Parse takes a really long time,
            # so we check with the faster "tags" before committing.
            parse = list(dep_parser.parse(words))[0]
            nodes = parse.nodes
            vbns = [n for n in nodes.values()
                    if 'auxpass' in n['deps'].keys()]
            for word in vbns:
                be_verb_node_ids = word['deps']['auxpass']
                be_verbs = [(nodes[i]['address']-1, nodes[i]['word'])
                            for i in be_verb_node_ids]
                ids = [inds[bv[0]] for bv in be_verbs]
                ids += [inds[word['address'] - 1]]
                ids.sort()
                toks = [tokens[i] for i in ids]
                tup = (toks, ids)
                if tup not in ignore_list:
                    errors += [tup]
                    suggests += [[]]
        return errors, suggests, ignore_list

    def _nominalization_errors(self, tokens, ignore_list=[]):
        errors = []
        suggests = []
        words, inds = self._gen_words(tokens)
        tags = self._gen_tags(words)
        nouns = [(t[0], t[1], inds[i])
                 for i, t in enumerate(tags)
                 if t[1].startswith('NN')]
        nouns_lemmas = [
            (self._lemmatizer.lemmatize(n[0], wn.NOUN), n[1], n[2])
            for n in nouns]
        for i, noun in enumerate(nouns_lemmas):
            denoms = denominalize(noun[0])
            tup = ([noun[0]], [noun[2]])
            if len(denoms) > 0 and tup not in ignore_list:
                errors += [tup]
                suggests += [denoms]
        return errors, suggests, ignore_list

    def _weak_words_errors(self, tokens, ignore_list=[]):
        errors = []
        suggests = []
        words, inds = self._gen_words(tokens)
        tags = self._gen_tags(words)
        for i, w in enumerate(tags):
            pos = self._penn2morphy(w[1])
            if pos is not None:
                lemma = self._lemmatizer.lemmatize(w[0], pos)
            else:
                lemma = w[0]
            tup = ([w[0]], [inds[i]])
            if (lemma in weak_verbs
                    or lemma in weak_adjs
                    or lemma in weak_modals
                    or lemma in weak_nouns
                    ) and tup not in ignore_list:
                errors += [tup]
                suggests += [self._thesaurus(w[0], w[1])]
        return errors, suggests, ignore_list

    def _filler_errors(self, tokens, ignore_list=[]):
        errors = []
        suggests = []
        for i, tok in enumerate(tokens):
            tup = ([tok], [i])
            if tok.lower() in filler_words and tup not in ignore_list:
                errors += [tup]
                suggests += [['']]
        return errors, suggests, ignore_list

    def _adverb_errors(self, tokens, ignore_list=[]):
        errors = []
        suggests = []
        words, inds = self._gen_words(tokens)
        tags = self._gen_tags(words)
        adverbs = [x[0] for x in tags
                   if x[1].startswith('RB') and x[0].endswith('ly')]
        if len(adverbs) > 0:
            # Parse takes a really long time,
            # so we check with the faster "tags" before committing.
            parse = list(dep_parser.parse(words))[0]
            nodes = parse.nodes
            adv_modified = sorted([n for n in nodes.values()
                                   if 'advmod' in n['deps'].keys()],
                                  key=lambda k: k['address'])
            for word in adv_modified:
                adv_node_ids = word['deps']['advmod']
                advs = [(nodes[i]['address']-1, nodes[i]['word'])
                        for i in adv_node_ids
                        if nodes[i]['word'].endswith('ly')]
                ids = [inds[adv[0]] for adv in advs]
                ids += [inds[word['address'] - 1]]
                ids.sort()
                toks = [tokens[i] for i in ids]
                tup = (toks, ids)
                if len(advs) > 0 \
                   and ids[1] - ids[0] <= 5 \
                   and word['tag'] is not None \
                   and tup not in ignore_list:
                    errors += [tup]
                    suggests += [self._thesaurus(word['word'], word['tag'])]
        return errors, suggests, ignore_list

    def spelling(self):
        """Run a spell check on the text!"""
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
        print("'%s' appeard %s times (%.02f%%)." % (
            word, freq, freq/len(self._words)*100))
        ans = input(
            'Would you like to view occurances in proximity? (%s) ' % close)
        while len(ans) < 1:
            ans = input('Sorry, try again: ')
        return ans[0].lower()

    def frequent_words(self, n=10):
        """
        Print a list of the most commonly used words.

        Ask user word-per-word if they'd like to view occurances
        in close proximity.
        """
        stopwords = nltk.corpus.stopwords.words('english')
        penn_tags = nltk.pos_tag(self._tokens)
        morphy_tags = [(x, self._penn2morphy(pos)) for x, pos in penn_tags]

        lemmas = np.array([
            (x, y, self._lemmatizer.lemmatize(x.lower(), y or 'n'))
            for x, y in morphy_tags])
        distr = [(k, v) for k, v in nltk.FreqDist(lemmas[:, 2]).items()
                 if v > 1
                 and k not in stopwords
                 and k not in punctuation
                 and k != '\n'
                 and k != ' ']
        distr.sort(key=lambda x: x[1], reverse=True)

        # Print n most frequent words.
        for i, j in distr[:n]:
            print('%s: %.02f%%' % (i, j/len(self._words)*100))
        print()

        # Ask if user wants to see words in proximity.
        print('-----')
        for word, freq in distr:
            occurs = np.array([
                i for i, lem in enumerate(lemmas) if lem[2] == word])
            dists = occurs[1:] - occurs[:-1]
            # m can be less than 30 if the word occurs a lot.
            m = min(30, int(len(self._words)/freq/3))
            to_print = occurs[np.where(dists < m)]
            if len(to_print) > 0:
                print('-----')
                yn = self._ask_user(word, freq, len(to_print))
                print('-----')
                if yn == 'y':
                    for i in to_print:
                        start = max(0, i-int(m/2))
                        stop = min(i+int(1.5*m), len(lemmas))
                        tokens = lemmas[start:stop]
                        indices = np.where(tokens[:, 2] == word)[0]
                        colors.tokenprint(tokens[:, 0], indices)
                        input('Enter to continue. ')
                        print('-----')

    def visualize_length(self, char='X'):
        """Produce a visualization of sentence length."""
        for i, sent in enumerate(self._sentences):
            if sent == '\n\n':
                print()
                continue
            n = len([x for x in sent if x != ' ' and x not in punctuation])
            print('{: >6}'.format('(%s)' % (i+1)), char*n)

    def polish(self):
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
        sents = self._sentences
        for i in ',:;.?! ':
            sents = [s.replace(' %s' % i, i) for s in sents]

        self._string = ' '.join(sents)
        self._sentences = sents
        self._tokens = self._gen_tokens(self._string)
        self._words, _ = self._gen_words(self._tokens)
        self._tags = self._gen_tags(self._words)

    def _gen_sent(self, string):
        """Generate a list of sentences."""
        # Remove newlines
        paragraphs = string.split('\n')
        # Sentence tokenize
        paragraphs = [
            self._tokenizer(p) for p in paragraphs]
        # Add newlines back in
        paragraphs = [p if p != [] else ['\n\n'] for p in paragraphs]
        # Return flattened
        return [s for p in paragraphs for s in p]

    def _fix_contractions(self, tokens):
        """Treat contractions as one token, not three."""
        cont_list = ['d', 'll', 'm', 're', 's', 't', 've']
        conts = [i for i, tok in enumerate(tokens)
                 if tok == "'"
                 and i > 0 and i+1 < len(tokens)
                 and tokens[i+1] in cont_list]
        for c in conts[::-1]:
            tokens = tokens[:c-1] + [''.join(tokens[c-1:c+2])] + tokens[c+2:]
        plural_possess = [i for i, tok in enumerate(tokens)
                          if tok == "'"
                          and i > 0 and i+1 < len(tokens)
                          and tokens[i-1].endswith('s')
                          and tokens[i+1] == ' ']
        for p in plural_possess[::-1]:
            tokens = tokens[:p-1] + [''.join(tokens[p-1:p+1])] + tokens[p+1:]
        return tokens

    def _gen_tokens(self, string):
        """Generate a list of tokens."""
        tokens = nltk.tokenize.regexp_tokenize(string, '\w+|[^\w\s]|\s')
        return self._fix_contractions(tokens)

    def _gen_words(self, tokens):
        """Generate a list of words."""
        mylist = [(i, tok) for i, tok in enumerate(tokens)
                  if tok != ' ' and tok not in punctuation]
        inds = [w[0] for w in mylist]
        words = [w[1] for w in mylist]
        return words, inds

    def _gen_tags(self, words):
        """Generate a list of parts of speech tags."""
        return nltk.pos_tag(words)

    def _penn2morphy(self, penntag):
        """Quick 'translation' between Penn and Morphy POS tags."""
        morphy_tag = {'NN': wn.NOUN,
                      'JJ': wn.ADJ,
                      'VB': wn.VERB,
                      'RB': wn.ADV}
        try:
            return morphy_tag[penntag[:2]]
        except KeyError:
            return None

    def _penn2gen(self, penntag):
        """Quick 'translation' between Penn and generic POS tags."""
        gen_tag = {'NN': 'noun',
                   'JJ': 'adj',
                   'VB': 'verb',
                   'RB': 'adv'}
        try:
            return gen_tag[penntag[:2]]
        except KeyError:
            return None

    @property
    def string(self):
        return self._string

    @string.setter
    def string(self, string):
        self._string = string
        self._sentences = self._gen_sent(self._string)
        self._tokens = self._gen_tokens(self._string)
        self._words, _ = self._gen_words(self._tokens)
        self._tags = self._gen_tags(self._words)

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, sents):
        self._string = ' '.join(sents)
        self._sentences = sents
        self._tokens = self._gen_tokens(self._string)
        self._words, _ = self._gen_words(self._tokens)
        self._tags = self._gen_tags(self._words)

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, toks):
        self._string = ''.join(toks)
        self._sentences = self._gen_sent(self._string)
        self._tokens = toks
        self._words, _ = self._gen_words(self._tokens)
        self._tags = self._gen_tags(self._words)

    @property
    def words(self):
        return self._words

    @property
    def tags(self):
        return self._tags


def main():
    # Import text
    global text  # NOTE: for debugging
    args = PARSER.parse_args()
    if args.o is None:
        args.o = '%s_out_%s.txt' % (args.file, datetime.now())
    with open(args.file) as myfile:
        text = Text(''.join(myfile.readlines()), save_file=args.o,
                    lang=args.d, train_sents=args.t)

    # Check that stuff
    text.quick_check()

    # Final result
    print('\n\n%s' % text.string)

    text.save()

if __name__ == '__main__':
    main()
