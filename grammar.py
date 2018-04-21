#!/bin/python3


from math import ceil
from string import punctuation

import argparse
import enchant  # Spell Check
import language_check  # Grammar Check
import lxml  # Needed for thesaurus
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import numpy as np
from py_thesaurus import Thesaurus  # Thesaurus

from cliches import cliches
import colors
from filler_words import filler_words
from homophone_list import homophone_list
from nominalizations import denominalize
from weak_words import weak_adjs, weak_nouns, weak_verbs


PARSER = argparse.ArgumentParser(description='Perform a deep grammar check.')
PARSER.add_argument('file', help='The file to be analyzed.')
PARSER.add_argument(
    '-o', default='out.txt', type=str, metavar='outfile',
    help='Name of output file (default out.txt)')
PARSER.add_argument(
    '-d', default='en_US', type=str, metavar='dictionary',
    help='Which dictionary to use (default: en_US)')


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


def thesaurus(word):
    """Provide a list of synonyms that are not weak words."""
    return [w for w in Thesaurus(word).get_synonym()
            if w not in filler_words
            and w not in weak_adjs
            and w not in weak_nouns
            and w not in weak_verbs]


class Text(object):
    """
    A fancy text object which can provide style suggestions.

    Instance variables:
    string - a string of the entire text
    sentences - a list of sententces within the text
    tokens - a list of tokens
    words - a list of words
    tags - a list of words and their parts of speech tags

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

    def __init__(self, string, lang='en_US'):
        """
        Initialize a Text object.

        Arguments:
        string - the text string to be parsed

        Optional arguments:
        lang - the language to be used (not fully implemented)
        """

        # Define dictionaries etc.
        # Train Punkt tokenizer on text (for better sentence breaks)
        self._tokenizer = nltk.tokenize.PunktSentenceTokenizer(string)
        self._dict = enchant.DictWithPWL(lang, 'scientific_word_list.txt')
        self._gram = language_check.LanguageTool(lang)
        self._gram.disable_spellchecking()
        self._lemmatizer = nltk.stem.WordNetLemmatizer()

        # Make all the things
        self._string = string.replace('“', '"').replace('”', '"')
        self._sentences = self._gen_sent(self._string)
        self._clean()  # Also makes tokens, words, tags.

    def save(self, filename):
        """Save the object to file."""
        with open(filename, 'w') as myfile:
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
        colors.tokenprint(tokens, indices)
        phrase = ''.join([tokens[i] for i in indices])
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
            if n > 0:
                ans = suggestions[n-1]
                tokens = tokens[:indices[0]] + [ans] + tokens[indices[-1]+1:]
        except ValueError:
            if user_input == 'ss':
                sent = input('Replace entire sentence: ')
                tokens = self._gen_tokens(sent)
            else:
                ans = user_input
                tokens = tokens[:indices[0]] + [ans] + tokens[indices[-1]+1:]
        return tokens

    def spelling(self):
        """Run a spell check on the text!"""
        # Courtesy of http://www.jpetrie.net/scientific-word-list-for-spell-checkersspelling-dictionaries/
        sents = []
        for sent in self._sentences:
            tokens = self._gen_tokens(sent)
            for i, tok in enumerate(tokens):
                if tok == ' ' or tok == '\n' or tok in punctuation:
                    continue
                if self._dict.check(tok) is False:
                    tokens = self._suggest_toks(
                        tokens, [i], self._dict.suggest(tok))
            sents += [''.join(tokens)]
        self.sentences = sents

    def grammar(self):
        """Run a grammar check on the text!"""
        sents = []
        for sent in self._sentences:
            errors = self._gram.check(sent)
            # Don't check for smart quotes
            errors = [err for err in errors if err.ruleId != 'EN_QUOTES']
            while len(errors) > 0:
                err = errors[0]
                if err.fromx < 0 or err.tox < 0:
                    errors = errors[1:]
                    continue
                tokens = [
                    sent[:err.fromx], sent[err.fromx:err.tox], sent[err.tox:]]
                tokens = self._suggest_toks(
                    tokens, [1], err.replacements, True)
                sent_new = ''.join(tokens)
                if sent_new == sent:
                    errors = errors[1:]
                else:
                    errors = self._gram.check(sent_new)
                sent = sent_new
            sents += [sent]
        self.sentences = sents

    def homophone_check(self):
        """Point out every single homophone, for good measure."""
        sents = []
        for sent in self._sentences:
            tokens = self._gen_tokens(sent)
            for i, tok in enumerate(tokens):
                for homophones in homophone_list:
                    for h in homophones:
                        if h == tok.lower():
                            tokens = self._suggest_toks(
                                tokens, [i], homophones)
            sents += [''.join(tokens)]
        self.sentences = sents

    def cliches(self):
        """Highlight cliches and offer suggestions."""
        sents = []
        for sent in self._sentences:
            tokens = self._gen_tokens(sent)
            lem = ''.join([
                self._lemmatizer.lemmatize(t) for t in tokens]).lower()
            for k in cliches.keys():
                if k in lem:
                    start = max(0, sent.find(k))
                    if start == 0:
                        end = len(sent)
                    else:
                        end = min(len(sent), start+len(k))
                    tokens = [sent[:start], sent[start:end], sent[end:]]
                    tokens = self._suggest_toks(
                        tokens, [1], cliches[k])
                    sent = ''.join(tokens)
            sents += [sent]
        self.sentences = sents
        self._clean()

    def passive_voice(self):
        """Point out (many) instances of passive voice."""
        sents = []
        for sent in self._sentences:
            tokens = self._gen_tokens(sent)
            words = self._gen_words(tokens=tokens)
            tags = self._gen_tags(words=words)
            vbns = [w[0] for w in tags if w[1] == 'VBN']
            if len(vbns) > 0:
                for vbn in vbns:
                    i = words.index(vbn)
                    if tags[i-1][1].startswith('V'):
                        j = tokens.index(vbn)
                        tokens = self._suggest_toks(
                            tokens, range(j-2, j+1), [], True)
                        words = self._gen_words(tokens=tokens)
                        tags = self._gen_tags(words=words)
                        sent = ''.join(tokens)
            sents += [sent]
        self.sentences = sents

    def nominalizations(self):
        """Find many nominalizations and suggest stronger verbs."""
        sents = []
        for sent in self._sentences:
            tokens = self._gen_tokens(sent)
            nouns = [w[0] for w in self._gen_tags(sent=sent)
                     if w[1].startswith('NN')]
            for noun in nouns:
                denoms = denominalize(noun)
                if len(denoms) > 0 and noun in tokens:
                    tokens = self._suggest_toks(
                        tokens, [tokens.index(noun)], denoms, True)
            sents += [''.join(tokens)]
        self.sentences = sents

    def weak_words(self):
        """Find weak words and suggest stronger ones."""
        sents = []
        for sent in self._sentences:
            tokens = self._gen_tokens(sent)
            tags = self._gen_tags(sent=sent)
            for w in tags:
                pos = self._penn2morphy(w[1])
                if pos is not None:
                    lemma = self._lemmatizer.lemmatize(w[0], pos)
                    if (lemma in weak_verbs
                            or lemma in weak_adjs
                            or lemma in weak_nouns) \
                            and w[0] in tokens:
                        tokens = self._suggest_toks(
                            tokens, [tokens.index(w[0])],
                            thesaurus(w[0]), True)
            sents += [''.join(tokens)]
        self.sentences = sents

    def filler_words(self):
        """Point out filler words and offer to delete them."""
        sents = []
        for sent in self._sentences:
            tokens = self._gen_tokens(sent)
            for i, tok in enumerate(tokens):
                if tok.lower() in filler_words:
                    tokens = self._suggest_toks(tokens, [i], [], True)
            sents += [''.join(tokens)]
        self.sentences = sents
        self._clean()

    def adverbs(self):
        """Find adverbs and verbs, offer better verbs."""
        sents = []
        for sent in self._sentences:
            tokens = self._gen_tokens(sent)
            tags = self._gen_tags(sent=sent)
            adverbs = [(i, w[0]) for i, w in enumerate(tags)
                        if w[1].startswith('RB')]
            for i, adv in adverbs:
                verbs = [w[0] for w in tags[i-2:i+3] if w[1].startswith('V')]
                if len(verbs) > 0:
                    verb = verbs[-1]  # ?
                    adv_i = tokens.index(adv)
                    verb_i = tokens.index(verb)
                    start = max(0, min(adv_i, verb_i))
                    end = max(adv_i, verb_i)
                    tokens = self._suggest_toks(
                        tokens, range(start, end+1), thesaurus(verb), True)
                    sent = ''.join(tokens)
            sents += [sent]
        self.sentences = sents

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
        self._words = self._gen_words(tokens=self._tokens)
        self._tags = self._gen_tags(words=self._words)

    def _gen_sent(self, string):
        """Generate a list of sentences."""
        # Remove newlines
        paragraphs = string.split('\n')
        # Sentence tokenize
        paragraphs = [
            self._tokenizer.sentences_from_text(p) for p in paragraphs]
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
        return tokens

    def _gen_tokens(self, string):
        """Generate a list of tokens."""
        tokens = nltk.tokenize.regexp_tokenize(string, '\w+|[^\w\s]|\s')
        return self._fix_contractions(tokens)

    def _gen_words(self, tokens=None, sent=None):
        """Generate a list of words."""
        if sent is not None:
            tokens = self._gen_tokens(sent)
        if tokens is not None:
            return [
                tok for tok in tokens if tok != ' ' and tok not in punctuation]
        raise NameError('_gen_words must be passed either tokens or sent.')

    def _gen_tags(self, words=None, sent=None):
        """Generate a list of parts of speech tags."""
        if words is not None:
            return nltk.pos_tag(words)
        if sent is not None:
            return nltk.pos_tag(self._gen_words(sent=sent))
        raise NameError('_gen_tags must be passed either words or sent.')

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

    @property
    def string(self):
        return self._string

    @string.setter
    def string(self, string):
        self._string = string
        self._sentences = self._gen_sent(self._string)
        self._tokens = self._gen_tokens(self._string)
        self._words = self._gen_words(tokens=self._tokens)
        self._tags = self._gen_tags(words=self._words)

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, sents):
        self._string = ' '.join(sents)
        self._sentences = sents
        self._tokens = self._gen_tokens(self._string)
        self._words = self._gen_words(tokens=self._tokens)
        self._tags = self._gen_tags(words=self._words)

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, toks):
        self._string = ''.join(toks)
        self._sentences = self._gen_sent(self._string)
        self._tokens = toks
        self._words = self._gen_words(tokens=self._tokens)
        self._tags = self._gen_tags(words=self._words)

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
    with open(args.file) as myfile:
        text = Text(''.join(myfile.readlines()))

    # Check that stuff
    text.polish()

    # Final result
    print('\n\n%s' % text.string)

    text.save(args.o)

if __name__ == '__main__':
    main()
