

from math import ceil
from string import punctuation

import nltk
from nltk.corpus import wordnet as wn


def fromx_to_id(fromx, tox, tokens):
    """
    Given character indices, return token indices

    Arguments:
        fromx - start index of character string
        tox - end index of character string
        tokens - tokenized sentence

    Returns:
        list of indices corresponding to selected tokens
    """
    i = 0
    x = 0
    ids = []
    while x < tox:
        if x >= fromx:
            ids += [i]
        x += len(tokens[i])
        i += 1
    return ids


def gen_sent(string):
    """Generate a list of sentences."""
    tokenizer = nltk.data.load(
        'tokenizers/punkt/english.pickle').tokenize
    # Remove newlines
    paragraphs = string.split('\n')
    # Sentence tokenize
    paragraphs = [
        tokenizer(p) for p in paragraphs]
    # Add newlines back in
    paragraphs = [p if p != [] else ['\n\n'] for p in paragraphs]
    # Return flattened
    #  return [Sentence(s) for p in paragraphs for s in p]
    return [s for p in paragraphs for s in p]


def gen_tokens(string):
    """Generate a list of tokens."""
    tokens = nltk.tokenize.regexp_tokenize(string, r'\w+|[^\w\s]|\s')
    return fix_contractions(tokens)


def gen_words(tokens):
    """Generate a list of words."""
    mylist = [(i, tok) for i, tok in enumerate(tokens)
              if tok != ' ' and tok not in punctuation]
    inds = [w[0] for w in mylist]
    words = [w[1] for w in mylist]
    return words, inds


def gen_tags(words):
    """Generate a list of parts of speech tags."""
    return nltk.pos_tag(words)


def fix_contractions(tokens):
    """Treat contractions as one token, not three."""
    cont_list = ['d', 'll', 'm', 're', 's', 't', 've']
    conts = [i for i, tok in enumerate(tokens)
             if tok == "'"
             and i > 0 and i+1 < len(tokens)
             and tokens[i+1] in cont_list]
    for cont in conts[::-1]:
        tokens = tokens[:cont-1] \
                 + [''.join(tokens[cont-1:cont+2])] \
                 + tokens[cont+2:]
    plural_possess = [i for i, tok in enumerate(tokens)
                      if tok == "'"
                      and i > 0 and i+1 < len(tokens)
                      and tokens[i-1].endswith('s')
                      and tokens[i+1] == ' ']
    for plupos in plural_possess[::-1]:
        tokens = tokens[:plupos-1] \
                 + [''.join(tokens[plupos-1:plupos+1])] \
                 + tokens[plupos+1:]
    return tokens


def now_checking_banner(word):
    """Print a pretty banner on the output."""
    mystr = '---  Now Checking %s!  ---' % word.title()
    dashstr = '-' * len(mystr)
    print('\n\n')
    print(dashstr)
    print(mystr)
    print(dashstr)


def penn2gen(penntag):
    """Quick 'translation' between Penn and generic POS tags."""
    gen_tag = {'NN': 'noun',
               'JJ': 'adj.',
               'VB': 'verb',
               'RB': 'adv.'}
    try:
        return gen_tag[penntag[:2]]
    except KeyError:
        return None


def penn2morphy(penntag):
    """Quick 'translation' between Penn and Morphy POS tags."""
    morphy_tag = {'NN': wn.NOUN,
                  'JJ': wn.ADJ,
                  'VB': wn.VERB,
                  'RB': wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except KeyError:
        return None


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
    row_ind = 0
    # Order stuff to read down each column.
    # (rather than across each row).
    for i, j in enumerate(lis):
        rows[row_ind] = rows[row_ind] + ['(%s)' % (i+1), j]
        row_ind = (row_ind+1) % nrows
    while row_ind != 0:
        rows[row_ind] = rows[row_ind] + ['', '']
        row_ind = (row_ind+1) % nrows
    for row in rows:
        print(mystr.format(*row))
