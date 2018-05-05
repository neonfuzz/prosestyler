

from nltk.corpus import wordnet as wn


nominalization_endings = [
    'ance',
    'cy',
    'ence'
    'ing',
    'ion',
    'ment',
    'ness',
    'nt',
    'ology',
    'ry',
    'sis',
    'ty',
    'ure',
    ]


random_nominalizations = [
    'belief',
    'sale',
    'success',
    ]

dont_check_list = [
    # words I use waaay too much to care about
    'simulation',
    ]


def denominalize(noun):
    """Return verb forms of noun, if it is a nominalization."""
    # Clean noun for processing.
    noun = wn.morphy(noun, wn.NOUN)
    if noun is None:
        return []

    # Determine if we should check the noun.
    should_check = False
    for ending in nominalization_endings:
        if noun.endswith(ending) and noun != ending:
            should_check = True

    if noun in random_nominalizations:
        should_check = True

    if noun in dont_check_list:
        should_check = False

    if should_check is False:
        return []

    # Get sets of synonyms.
    synsets = wn.synsets(noun, wn.NOUN)

    # Lemmatize (get base word) of each synset.
    lemmas = []
    for syn in synsets:
        lemmas += syn.lemmas()

    # Get derivations from each lemma.
    derivs = []
    for lem in lemmas:
        derivs += lem.derivationally_related_forms()
    # Filter to only have verbs.
    derivs = [d.name() for d in derivs if d.synset().pos() == 'v']

    # Return words (no duplicates)
    return list(set(derivs))
