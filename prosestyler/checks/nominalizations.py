

"""
Detect and correct nominalizations.

Variables:
    NOMINALIZATION_ENDINGS (tuple) - word endings which imply nominalization
    RANDOM_NOMINALIZATIONS (list) - additional nominalizations
    DONT_CHECK_LIST (list) - ignore these nominalizations.

Functions:
    nominalize_check - check if a word is a nominalization
    filter_syn_verbs - filter synonyms to be only verbs
"""


# NOTE: 'consistent', 'coherent' are not nouns,
#       but they definitely seem like nominalizations


from ..sentence import NLP


NOMINALIZATION_ENDINGS = (
    'ance',
    'cy',
    'ence'
    'ing',
    'tion',
    'ment',
    'ness',
    'nt',
    'ology',
    'ry',
    'sis',
    'ty',
    'ure',
    )


RANDOM_NOMINALIZATIONS = [
    'belief',
    'sale',
    'success',
    ]

DONT_CHECK_LIST = [
    # words I use waaay too much to care about
    'simulation',
    ]


def nominalize_check(noun_lemma):
    """Return verb forms of noun, if it is a nominalization."""
    # Determine if we should check the noun.
    should_check = noun_lemma.endswith(NOMINALIZATION_ENDINGS)
    if noun_lemma in RANDOM_NOMINALIZATIONS:
        should_check = True
    if noun_lemma in DONT_CHECK_LIST:
        should_check = False
    return should_check


def filter_syn_verbs(syns):
    """Return only verb synonyms."""
    docs = [NLP(s) for s in syns]
    verbs = [d[0].lemma_ for d in docs if d[0].tag_.startswith('V')]

    # Remove duplicates while preserving order.
    return_verbs = []
    for verb in verbs:
        if verb not in return_verbs:
            return_verbs.append(verb)
    return return_verbs
