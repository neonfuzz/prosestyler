

"""
Tools to detect and correct nominalizations.
"""


# NOTE: 'consistent', 'coherent' are not nouns,
#       but they definitely seem like nominalizations


from thesaurus import get_synonyms
from sentence import NLP


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


def denominalize(noun_lemma):
    """Return verb forms of noun, if it is a nominalization."""
    # Determine if we should check the noun.
    should_check = noun_lemma.endswith(NOMINALIZATION_ENDINGS)

    if noun_lemma in RANDOM_NOMINALIZATIONS:
        should_check = True

    if noun_lemma in DONT_CHECK_LIST:
        should_check = False

    if should_check is False:
        return []

    syns = get_synonyms(noun_lemma)
    docs = [NLP(s) for s in syns]
    verbs = [d[0].lemma_ for d in docs if d[0].tag_.startswith('V')]

    # Remove duplicates while preserving order.
    return_verbs = []
    for v in verbs:
        if v not in return_verbs:
            return_verbs.append(v)
    return return_verbs
