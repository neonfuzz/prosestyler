

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


from ..sentence import NLP


NOMINALIZATION_ENDINGS = (
    'ance',
    'cy',
    'ence',
    'ing',
    'ment',
    'ness',
    'nt',
    'ology',
    'ry',
    'sion',
    'sis',
    'tion',
    'ty',
    'ure',
    )


RANDOM_NOMINALIZATIONS = [
    'abstract',
    'addict',
    'address',
    'advocate',
    'affect',
    'affix',
    'alloy',
    'ally',
    'annex',
    'array',
    'assay',
    'attribute',
    'belay',
    'belief',
    'bombard',
    'change',
    'combat',
    'combine',
    'commune',
    'compact',
    'complex',
    'composite',
    'compost',
    'compound',
    'compress',
    'concert',
    'conduct',
    'confect',
    'confines',
    'conflict',
    'conscript',
    'conserve',
    'consist',
    'console',
    'consort',
    'construct',
    'consult',
    'contest',
    'contract',
    'contrast',
    'converse',
    'convert',
    'convict',
    'costume',
    'cushion',
    'debut',
    'decrease',
    'default',
    'defect',
    'desert',
    'detail',
    'dictate',
    'digest',
    'discard',
    'discharge',
    'discourse',
    'embed',
    'envelope',
    'escort',
    'essay',
    'excise',
    'exploit',
    'export',
    'extract',
    'foretaste',
    'foretoken',
    'forward',
    'impact',
    'import',
    'impound',
    'impress',
    'incense',
    'incline',
    'increase',
    'inlay',
    'insert',
    'insult',
    'intercept',
    'interchange',
    'intercross',
    'interdict',
    'interlink',
    'interlock',
    'intern',
    'interplay',
    'interspace',
    'interweave',
    'intrigue',
    'invert',
    'invite',
    'involute',
    'mandate',
    'mentor',
    'mismatch',
    'murder',
    'object',
    'offset',
    'overlap',
    'overlay',
    'overlook',
    'override',
    'overrun',
    'overturn',
    'perfect',
    'perfume',
    'permit',
    'pervert',
    'prefix',
    'proceeds',
    'process',
    'produce',
    'progress',
    'project',
    'protest',
    'purport',
    'rebel',
    'recall',
    'recap',
    'recess',
    'recoil',
    'record',
    'redirect',
    'redo',
    'redress',
    'refill',
    'refresh',
    'refund',
    'refuse',
    'regress',
    'rehash',
    'reject',
    'relapse',
    'relay',
    'remake',
    'repeat',
    'research',
    'reserve',
    'reset',
    'retake',
    'retract',
    'retread',
    'rewrite',
    'sale',
    'separate',
    'start',
    'subject',
    'success',
    'survey',
    'suspect',
    'transect',
    'transfer',
    'transform',
    'transport',
    'transpose',
    'traverse',
    'underlay',
    'underline',
    'underscore',
    'update',
    'upgrade',
    'uplift',
    'upset',
    'use',
    ]


DONT_CHECK_LIST = [
    'creature',
    'future',
    'geometry',
    'possibility',
    'property',
    'reality',
    'simulation',
    'temperature',
    'velocity',
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
