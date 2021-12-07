

"""
Check for nominalizations.

Classes:
    Nominalizations - said nominalization checker

Variables:
    NOMINALIZATION_ENDINGS (tuple) - word endings which imply nominalization
    RANDOM_NOMINALIZATIONS (list) - additional nominalizations
    DONT_CHECK_LIST (list) - ignore these nominalizations.

Functions:
    nominalize_check - check if a word is a nominalization
    filter_syn_verbs - filter synonyms to be only verbs
"""


# pylint: disable=unused-import
#   `pyinflect` is used, just quietly.
import pyinflect

from .base_check import BaseCheck
from ..sentence import NLP
from ..tools.thesaurus import THESAURUS


class Nominalizations(BaseCheck):
    """
    Check a text's use of nominalizations.

    Arguments:
        text (Text) - the text to check

    Iterates over each Sentence and applies a homophone check.
    Text is saved and cleaned after each iteration.
    """

    def __repr__(self):
        """Represent Nominalizations with a string."""
        return 'Nominalizations'

    def _check_sent(self, sentence, ignore_list=None):
        errors, suggests, ignore_list, messages = super()._check_sent(
            sentence, ignore_list)

        nom_nouns = [n for n in sentence.nodes
                     if n.tag_.startswith('NN')
                     and nominalize_check(n.lemma_)]
        for noun in nom_nouns:
            syns = THESAURUS.get_synonyms(noun.text)
            denoms = filter_syn_verbs(syns)

            raw_ids = [noun.i]
            raw_ids += [c.i for c in noun.children]
            try:
                raw_ids = [i-sentence.nodes.start for i in raw_ids]
            except AttributeError:
                pass
            ids = [sentence.inds[i] for i in raw_ids]
            ids.sort()
            tup = ([noun.text, ids])
            if tup not in ignore_list:
                errors += [tup]
                suggests += [denoms]

        messages = [None] * len(errors)

        return errors, suggests, ignore_list, messages


NOMINALIZATION_ENDINGS = (
    'ance',
    'cy',
    'ence',
    'ency',
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
    'disbelief',
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
    'need',
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
    'anything',
    'consultant',
    'creature',
    'everything',
    'future',
    'geometry',
    'nothing',
    'possibility',
    'property',
    'reality',
    'research',
    'simulation',
    'something',
    'temperature',
    'thing',
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


def filter_syn_verbs(syns, tense='VBG'):
    """Return only verb synonyms, conjugated."""
    docs = [NLP(s) for s in syns]
    verbs = [d[0]._.inflect(tense) for d in docs]
    verbs = [v for v in verbs if v]

    # Remove duplicates while preserving order.
    return_verbs = []
    for verb in verbs:
        if verb not in return_verbs:
            return_verbs.append(verb)
    return return_verbs
