

"""
Provide a checker for passive voice.

Classes:
    Passive - said passive voice checker
"""


# pylint: disable=unused-import
#   `pyinflect` is used, just behind the scenes.
import pyinflect

from .base_check import BaseCheck


SUBJ_OBJ = {
    'i': 'me',
    'we': 'us',
    'he': 'him',
    'she': 'her',
    'they': 'them',
    'who': 'whom',
    }


def _get_subject_chunk(nodes, vbn):
    try:
        subj = [c for c in vbn.children if c.dep_.startswith('nsubj')][0]
        subj_chunk = [nc for nc in nodes.noun_chunks if nc.root == subj][0]
        return subj_chunk
    except IndexError:
        return []


def _get_object_chunk(nodes, vbn):
    try:
        by_prep = [c for c in vbn.children if c.lower_ == 'by'][0]
        obj = [c for c in by_prep.children if c.dep_.endswith('obj')][0]
        obj_chunk = [nc for nc in nodes.noun_chunks if nc.root == obj][0]
        return obj_chunk
    except IndexError:
        return []


# TODO: match singular/plural of the (new) subject
# TODO: future tense isn't at all caught
def _conjugate(verb, be_verbs):
    try:
        conj = verb._.inflect(be_verbs[0].tag_) or verb.lower_
    except IndexError:
        conj = verb.lower_
    return conj


def _subject_to_object(subj):
    if isinstance(subj, list):
        return '[OBJECT]'
    return SUBJ_OBJ.get(subj.lower_, subj.text)


def _object_to_subject(obj):
    if isinstance(obj, list):
        return '[SUBJECT]'
    subj = {v: k for k, v in SUBJ_OBJ.items()}.get(obj.lower_, obj.text)
    if subj == 'i':
        subj = 'I'
    return subj


class Passive(BaseCheck):
    """
    Check a text's use of passive voice.

    Arguments:
        text (Text) - the text to check

    Iterates over each Sentence and applies a homophone check.
    Text is saved and cleaned after each iteration.
    """

    _description = (
        'Passive voice takes the focus off the person doing the '
        'action, and makes your writing more impersonal and drab. This '
        "can be desired if you're writing for a formal audience, "
        'but even then should be used sparingly.')

    def __repr__(self):
        """Represent Passive with a string."""
        return 'Passive Voice'

    def _check_sent(self, sentence, ignore_list=None):
        errors, suggests, ignore_list, messages = super()._check_sent(
            sentence, ignore_list)

        vbns = [n for n in sentence.nodes
                if 'auxpass' in [c.dep_ for c in n.children]
                and n.tag_.startswith('V')]
        for verb in vbns:
            be_verbs = [c for c in verb.children if c.dep_ == 'auxpass']
            subj = _get_subject_chunk(sentence.nodes, verb)
            obj = _get_object_chunk(sentence.nodes, verb)

            # Passive voice involves a lot of tokens.
            # Let's keep track.
            ids = [verb.i]
            ids += [bv.i for bv in be_verbs]
            ids += [s.i for s in subj]
            ids += [c.i for c in verb.children if c.lower_ == 'by']
            ids += [o.i for o in obj]
            ids = [sentence.inds[i-sentence.nodes[:].start] for i in ids]
            ids.sort()
            tup = (
                [sentence.tokens[i] for i in ids],  # tokens
                ids)

            # Get suggestion string in order.
            conj = _conjugate(verb, be_verbs)
            obj, subj = _subject_to_object(subj), _object_to_subject(obj)
            if tup not in ignore_list:
                errors += [tup]
                suggests += [[f'{subj} {conj} {obj}']]

        messages = [None] * len(errors)

        return errors, suggests, ignore_list, messages
