"""
Provide a checker for passive voice.

Classes:
    Passive - said passive voice checker
    PassiveClause - hold information about a clause in passive voice

Variables:
    SUBJ_OBJ (dict) - mapping of subject pronouns to object pronouns
"""


from copy import copy

# pylint: disable=unused-import
#   `pyinflect` is used, just behind the scenes.
import pyinflect

from .base_check import BaseCheck


SUBJ_OBJ = {
    "i": "me",
    "we": "us",
    "he": "him",
    "she": "her",
    "they": "them",
    "who": "whom",
}


def _is_proper(obj):
    if not obj:
        return True
    return obj.root.pos_ == "PROPN"


def _subj_to_obj(subj):
    if isinstance(subj, list):
        return "[OBJECT]"
    try:
        return SUBJ_OBJ.get(subj.lower_, subj.text)
    except AttributeError:
        return subj.text


def _obj_to_subj(obj):
    if isinstance(obj, list):
        return "[SUBJECT]"
    try:
        subj = {v: k for k, v in SUBJ_OBJ.items()}.get(obj.lower_, obj.text)
    except AttributeError:
        subj = obj.text
    if subj == "i":
        subj = "I"
    return subj


class PassiveClause:
    """
    Represent a clause in the passive voice.

    Instance Attributes:
        nodes (spacy.Span) - the sentence from whence the clause came
        verb (spacy.Token) - the passive verb
        aux (list of Spacy.Token) - auxillary helpers to `verb`
        be_verbs (list of Spacy.Token) - auxpass verb to `verb`
        subj (spacy.Span) - subject of the clause
        dobj (spacy.Span) - direct object of the clause
        iobj (spacy.Span) - indirect object of the clause
        ids (list of ints) - list of indices, as related to `nodes`

    Methods:
        suggest - suggest an active clause
    """

    def __init__(self, nodes, verb):
        """
        Initialize PassiveClause.

        Arguments:
            nodes (spacy.Span) - the sentence from whence the clause came
            verb (spacy.Token) - the passive verb
        """
        self.nodes = nodes
        self.verb = verb
        self.aux = [
            c for c in self.verb.children if c.dep_ in ("aux", "advmod")
        ]
        self.be_verbs = [c for c in verb.children if c.dep_ == "auxpass"]
        self.subj = self._get_subject_chunk()
        self.dobj = self._get_object_chunk()
        self.iobj = self._get_object_chunk("to")

    def __repr__(self):
        """Represent PassiveClause."""
        return (
            f"{self.subj} {self.be_verbs} {self.aux} {self.verb} "
            f"{{by}} {self.dobj} {{to}} {self.iobj}"
        )

    def _get_subject_chunk(self):
        try:
            subj = [
                c for c in self.verb.children if c.dep_.startswith("nsubj")
            ]
            subj = subj or [
                c for c in self.nodes if c.dep_.startswith("nsubj")
            ]
            subj = subj[0]
            subj_chunk = [
                nc for nc in self.nodes.noun_chunks if nc.root == subj
            ][0]
            return subj_chunk
        except IndexError:
            return []

    def _get_object_chunk(self, prep="by"):
        try:
            adp = [c for c in self.verb.children if c.lower_ == prep]
            adp = adp or [c for c in self.nodes if c.lower_ == prep]
            adp = adp[0]
            obj = [c for c in adp.children if c.dep_.endswith("obj")][0]
            obj_chunk = [
                nc for nc in self.nodes.noun_chunks if nc.root == obj
            ][0]
            return obj_chunk
        except IndexError:
            return []

    def _conjugate(self, subj):
        aux = copy(self.aux)
        try:
            # Match tense of auxpass "be" verb.
            tag = self.be_verbs[0].tag_
            # Match plurality of subject.
            if subj and subj.root.tag_.endswith("S"):
                if tag == "VBZ":
                    tag = "VBP"
                for i, tok in enumerate(aux):
                    if tok.tag_ == "VBZ":
                        aux[i] = tok._.inflect("VBP", 1)
            elif subj and tag == "VBP":
                tag = "VBZ"
            # Conjugate.
            conj = self.verb._.inflect(tag) or self.verb.lower_
        except IndexError:
            conj = self.verb.lower_
        return " ".join([str(a) for a in aux] + [conj])

    @property
    def ids(self):
        """List of indices that participate in the clause."""
        ids = [self.verb.i]
        ids += [a.i for a in self.aux]
        ids += [bv.i for bv in self.be_verbs]
        ids += [s.i for s in self.subj]
        ids += [c.i for c in self.verb.children if c.lower_ == "by"]
        ids += [o.i for o in self.dobj]
        ids += [o.i for o in self.iobj]
        ids.sort()
        return ids

    def suggest(self):
        """
        Suggest an active clause.

        Returns:
            suggest (list of strings) - final suggestions
        """
        # Conjugate verb to new subject.
        conj = self._conjugate(self.dobj)

        # Remember things for later casing.
        # Note that variable names refer to new subject and object.
        dobj_proper = _is_proper(self.subj)
        subj_proper = _is_proper(self.dobj)
        is_sent_start = (self.ids[0] - self.nodes[:].start) == 0

        # Swap subject and dobject.
        # These are strings now.
        dobj, subj = _subj_to_obj(self.subj), _obj_to_subj(self.dobj)

        # Casing.
        if not is_sent_start and not subj_proper:
            subj = subj.lower()
        if is_sent_start and subj and subj == subj.lower():
            subj = " ".join(
                [
                    tok.title() if i == 0 else tok
                    for i, tok in enumerate(subj.split())
                ]
            )
        if not dobj_proper:
            dobj = dobj.lower()

        # Formulate and return suggestions.
        ret = [f"{subj} {conj} {dobj}"]
        if self.iobj:
            ret.append(f"{subj} {conj} {dobj} to {self.iobj}")
        return ret


class Passive(BaseCheck):
    """
    Check a text's use of passive voice.

    Arguments:
        text (Text) - the text to check

    Iterates over each Sentence and applies a homophone check.
    Text is saved and cleaned after each iteration.
    """

    _description = (
        "Passive voice takes the focus off the person doing the "
        "action, and makes your writing more impersonal and drab. This "
        "can be desired if you're writing for a formal audience, "
        "but even then should be used sparingly."
    )

    def __repr__(self):
        """Represent Passive with a string."""
        return "Passive Voice"

    def _check_sent(self, sentence, ignore_list=None):
        errors, suggests, ignore_list, messages = super()._check_sent(
            sentence, ignore_list
        )

        vbns = [
            n
            for n in sentence.nodes
            if "auxpass" in [c.dep_ for c in n.children]
            and n.tag_.startswith("V")
        ]
        for verb in vbns:
            clause = PassiveClause(sentence.nodes, verb)
            ids = [
                sentence.inds[i - sentence.nodes[:].start] for i in clause.ids
            ]
            tup = ([sentence.tokens[i] for i in ids], ids)
            if tup not in ignore_list:
                errors += [tup]
                suggests.append(clause.suggest())

        messages = [None] * len(errors)

        return errors, suggests, ignore_list, messages
