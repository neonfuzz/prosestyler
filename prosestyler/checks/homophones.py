

"""
Provide a checker for frequently-confused homophones.

Classes:
    Homophones - said homophone checker

Variables:
    HOMOPHONE_LIST (list) - lists of homophones
"""


from .base_check import BaseCheck


class Homophones(BaseCheck):
    """
    Check a text's homophones.

    Arguments:
        text (Text) - the text to check

    Iterates over each Sentence and applies a homophone check.
    Text is saved and cleaned after each iteration.
    """

    def __repr__(self):
        """Represent Homophones with a string."""
        return 'Homophones'

    def _check_sent(self, sentence, ignore_list=None):
        errors, suggests, ignore_list, messages = super()._check_sent(
            sentence, ignore_list)

        for i, tok in enumerate(sentence.tokens):
            for homophones in HOMOPHONE_LIST:
                for hom in homophones:
                    if hom == tok.lower() and ([tok], [i]) not in ignore_list:
                        other_homs = [h for h in homophones if h != hom]
                        errors += [([tok], [i])]
                        suggests += [homophones]
                        ignore_list += [([h], [i]) for h in other_homs]
        messages = [None] * len(errors)

        return errors, suggests, ignore_list, messages


HOMOPHONE_LIST = [
    ['accept', 'except'],
    ['affect', 'effect'],
    ['air', 'heir'],
    ['allowed', 'aloud'],
    ['ant', 'aunt'],
    ['are', 'our'],
    ['bare', 'bear'],
    ['be', 'bee'],
    ['berry', 'bury'],
    ['board', 'bored'],
    ['brake', 'break'],
    ['breach', 'breech'],
    ['breaches', 'breeches'],
    ['bread', 'bred'],
    ['but', 'butt'],
    ['buy', 'by'],
    ['capital', 'capitol'],
    ['coarse', 'course'],
    ['complement', 'compliment'],
    ['creak', 'creek'],
    ['data', 'datum'],
    ['die', 'dye'],
    ['ensure', 'insure'],
    ['feat', 'feet'],
    ['fisher', 'fissure'],
    ['for', 'four'],
    ['hear', 'here'],
    ['hole', 'whole'],
    ['it\'s', 'its'],
    ['know', 'no'],
    ['lay', 'lie'],
    ['loose', 'lose'],
    ['meat', 'meet'],
    ['might', 'mite'],
    ['one', 'won'],
    ['pair', 'pear'],
    ['peak', 'peek', 'pique'],
    ['pore', 'pour'],
    ['principal', 'principle'],
    ['read', 'red'],
    ['read', 'reed'],
    ['road', 'rode'],
    ['segue', 'segway'],
    ['sew', 'so'],
    ['sight', 'site'],
    ['stalk', 'stock'],
    ['stalked', 'stocked'],
    ['tack', 'tact'],
    ['tail', 'tale'],
    ['than', 'then'],
    ['their', 'there', 'they\'re'],
    ['threw', 'through'],
    ['thyme', 'time'],
    ['to', 'too', 'two'],
    ['toe', 'tow'],
    ['tore', 'tour'],
    ['weak', 'week'],
    ['weather', 'whether'],
    ['were', 'where'],
    ['wheeled', 'wield'],
    ['which', 'witch'],
    ['whine', 'wine'],
    ['you\'re', 'your'],
    ]
