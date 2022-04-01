"""
Provide a checker for filler words.

Classes:
    Filler - said filler checker

Variables:
    FILLER_WORDS (list) - words which can usually be deleted.
"""


from .base_check import BaseCheck


class Filler(BaseCheck):
    """
    Check a text's use of filler words.

    Arguments:
        text (Text) - the text to check

    Iterates over each Sentence and applies a homophone check.
    Text is saved and cleaned after each iteration.
    """

    _description = (
        "Some words can be outright deleted without changing your "
        "meaning. These words slow down the reader and make your "
        "writing more stilted. There will be a lot of suggestions in "
        "this check and you'll find you can apply approximately half. "
        "Don't spend too much time on any individual suggestion, or "
        "this check will take forever."
    )

    def __repr__(self):
        """Represent Filler with a string."""
        return "Filler Words"

    def _check_sent(self, sentence, ignore_list=None):
        errors, suggests, ignore_list, messages = super()._check_sent(
            sentence, ignore_list
        )

        for i, tok in enumerate(sentence.tokens):
            tup = ([tok], [i])
            if tok.lower() in FILLER_WORDS and tup not in ignore_list:
                errors += [tup]
                suggests += [[""]]
        messages = [None] * len(errors)

        return errors, suggests, ignore_list, messages


FILLER_WORDS = [
    "about",
    "absolutely",
    "across",
    "actually",
    "after",
    "afterwards",
    "again",
    "all",
    "alone",
    "along",
    "already",
    "also",
    "always",
    "among",
    "amongst",
    "amoungst",
    "any",
    "anyhow",
    "anything",
    "anyway",
    "around",
    "as",
    "back",
    "basically",
    "because",
    "began",
    "begin",
    "begun",
    "behind",
    "between",
    "certainly",
    "commonly",
    "completely",
    "current",
    "currently",
    "dear",
    "definitely",
    "down",
    "due",
    "during",
    "each",
    "either",
    "entire",
    "even",
    "ever",
    "every",
    "everything",
    "everywhere",
    "forward",
    "from",
    "hence",
    "here",
    "here's",
    "hereafter",
    "hereby",
    "herein",
    "hereupon",
    "how",
    "how's",
    "however",
    "indeed",
    "just",
    "less",
    "like",
    "likely",
    "literally",
    "main",
    "major",
    "many",
    "may",
    "maybe",
    "meanwhile",
    "might",
    "more",
    "moreover",
    "most",
    "mostly",
    "much",
    "must",
    "namely",
    "neither",
    "nevertheless",
    "nothing",
    "now",
    "often",
    "okay",
    "only",
    "opinion",
    "other",
    "others",
    "otherwise",
    "over",
    "own",
    "part",
    "particular",
    "particularly",
    "perhaps",
    "please",
    "probably",
    "quite",
    "rather",
    "really",
    "right",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "serious",
    "seriously",
    "simply",
    "sincere",
    "slightly",
    "so",
    "some",
    "somehow",
    "something",
    "sometime",
    "sometimes",
    "somewhat",
    "sort",
    "start",
    "still",
    "straight",
    "such",
    "than",
    "that",
    "that's",
    "then",
    "there",
    "there's",
    "thereafter",
    "thereby",
    "therefore",
    "therein",
    "thereupon",
    "these",
    "this",
    "those",
    "though",
    "through",
    "throughout",
    "thru",
    "together",
    "too",
    "totally",
    "up",
    "very",
    "virtually",
    "well",
    "what",
    "when",
    "when's",
    "whence",
    "whenever",
    "where",
    "where's",
    "whereafter",
    "whereas",
    "whereby",
    "wherein",
    "whereupon",
    "which",
    "while",
    "who",
    "whole",
    "why",
    "why's",
    "yet",
]
