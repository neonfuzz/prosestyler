

"""
Provide a base class for checks.

Classes:
    BaseCheck - said base class
"""


from ..sentence import Sentence, gen_tokens
from ..tools import colors
from ..tools.gui import visual_edit
from ..tools.helper_functions import now_checking_banner, print_rows


class BaseCheck():
    """
    Checker base class; meant to be sub-classed.

    Define the following methods when sub-classed:
        __repr__
        _check_sent

    Sub-classed docstrings should follow the following format:

        <1-line description of checker>

        Arguments:
            text (Text) - the text to check

        Iterates over each Sentence and applies <describe check>.
        Text is saved and cleaned after each iteration.
    """

    def __repr__(self):
        """One- or two-word description of the check."""
        return ''

    def _suggest_toks(self, tokens, indices, suggestions, message,
                      can_replace_sent=False):
        """
        Ask the user to provide input on errors or style suggestions.

        Arguments:
            tokens (list) - tokens of the sentence in question
            indices (list) - indices of tokens to be replaced
            suggestions (list) - possible suggestions
            message (str) - error message, if any

        Optional arguments:
            can_replace_sent (bool) - should the user have the explicit option
                of replacing the entire sentence? default `False`
        """
        # Print the sentence with the desired token underlined.
        print()
        inds = range(indices[0], indices[-1]+1)
        colors.tokenprint(tokens, inds)
        phrase = ''.join([tokens[i] for i in inds])
        if message is not None:
            print()
            print('REASON:', message)
        print('Possible suggestions for "%s":' % phrase)

        # Print list of suggestions, as well as custom options.
        print_rows(suggestions)
        print(' (0) Leave be.')
        if can_replace_sent is True:
            print('(ss) Edit entire sentence.')
        print(' (?) Input your own.')

        # Get user input.
        # If a number, replace with suggestion.
        # If 0, return sentence as-is.
        # If 'ss', ask user to replace entire sentence.
        # Else: return user-input.
        user_input = input('Your choice: ')
        try:
            user_choice = int(user_input)
            if len(suggestions) >= user_choice > 0:
                ans = suggestions[user_choice-1]
                # Replace everything between the first and last tokens.
                tokens = tokens[:indices[0]] + [ans] + tokens[indices[-1]+1:]
            elif user_choice != 0:
                print('\n\n-------------\nINVALID VALUE\n-------------')
                tokens = self._suggest_toks(
                    tokens, indices, suggestions, can_replace_sent)
        except ValueError:
            if user_input == 'ss':
                sent = visual_edit(''.join(tokens))
                tokens = gen_tokens(sent)
            else:
                ans = user_input
                tokens = tokens[:indices[0]] + [ans] + tokens[indices[-1]+1:]
        return tokens

    def _check_sent(self, sentence, ignore_list=None):
        """
        Check a single loop (on one sentence).

        Arguments:
            sentence (Sentence) - the sentence to check

        Optional Arguments:
            ignore_list (list) - error tuples to ignore
                (default: empty)

        Returns:
            errors (list of 2-tuples) - each tuple contains
                toks - the tokens
                ids - the token (not word) indices of the sentence
            suggests (list of lists) -
                inner lists are suggestions to fix the error
            ignore_list (list of 2-tuples) - errors to ignore
            messages (list of strings) - error message to display, if any
        """
        # TO SUB-CLASS.
        errors = []
        suggests = []
        if ignore_list is None:
            ignore_list = []
        messages = []

        return errors, suggests, ignore_list, messages

    def __call__(self, text):
        """
        Check an entire `Text` for errors.

        Arguments:
            text (Text) - the text to check

        Iterates over each Sentence and applies `_check_sent`.
        Text is saved and cleaned after each iteration.
        """
        now_checking_banner(str(self))
        for i, sent in enumerate(text):
            errors, suggests, ignore_list, messages = self._check_sent(sent)
            tmp_sent = sent
            while errors:
                err = errors[0]
                new_tokens = self._suggest_toks(
                    tmp_sent.tokens, err[1], suggests[0], messages[0], True)
                if new_tokens == tmp_sent.tokens:
                    ignore_list += [err]
                    errors = errors[1:]
                    suggests = suggests[1:]
                    messages = messages[1:]
                else:
                    tmp_sent = Sentence(''.join(new_tokens))
                    errors, suggests, ignore_list, messages = self._check_sent(
                        tmp_sent, ignore_list)
            text[i] = tmp_sent
            text.clean()
            text.save()
