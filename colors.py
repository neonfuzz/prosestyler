

"""
Print in different colors/styles to the command line.
"""


PURPLE = '\033[95m'
CYAN = '\033[96m'
DARKCYAN = '\033[36m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
BORING = '\033[0m'


def tokenprint(tokens, indices=None, style=None):
    """
    Print sentences from token form with given indices underlined.
    """

    if indices is None:
        print(''.join(tokens))
        return

    if style is None:
        style = UNDERLINE

    mystr = ''.join(tokens[:indices[0]])
    for i, index in enumerate(indices):
        if i == len(indices)-1:
            continue
        mystr += style + tokens[index] + BORING
        mystr += ''.join(tokens[index+1:indices[i+1]])
    mystr += style + tokens[indices[-1]] + BORING
    mystr += ''.join(tokens[indices[-1]+1:])
    print(mystr)
