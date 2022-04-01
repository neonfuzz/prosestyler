"""
Miscelaneous functionality.

Functions:
    fromx_to_id - get token indices from character indices
    pretty_print_prose - print long strings of text nicely
    now_checking_banner - pretty banner
    print_rows - print items in a neat, ordered list
"""

from math import ceil
from pprint import pformat
import re


def fromx_to_id(fromx, tox, tokens):
    """
    Given character indices, return token indices.

    Arguments:
        fromx - start index of character string
        tox - end index of character string
        tokens - tokenized sentence

    Returns:
        list of indices corresponding to selected tokens
    """
    # pylint: disable=invalid-name
    i = 0
    x = 0
    ids = []
    while x < tox:
        if x >= fromx:
            ids += [i]
        x += len(tokens[i])
        i += 1
    return ids


def pretty_print_prose(string: str, *args, **kwargs):
    """Print a string to the screen with proper linebreaks."""
    out = pformat(string, *args, **kwargs)
    out = re.sub(r'["\']\n ["\']', "\n", out)
    # Strip parenthesis and quotes from beginning and end.
    if out[1] == "(":
        out = out[2:-2]
    else:
        out = out[1:-1]
    print(out)


def now_checking_banner(check):
    """Print a pretty banner on the output."""
    mystr = f"---  Now Checking {str(check).title()}!  ---"
    dashstr = "-" * len(mystr)
    print("\n\n")
    print(dashstr)
    print(mystr)
    print(dashstr)
    pretty_print_prose(check.description)
    print(dashstr)


def print_rows(lis, max_rows=21, cols=3, item_width=18):
    """Given a list of items, print them, numbered, in columns and rows."""
    if len(lis) == 1 and lis[0] == "":
        print(" (1) <delete>")
        return

    max_items = max_rows * cols
    if len(lis) > max_items:
        lis = lis[:max_items]
    if len(lis) < 2 * cols:
        cols = 1

    # Make a string template holding each column.
    mystr = f"{{: >4}} {{: <{(item_width) * cols}}}"
    nrows = ceil(len(lis) / cols)
    rows = [[]] * nrows
    row_ind = 0
    # Order stuff to read down each column.
    # (rather than across each row).
    for i, j in enumerate(lis):
        rows[row_ind] = rows[row_ind] + [f"({(i+1)})", j]
        row_ind = (row_ind + 1) % nrows
    while row_ind != 0:
        rows[row_ind] = rows[row_ind] + ["", ""]
        row_ind = (row_ind + 1) % nrows
    for row in rows:
        print(mystr.format(*row))
