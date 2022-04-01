"""
GUI functionality.

Classes:
    TextEditWindow - GUI window handler

Functions:
    visual_edit - provide edit window for text
"""


import tkinter as tk
from tkinter import ttk


class TextEditWindow:
    # pylint: disable=too-few-public-methods
    """
    Present a GUI window where the user can edit text.

    Instance variables:
        text - the current text displayed (or edited)
    """

    def __init__(self, master, start_text, **kwargs):
        """
        Initialize and pack tkinter window.

        Arguments:
            master - a tkinter parent window
            start_text - the inital text to be displayed

        Optional Arguments
            label_text (str) - label above text editor
                               (default: 'Edit your text:')
            select_start (str) - where to start text selection
                                 (default: 1.0)
            select_end (str) - where to end text selection
                               (default: tk.END)
        """
        kwargs.update(select_start=kwargs.get("select_start", "1.0"))
        kwargs.update(select_end=kwargs.get("select_end", tk.END))
        kwargs.update(label_text=kwargs.get("label_text", "Edit your text:"))

        self.text = start_text
        self._master = master

        self._label = ttk.Label(self._master, text=kwargs["label_text"])
        self._text_box = tk.Text(self._master, wrap=tk.WORD)
        self._text_box.insert(tk.END, self.text)
        self._text_box.bind("<Return>", self._go)
        self._text_box.bind("<KP_Enter>", self._go)
        self._text_box.tag_add(
            tk.SEL, kwargs["select_start"], kwargs["select_end"]
        )
        self._text_box.mark_set(tk.INSERT, kwargs["select_end"])
        self._text_box.focus()
        self._button = ttk.Button(self._master, text="Go", command=self._go)

        self._label.pack()
        self._text_box.pack()
        self._button.pack()

    def _go(self, event=None):
        # pylint: disable=unused-argument
        self.text = self._text_box.get("1.0", tk.END)[:-1]
        self._master.destroy()


def visual_edit(tokens, indices=None, thestyle="clam", **kwargs):
    """
    Generate a GUI box where the user can edit text.

    Arguments:
        tokens - iterable of tokens comprising display text

    Optional Arguments:
        indices - list of ints to be selected in edit window (default all)
        thestyle - ttk style to use for the text box (default 'clam')

    Additional arguments are passed to `TextEditWindow`.

    Returns:
        text - the edited text once the user closes the window
    """
    master = tk.Tk()
    style = ttk.Style()
    style.theme_use(thestyle)
    pointer_x, pointer_y = master.winfo_pointerxy()
    master.geometry(f"+{pointer_x}+{pointer_y}")

    # Calculate selection area.
    if indices is None:
        indices = [0, len("".join(tokens))]
    sel_start = 0
    sel_end = len("".join(tokens))
    for i, tok in enumerate(tokens):
        if i < indices[0]:
            sel_start += len(tok)
        elif i > indices[-1]:
            sel_end -= len(tok)

    window = TextEditWindow(
        master,
        "".join(tokens),
        select_start=f"1.{sel_start}",
        select_end=f"1.{sel_end}",
        **kwargs,
    )
    master.mainloop()
    return window.text
