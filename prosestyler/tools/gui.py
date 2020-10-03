

"""
GUI functionality.

Classes:
    TextEditWindow - GUI window handler

Functions:
    visual_edit - provide edit window for text
"""


import tkinter as tk
from tkinter import ttk


class TextEditWindow():
    # pylint: disable=too-few-public-methods
    """
    Present a GUI window where the user can edit text.

    Instance variables:
        text - the current text displayed (or edited)
    """

    def __init__(self, master, start_text, label_text='Edit your text:'):
        """
        Initialize and pack tkinter window.

        Arguments:
            master - a tkinter parent window
            start_text - the inital text to be displayed

        Optional Arguments
            label_text - label above text editor
                         (default: 'Edit your text:')
        """
        self.text = start_text
        self._master = master

        self._label = ttk.Label(self._master, text=label_text)
        self._text_box = tk.Text(self._master, wrap=tk.WORD)
        self._text_box.insert(tk.END, self.text)
        self._text_box.bind('<Return>', self._go)
        self._text_box.bind('<KP_Enter>', self._go)
        self._text_box.tag_add(tk.SEL, '1.0', tk.END)
        self._text_box.focus()
        self._button = ttk.Button(self._master, text='Go', command=self._go)

        self._label.pack()
        self._text_box.pack()
        self._button.pack()

    def _go(self, event=None):
        # pylint: disable=unused-argument
        self.text = self._text_box.get('1.0', tk.END)[:-1]
        self._master.destroy()


def visual_edit(text, thestyle='clam'):
    """
    Generate a GUI box where the user can edit text.

    Arguments:
        text - the initial text to display

    Optional Arguments:
        thestyle - ttk style to use for the text box (default 'clam')

    Returns:
        text - the edited text once the user closes the window
    """
    master = tk.Tk()
    style = ttk.Style()
    style.theme_use(thestyle)
    pointer_x, pointer_y = master.winfo_pointerxy()
    master.geometry('+%d+%d' % (pointer_x, pointer_y))

    window = TextEditWindow(master, text)
    master.mainloop()
    return window.text
