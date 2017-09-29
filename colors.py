

class Color(object):
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'

    def print(sent, start=0, end=None, style=None):
        if end is None:
            end = len(sent)
        if style is None:
            style = Color.underline
        print(sent[:start+1]
              + style + sent[start+1:end+1]
              + Color.end + sent[end+1:])
