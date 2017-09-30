

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

    def tokenprint(tokens, indices, style=None):
        if style is None:
            style = Color.underline
        mystr = ''.join(tokens[:indices[0]])
        for i, index in enumerate(indices):
            if i == len(indices)-1:
                continue
            mystr += style + tokens[index] + Color.end
            mystr += ''.join(tokens[index+1:indices[i+1]])
        mystr += style + tokens[indices[-1]] + Color.end
        mystr += ''.join(tokens[indices[-1]+1:])
        print(mystr)
