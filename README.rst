prosestyler
=============

An interactive grammar and style tool.

Usage
-----

usage: prosestyler_app [-h] [-o outfile] [-d dictionary] [--spelling]
                       [--grammar] [--homophones] [--cliches] [--passive]
                       [--nominalizations] [--weak] [--filler] [--adverbs]
                       [--lint] [--frequent] [--vis-length]
                       file

Perform a deep grammar and style check.

positional arguments:
  file                  The file to be analyzed.

optional arguments:
  -h, --help            show this help message and exit
  -o outfile            Name of output file (default:
                        <filename>_out_<datetime>)
  -d dictionary         Which dictionary to use (default: en_US)
  --spelling, --no-spelling
                        Run a spellcheck (default: True)
  --grammar, --no-grammar
                        Run a grammar check (default: True)
  --homophones, --no-homophones
                        Show every detected homophone (default: False)
  --cliches, --no-cliches
                        Check for cliches (default: True)
  --passive, --no-passive
                        Check for passive voice (default: True)
  --nominalizations, --no-nominalizations
                        Check for nominalizations (default: True)
  --weak, --no-weak     Check for weak words (default: False)
  --filler, --no-filler
                        Check for filler words (default: True)
  --adverbs, --no-adverbs
                        Check for adverbs (default: True)
  --lint, --no-lint     Run Proselint on the text (default: False)
  --frequent, --no-frequent
                        Show the most frequently-used words (default: False)
  --vis-length, --no-vis-length
                        Visualize sentence lengths (default: False)

Installation
------------

`python3 -m spacy download en_core_web_sm`
`pip install .`
`export PATH=$PATH:$(pwd)`

Requirements
^^^^^^^^^^^^

hspell
hunspell
hunspell-en_US
libvoikko
nuspell
tk

Authors
-------

`prosestyler` was written by `neonfuzz`.

Attributions
------------

.. image:: https://travis-ci.org/kragniz/cookiecutter-pypackage-minimal.png
   :target: https://travis-ci.org/kragniz/cookiecutter-pypackage-minimal
   :alt: Latest Travis CI build status
