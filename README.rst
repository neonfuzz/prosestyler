ProseStyler
=============

An interactive grammar and style tool.

Usage
-----

::

    prosestyler_app [-h] [-o outfile] [-d dictionary] [-l check_name [check_name ...]] [--all]
                    [--spelling | --no-spelling] [--grammar | --no-grammar]
                    [--cliches | --no-cliches] [--passive | --no-passive]
                    [--nominalizations | --no-nominalizations] [--filler | --no-filler]
                    [--adverbs | --no-adverbs] [--noun_phrases | --no-noun_phrases]
                    [--homophones | --no-homophones] [--weak | --no-weak] [--lint | --no-lint]
                    file

positional arguments:
^^^^^^^^^^^^^^^^^^^^^
    **file**
        The file to analyze.

optional arguments:
^^^^^^^^^^^^^^^^^^^
  -h, --help            show this help message and exit
  -o <outfile>          Name of output file (default: <filename>_out_<datetime>)
  -d <dictionary>       Which dictionary to use (default: en_US)
  -l <check_name [check_name ...]>
                        List of checks to use (overrides all other options, except --all).
  --all                 Use ALL checks (overrides all other options, including -l).
  --spelling, --no-spelling
                        Run a spellcheck (default: True)
  --grammar, --no-grammar
                        Run a grammar check (default: True)
  --cliches, --no-cliches
                        Check for cliches (default: True)
  --passive, --no-passive
                        Check for passive voice (default: True)
  --nominalizations, --no-nominalizations
                        Check for nominalizations (default: True)
  --filler, --no-filler
                        Check for filler words (default: True)
  --adverbs, --no-adverbs
                        Check for adverbs (default: True)
  --noun_phrases, --no-noun_phrases
                        Check for adverbs (default: True)
  --homophones, --no-homophones
                        Show every detected homophone (default: False)
  --weak, --no-weak     Check for weak words (default: False)
  --lint, --no-lint     Run Proselint on the text (default: False)

Installation
------------

.. code-block:: console

    python3 -m spacy download en_core_web_sm
    pip install .
    export PATH=$PATH:$(pwd)

Requirements
^^^^^^^^^^^^

* hspell
* hunspell
* hunspell-en_US
* libvoikko
* nuspell
* tk

Authors
-------

`ProseStyler` was written by `neonfuzz`.

Attributions
------------

.. image:: https://travis-ci.org/kragniz/cookiecutter-pypackage-minimal.png
   :target: https://travis-ci.org/kragniz/cookiecutter-pypackage-minimal
   :alt: Latest Travis CI build status
