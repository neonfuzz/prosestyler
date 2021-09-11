import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    """Read in README and parse nicely for `setup`."""
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u'')
    with io.open(filename, mode='r', encoding='utf-8') as infile:
        return re.sub(
            text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'),
            infile.read())


setup(
    name="prosestyler",
    version="0.1.0",
    license='MIT',

    author="neonfuzz",

    description="An interactive grammar and style tool.",
    long_description=read("README.rst"),

    packages=find_packages(exclude=('tests',)),
    package_data={'': ['*pkl.gz', '*.txt']},
    include_package_data=True,

    install_requires=[
        'argparse',
        'language-tool-python',
        'numpy',
        'pandas',
        'proselint',
        'pyenchant',
        'spacy',
        ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
)
