# verify_installations.py

import sys
import spacy
import gensim
import scipy
import pandas
import numpy
import nltk
import matplotlib
import openpyxl
import pydantic
import typing_extensions
import praw

print("Python version:", sys.version)
print("praw version:", praw.__version__)
print("spaCy version:", spacy.__version__)
print("gensim version:", gensim.__version__)
print("scipy version:", scipy.__version__)
print("pandas version:", pandas.__version__)
print("numpy version:", numpy.__version__)
print("nltk version:", nltk.__version__)
print("matplotlib version:", matplotlib.__version__)
print("openpyxl version:", openpyxl.__version__)

# Handle potential absence of __version__ in pydantic
try:
    pydantic_version = pydantic.__version__
except AttributeError:
    pydantic_version = pydantic.VERSION

print("pydantic version:", pydantic_version)
# print("typing_extensions version:", typing_extensions.version)
