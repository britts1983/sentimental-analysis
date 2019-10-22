from nltk.corpus import  gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")

#print nltk.__file__	(to find the location)

tok = sent_tokenize(sample)

print tok[5:15]