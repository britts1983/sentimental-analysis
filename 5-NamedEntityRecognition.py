import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sentence_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sentence_tokenizer.tokenize(sample_text)

for token in tokenized:
    words = word_tokenize(token)
    tagged = nltk.pos_tag(words)
    namedEnt = nltk.ne_chunk(tagged,binary=True)
    namedEnt.draw()
    print tagged