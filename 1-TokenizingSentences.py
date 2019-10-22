from nltk.tokenize import word_tokenize, sent_tokenize

# tokenizing - word  tokenizers and sentence tokenizer
# lexicon and corporas
# corporas - body of text. ex: medical journals, presidential speeches, anything in english language
# lexicons - words and their meanings

sentence = "This is the sentence. That is being subjected to natural language processing!"

w_tokens = word_tokenize(sentence)
# this is basically stored as a list

# here are few ways of printing out the tokens
print word_tokenize(sentence)

for token in w_tokens:
    print token

s_tokens = sent_tokenize(sentence)
for token in s_tokens:
    print token
