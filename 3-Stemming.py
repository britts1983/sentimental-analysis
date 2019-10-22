from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_sentence = "Every python code must be pythoned properly by a pythoner. Every pythoning pythoner has pythoned poorly at least once. He must solve them pythonly ahead."
words = word_tokenize(example_sentence)

for word in words:
    print ps.stem(word)