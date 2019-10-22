from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "This is a sentence that is being employed to find the stop words!"
stop_words = set(stopwords.words("english"))

print stop_words

filtered_sentence = []
words = word_tokenize(example_sentence)

for word in words:
    if word not in stop_words:
        filtered_sentence.append(word)

print filtered_sentence

# one liner for the above logic goes like this

filtered_sentence = [word for word in words if not word in stop_words]
print filtered_sentence
