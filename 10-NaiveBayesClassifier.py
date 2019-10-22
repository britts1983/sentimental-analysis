import nltk
from nltk.corpus import movie_reviews
import random
import pickle

documents = [(list(movie_reviews.words(fileid)),category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


classifier_f = open("support/naiveBayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

'''
This is the method of loading back a saved and trained dataset from a pickle file
'''

feature_set = [(find_features(rev), category) for (rev, category) in documents]
#print feature_set

train_set = feature_set[:1900]
test_set = feature_set[1900:]

#classifier = nltk.NaiveBayesClassifier.train(train_set)
print "The accuracy percentage is ", nltk.classify.accuracy(classifier,test_set) * 100
classifier.show_most_informative_features(20)

'''save_classifier = open("support/naiveBayes.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()'''

