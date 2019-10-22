from __future__ import division
import nltk
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode
import random
from nltk.classify import ClassifierI
import pickle


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = float(choice_votes / len(votes))
        return conf

documents = [(list(movie_reviews.words(fileid)), category)
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
        features[w] = w in words

    return features

featuresets = [(find_features(rev),category) for (rev, category) in documents]

train_set = featuresets[:1900]
test_set = featuresets[1900:]

'''
classifier = nltk.NaiveBayesClassifier.train(train_set)
print "The accuracy percentage is ", nltk.classify.accuracy(classifier,test_set) * 100
classifier.show_most_informative_features(20)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)
print "The accuracy of Multinomial NB classifier : ", nltk.classify.accuracy(MNB_classifier,test_set)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(train_set)
print "The accuracy of BernoulliNB_classifier: ", nltk.classify.accuracy(BernoulliNB_classifier,test_set)

Logistic_Regression_Classifier = SklearnClassifier(LogisticRegression())
Logistic_Regression_Classifier.train(train_set)
print "The accuracy of Logistic_Regression_Classifier : ", nltk.classify.accuracy(Logistic_Regression_Classifier,test_set)

SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(train_set)
print "The accuracy of SGD_classifier : ", nltk.classify.accuracy(SGD_classifier,test_set)

SVM_classifier= SklearnClassifier(SVC())
SVM_classifier.train(train_set)
print "The accuracy of SVM_classifier : ", nltk.classify.accuracy(SVM_classifier,test_set)

NuSVC_classifier= SklearnClassifier(NuSVC())
NuSVC_classifier.train(train_set)
print "The accuracy of NuSVC_classifier : ", nltk.classify.accuracy(NuSVC_classifier,test_set)

LinearSVC_classifier= SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)
print "The accuracy of LinearSVC_classifier : ", nltk.classify.accuracy(LinearSVC_classifier,test_set)

'''

classifier_f = open( "support/LinearSVC_classifier.pickle","rb")
LinearSVC_classifier = pickle.load(classifier_f)
classifier_f.close()


classifier_f = open( "support/NuSVC_classifier.pickle","rb")
NuSVC_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open( "support/BernoulliNB_classifier.pickle","rb")
BernoulliNB_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open( "support/classifier.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open( "support/SVM_classifier.pickle","rb")
SVM_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open( "support/SGD_classifier.pickle","rb")
SGD_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open( "support/Logistic_Regression_Classifier.pickle","rb")
Logistic_Regression_Classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open( "support/MNB_classifier.pickle","rb")
MNB_classifier = pickle.load(classifier_f)
classifier_f.close()


vote_classifier = VoteClassifier(classifier,
                                 BernoulliNB_classifier,
                                 MNB_classifier,
                                 SGD_classifier,
                                 LinearSVC_classifier,
                                 Logistic_Regression_Classifier,
                                 NuSVC_classifier)

'''print "Voted Classifier accuracy % : ",(nltk.classify.accuracy(vote_classifier,test_set)*100)'''

print "Classification:", vote_classifier.classify(test_set[0][0]), "Confidence % : ", vote_classifier.confidence(test_set[0][0])
print "Classification:", vote_classifier.classify(test_set[1][0]), "Confidence % : ", vote_classifier.confidence(test_set[1][0])
print "Classification:", vote_classifier.classify(test_set[2][0]), "Confidence % : ", vote_classifier.confidence(test_set[2][0])
print "Classification:", vote_classifier.classify(test_set[3][0]), "Confidence % : ", vote_classifier.confidence(test_set[3][0])
print "Classification:", vote_classifier.classify(test_set[4][0]), "Confidence % : ", vote_classifier.confidence(test_set[4][0])
