import nltk
from nltk.corpus import movie_reviews
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC

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

'''
classifier_f = open("support/naiveBayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

This is the method of loading back a saved and trained dataset from a pickle file
'''

feature_set = [(find_features(rev), category) for (rev, category) in documents]

train_set = feature_set[:1900]
test_set = feature_set[1900:]

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


#save the classifiers into pickle


save_classifier = open("support/LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

save_classifier = open("support/MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

save_classifier = open("support/NuSVC_classifier.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

save_classifier = open("support/BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

save_classifier = open("support/classifier.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

save_classifier = open("support/SVM_classifier.pickle","wb")
pickle.dump(SVM_classifier, save_classifier)
save_classifier.close()

save_classifier = open("support/SGD_classifier.pickle","wb")
pickle.dump(SGD_classifier, save_classifier)
save_classifier.close()

save_classifier = open("support/Logistic_Regression_Classifier.pickle","wb")
pickle.dump(Logistic_Regression_Classifier, save_classifier)
save_classifier.close()
