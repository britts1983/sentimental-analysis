from __future__ import division
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import nltk
from nltk.tokenize import word_tokenize


# characteristics of Samarth Deyagond
sam_words = open('Samarth_words','r').read().lower()
sam_words = sam_words.decode('utf-8')
sam_words = word_tokenize(sam_words)
# print sam_words
freq_sam = nltk.FreqDist(sam_words)
sam_words_prob = {}
for key in freq_sam:
    sam_words_prob[key] = float(freq_sam[key]/len(freq_sam))

# characteristics of Rachana Kalachetty
dums_words = open('Panda_words','r').read().decode('utf-8').lower()
dums_words = word_tokenize(dums_words)
freq_dums = nltk.FreqDist(dums_words)
dum_words_prob = {}
for key in freq_dums:
    dum_words_prob[key] = float(freq_dums[key]/len(freq_dums))

sam_word_set_count = len(set(sam_words))
dums_word_set_count = len(set(dums_words))

print sam_word_set_count
print dums_word_set_count

message = raw_input("Enter the message whose sender is to be predicted :").strip().lower()
message_tokens = word_tokenize(message)

sam = 0.5
dummu = 0.5

for token in message_tokens:
    if token in sam_words and token in dums_words:
        sam = sam*sam_words_prob[token]
        dummu = dummu * dum_words_prob[token]

    elif token not in dums_words and token in sam_words:
        dummu = dummu*min(dum_words_prob.values())
        sam = sam * sam_words_prob[token]

    elif token in dums_words and token not in sam_words:
        sam = sam * min(sam_words_prob.values())
        dummu = dummu * dum_words_prob[token]

    # else:
        # print "There is a 50-50 chance that either Teddy Winters / Panda has made this sentence! Sorry! But may be"

# normalization
sam_normal = float(sam/sam+dummu)
dum_normal = float(dummu/sam+dummu)

if sam_normal > dum_normal:
    print "\nSamarth is more likely to have made that statement!\n"
else:
    print "\nRachana is more likely to have made that statement!\n"

