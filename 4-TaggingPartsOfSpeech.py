from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import state_union
import nltk

text_train = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_text_tokenizer = PunktSentenceTokenizer(text_train)
# tokenized = custom_text_tokenizer.tokenize(sample_text) # sentence tokenizer
tokenized = custom_text_tokenizer.tokenize("Poornima is an unclear Gemini!")

f = open('support/pos_tags.txt','r')
line = f.read()
lines = line.split('\n')

pos_dict = {}
for line in lines:
    key_val = line.split('\t')
    pos_dict[key_val[0]] = key_val[1]

POS = []

def process_content():
    try:
        for i in tokenized:
            words = word_tokenize(i)
            tagged = list(nltk.pos_tag(words))

            for j in range(len(tagged)):
                tagged[j] = list(tagged[j])
                tagged[j][1] = pos_dict[tagged[j][1]]
                print tagged[j]

    except Exception as e:
        print str(e)

process_content()

'''

POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
'''