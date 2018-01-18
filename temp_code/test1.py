import nltk
from nltk.corpus import reuters
import re

from spc import read_sentences_from_annotated

for title, sent in read_sentences_from_annotated("data/TRAIN.annotations.min"):
    _,person,_,location = title.strip().split('\t')
    print ("person :{}, location {}".format(person,location))
    a  = [" ".join(s) for s in reuters.sents() if person in s]
    print (a)
    print("\n")



#a = [s for s in brown.sents()]
#print (a)


