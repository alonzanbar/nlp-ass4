import nltk
from nltk import word_tokenize, pos_tag, ne_chunk, Tree
import sys

from spc import read_sentences_from_annotated, nlp

LABELS = ['PERSON','LOC','GPE']

match = 0.0
tokens_len = 0.0
sentences=0.0
matched_sentences=0.0

for i, st in enumerate(read_sentences_from_annotated(sys.argv[1])):
    sentences +=1
    sent_title, sent_str = st
    nlpline = nlp(unicode(sent_str.strip()))
    a = nltk.parse.BLLIP.parse(sent_str)
    nltkents = [en for en in a if type(en)==Tree]
    if len(nltkents) != len(nlpline.ents):
        continue
    match_bool=True
    tokens_len += len(nltkents)
    ents=[]
    for i,ent in enumerate(nlpline.ents):
        if (nltkents[i].label() not in LABELS) and  (ent.label_ not in LABELS):
            continue
        if nltkents[i].label() == ent.label_:
            match+=1
            ents.append(ent)

print(match, (match/tokens_len))
print("matched sentences: {0} , sentences: {1}".format(matched_sentences,sentences))