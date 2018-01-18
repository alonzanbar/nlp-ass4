import sys

from nltk.tag.stanford import CoreNLPNERTagger

from spc import read_sentences_from_annotated, nlp
from pycorenlp import StanfordCoreNLP
import networkx as nx

from nltk.tag import StanfordNERTagger

class StanEn:
    def __init__(self,ids,words,tag):
        self.ids = ids
        self.words=words
        self.tag  = tag
        self.text = " ".join(words)
        self.root = words[0]
    def __repr__(self):
        return self.text+", "+self.tag



def getStanfordEnts(sent_str):
    ents = []
    LABELS = ['PERSON', 'LOCATION', 'GPE']
    prev_tag = ""
    words_chunk = []
    id_chunk=[]
    tagger = CoreNLPNERTagger(url='http://localhost:9000')
    parsed = tagger.parse(sent_str.split())
    nltk
    pent_tuples = tagger.tag(sent_str.split())
    for i,(word, tag) in enumerate(pent_tuples):
        if tag != prev_tag and prev_tag != "" and tag!="":
            conv_tag  = prev_tag if prev_tag!='LOCATION' else 'LOC'
            ents.append(StanEn(id_chunk,words_chunk, conv_tag))
            words_chunk = []
            id_chunk=[]
        words_chunk.append(word)
        id_chunk.append(i)
        prev_tag = tag

    conv_tag = tag if tag != 'LOCATION' else 'LOC'
    ents.append(StanEn(id_chunk, words_chunk, conv_tag))

    ents = [en for en in ents if en.tag != 'O']
    return  ents


if __name__=="__main__":
    sentences = 0
    sent_str = "An enraged Nikita Khrushchev instructed Soviet ships to ignore President Kennedy 's naval blockade during the Cuban missile crisis , but the order was reversed just hours before an inevitable confrontation , according to a new book ."
    ents  = getStanfordEnts(sent_str)
    print ents
    pass

