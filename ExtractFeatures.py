import sys

import networkx as nx
from spacy.matcher import Matcher

from spc import read_sentences_from_annotated, nlp

PERSON = 'PERSON'

LOCTATION_STRS = ['LOC','GPE']


def extract_features(pair,graph,doc):
    features={}
    en1,en2  = pair
    features['entity1-type'] = en1.label
    features['entity1-head'] = en1.root
    features['entity2-type'] = en2.label
    features['entity2-head'] = en2.root
    features['concatenatedtypes'] = en1.label+en2.label

    features['between-entities-words'] = "-".join([token.text for token in doc[en1.end+1:en2.start]])

    features['word-before-entity1']  = doc[en1.start-1] if en1.start>0 else 'None'
    features['word-after-entity2'] = doc[en2.end] if en2.end < len(doc) else 'None'

    path  = nx.shortest_path(graph, source=format_token(doc[en1.start]), target=format_token(doc[en2.end-1]))
    base_syn_path =[]
    for str in path:
        tag = str.split("-")[1]
        base_syn_path.append(tag)
    features['base-syntatic-path'] = "-".join(base_syn_path)


    return features


def build_graph(doc):
    edges = []
    for token in doc:
        # FYI https://spacy.io/docs/api/token
        for child in token.children:
            edges.append((format_token(token),
                          format_token(child)))

    return nx.Graph(edges)


def format_token(token):
    return '{0}-{1}'.format(token.lower_, token.tag_)


def process_file(infile):

    feature_set=set()
    samples=[]
    for sent_title, sent_str in read_sentences_from_annotated(infile):

        pairs,doc= extract_pairs(sent_str, sent_title)
        graph = build_graph(doc)
        for pair,label in pairs:
            features_dict =  extract_features(pair,graph,doc)
            features = ["{0}-{1}".format(k,v) for k,v in features_dict.items()]
            samples.append((features,label))
            feature_set|=set(features)

    print(len(samples))


def extract_pairs(sent_str, sent_title):
    sent_id, person_ent, rel, loc_en = sent_title.split("\t")
    pairs = []

    matcher = Matcher(nlp.vocab)

    # Get the ID of the 'EVENT' entity type. This is required to set an entity.
    PERSON_ID = nlp.vocab.strings[PERSON]
    LOCATION_ID = nlp.vocab.strings[LOCTATION_STRS[0]]

    def replace_en(doc, i, matches,label):
        match_id, start, end = matches[i]

        for en in doc.ents:
            if en.start==start and en.end==end:
                break
        doc.ents = [e for e in doc.ents if e !=en]
        doc.ents += ((label, start, end),)

    matcher.add('person_ent', lambda macher,doc,i, matches: replace_en(doc,i,matches,PERSON_ID),
                [{'ORTH': k} for k in person_ent.split(" ")])

    matcher.add('location_ent', lambda macher,doc,i, matches: replace_en(doc,i,matches,LOCATION_ID),
                [{'ORTH': k} for k in loc_en.split(" ")])

    doc = nlp(unicode(sent_str))
    matcher(doc)
    cnt = 0
    fnd=1
    for ne1 in doc.ents:
        for ne2 in doc.ents:
            if (ne1.label_ == PERSON and ne2.label_ in LOCTATION_STRS):
                if ne1.text == person_ent and ne2.text == loc_en:
                    fnd = 1
                    cnt+=1
                pairs.append(([ne1,ne2],fnd))



    return pairs, doc






if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    process_file(infile)