import sys

import networkx
import networkx as nx
from spacy.matcher import Matcher

from spc import read_sentences_from_annotated, nlp
from parse_tree_utils import get_parse_tree_path

PERSON = 'PERSON'

LOCTATION_STRS = ['LOC','GPE']


def extract_features(pair,graph,doc):
    features={}
    en1,en2  = pair
    features['entity1-type'] = en1.label_
    features['entity1-head'] = en1.root
    features['entity2-type'] = en2.label_
    features['entity2-head'] = en2.root
    features['concatenatedtypes'] = en1.label_+en2.label_

    features['between-entities-words'] = "-".join([token.text for token in doc[en1.end+1:en2.start]])

    features['word-before-entity1']  = doc[en1.start-1] if en1.start>0 else 'None'
    features['word-after-entity2'] = doc[en2.end] if en2.end < len(doc) else 'None'
    path=None
    try:
        path  = nx.shortest_path(graph, source=doc[en1.start].i, target=doc[en2.end-1].i)
    except (networkx.exception.NetworkXNoPath,networkx.exception.NodeNotFound):
        pass
    if (path):
        dep_map, typed_dep_map = extract_dep_map(doc, path)
        #features['dep-path-typed'] = "-".join(typed_dep_map)
        features['dep-path'] = "-".join(dep_map)
    #features['base-syntatic-path'] = "-".join([token.tag_ for token in doc[en1.end+1:en2.start]])
    features['consitutient_path'] = "-".join(get_parse_tree_path(doc.text,en1.root.text,en2.root.text))

    return features


def extract_dep_map(doc, path):
    typed_dep_map = []
    dep_map = []
    for i, token_id in enumerate(path):
        token = doc[token_id]
        level = []
        dirc = 'up'
        typed_dep_map.append(token.text)
        if i > 0:
            next_token = doc[path[i - 1]]
            if token.head == next_token:
                dirc = 'down'
            level.append(dirc)
            level.append(token.dep_)
            dep_map.extend(level)
            typed_dep_map.extend(level)
    return dep_map, typed_dep_map


def build_graph(doc):
    edges = []
    for token in doc:
        # FYI https://spacy.io/docs/api/token
        for child in token.children:
            edges.append((token.i,
                          child.i))

    return nx.Graph(edges)


def format_token(token):
    return '{0}-{1}'.format(token.lower_, token.tag_)


def process_file(infile):
    samples=[]
    for sent_title, sent_str in read_sentences_from_annotated(infile):
        samples.extend(extract_sentence(sent_title,sent_str,True))
    return samples


def extract_sentence(sent_title,sent_str,supervised=False):
    doc = nlp(unicode(sent_str))
    if supervised:
        pairs= get_TRAIN_pairs(doc, sent_title)
    else:
        pairs = get_DEV_pairs(doc, sent_title)
    samples = extract_features_pairs(doc, pairs)
    return samples


def extract_features_pairs(doc, pairs):
    samples=[]
    tag_graph = build_graph(doc)
    for pair, label in pairs:
        print(pair)
        if not pair[0] or not pair[1] or not pair[0].root.text.strip() or not pair[1].root.text.strip():
            continue
        features_dict = extract_features(pair, tag_graph, doc)
        features = ["{0}={1}".format(k, v) for k, v in features_dict.items()]
        samples.append((label, features))
    return samples


def save_words(output_file,samples):
    with open(output_file, 'w') as outfile:
        for label,features in samples:
                 outfile.write(str(label) +'\t' + '\t'.join(features)+"\n")

def get_TRAIN_pairs(doc, sent_title):
    sent_id, person_ent, rel, loc_en = sent_title.split("\t")
    matcher = Matcher(nlp.vocab)
    # Get the ID of the 'EVENT' entity type. This is required to set an entity.
    PERSON_ID = nlp.vocab.strings[PERSON]
    LOCATION_ID = nlp.vocab.strings[LOCTATION_STRS[0]]

    matcher.add('person_ent', lambda macher,doc,i, matches: replace_en(doc,i,matches,PERSON_ID),
                [{'ORTH': k} for k in person_ent.split(" ")])

    matcher.add('location_ent', lambda macher,doc,i, matches: replace_en(doc,i,matches,LOCATION_ID),
                [{'ORTH': k} for k in loc_en.split(" ")])

    matcher(doc)
    def search_pairs():
        pairs = []
        neg = 0
        pos = 0
        for ne1 in doc.ents:
            for ne2 in doc.ents:
                if pos == 1 and neg > 1:
                    return pairs
                if (ne1.label_ == PERSON and ne2.label_ in LOCTATION_STRS):
                    if ne1.text == person_ent and ne2.text == loc_en:
                        pos = 1
                        pairs.append(([ne1, ne2], 1))
                    elif neg<2:
                        neg +=1
                        pairs.append(([ne1,ne2],0))
        print(pos,neg)
        return pairs
    pairs= search_pairs()

    return pairs

def get_DEV_pairs(doc, sent_title):
    pairs=[]
    sent_id, person_ent, rel, loc_en = sent_title.split("\t")
    for ne1 in doc.ents:
        for ne2 in doc.ents:
            if (ne1.label_ == PERSON and ne2.label_ in LOCTATION_STRS):
                if ne1.text == person_ent and ne2.text == loc_en:
                    pairs.append(([ne1,ne2],1))
                else:
                    pairs.append(([ne1, ne2],0))
    return pairs

def replace_en(doc, i, matches,label):
    match_id, start, end = matches[i]
    for en in doc.ents:
        if en.start==start and en.end==end:
            break
    doc.ents = [e for e in doc.ents if e !=en]
    doc.ents += ((label, start, end),)




if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    samples = process_file(infile)
    save_words(outfile,samples)