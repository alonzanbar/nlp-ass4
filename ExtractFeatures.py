import re
import sys

import itertools
import networkx
import networkx as nx
from spacy.matcher import Matcher

from spc import read_sentences_from_annotated, nlp, read_annotated_line
from parse_tree_utils import get_parse_tree_path

PERSON = 'PERSON'

LOCTATION_STRS = ['LOC','GPE','NORP']

def extract_features(pair,graph,doc):
    features={}
    en1,en2  = pair
    features['entity1-type'] = en1.label_
    features['entity1-head'] = en1.root
    features['entity2-type'] = en2.label_
    features['entity2-head'] = en2.root
    features['concatenatedtypes'] = en1.label_+en2.label_
    features['between-entities-word']=[]
    if en1.start<en2.start:
        start = en1.end+1
        end = en2.start
    else:
        start = en2.end + 1
        end = en1.start
    words_set=set([t.text for t in doc[start:end]])
    for i,w  in enumerate(words_set):
        features['between-entities-word'].append(w)
    #features['between-entities-words'] = "-".join([token.text for token in doc[en1.end+1:en2.start]])
    features['word-before-entity1']  = doc[en1.start-1].text if en1.start>0 else 'None'
    features['word-after-entity2'] = doc[en2.end].text if en2.end < len(doc) else 'None'
    path=None
    try:
        path  = nx.shortest_path(graph, source=doc[en1.start].i, target=doc[en2.end-1].i)
    except (networkx.exception.NetworkXNoPath,networkx.exception.NodeNotFound):
        pass
    if (path):
        dep_map, typed_dep_map = extract_dep_map(doc, path)
        #features['dep-path-typed'] = "-".join(typed_dep_map)
        features['dep-path'] = "-".join(dep_map)
        features['dis_ent_distance'] = len(path)
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
        if sent_title.split("\t")[2]=="Live_In":
            samples.extend(extract_sentence(sent_title.strip(),sent_str.strip(),True))
    return samples


def extract_sentence(sent_title,sent_str,supervised=False):

    doc = nlp(unicode(sent_str))
    if supervised:
        pairs= get_TRAIN_pairs(doc, sent_title)
    else:
        pairs = get_DEV_pairs(doc, sent_title)
    if pairs:
        samples = extract_features_pairs(doc, pairs)
        return samples
    else:
        return []


def extract_features_pairs(doc, pairs):
    samples=[]
    tag_graph = build_graph(doc)
    for pair, label in pairs:
        print(pair)
        if not pair[0] or not pair[1] or not pair[0].root.text.strip() or not pair[1].root.text.strip():
            continue
        features_dict = extract_features(pair, tag_graph, doc)
        features=[]
        for k, v in features_dict.items():
            if isinstance(v,list):
                for it in v:
                    features.append("{0}={1}".format(k, it))
            else:
                features.append("{0}={1}".format(k, v))
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

    replace_en(doc,person_ent,PERSON_ID)
    replace_en(doc, loc_en, LOCATION_ID)
    def search_pairs():
        pairs = []
        neg = 0
        pos = 0
        per_ents=[]
        loc_ents=[]
        for en in doc.ents:
            if en.label_ == PERSON or en.text == person_ent: per_ents.append(en)
            if en.label_ in LOCTATION_STRS or en.text==loc_en : loc_ents.append(en)
        all_pairs = list(itertools.product(per_ents, loc_ents))

        for ne1,ne2 in all_pairs:
            if pos == 1 and neg > 1:
                return pairs
            if (ne1.text == person_ent and ne2.text == loc_en) :
                pos = 1
                pairs.append(([ne1, ne2], 1))
            elif neg<2:
                neg +=1
                pairs.append(([ne1,ne2],0))
        print(pos,neg)
        if pos==0:
            pass
        return pairs
    pairs= search_pairs()

    return pairs

def get_DEV_pairs(doc, sent_title):
    pairs=[]
    sent_id, person_ent, rel, loc_en = sent_title.split("\t")
    all_pairs = list(itertools.product(doc.ents, doc.ents))
    for ne1, ne2 in all_pairs:
        if (ne1.label_ in [PERSON, 'ORG'] and ne2.label_ in LOCTATION_STRS):
            if (ne1.text == person_ent and ne2.text == loc_en):
                pairs.append(([ne1, ne2], 1))
            else:
                pairs.append(([ne1, ne2], 0))
    return pairs


def replace_en(doc,str,label):
    gold_words  = re.split(' |-',str)
    sent_words = [w.text for w in doc]
    try:
        start, end  = sent_words.index(gold_words[0]),sent_words.index(gold_words[len(gold_words)-1])+1
    except ValueError:
        return
    span = None
    for en in doc.ents:
        if en.start==start or en.end==end:
            span=en
            break
    if span:
        doc.ents = [e for e in doc.ents if e !=span]
        label = span.label
    doc.ents += ((label, start, end),)




if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    samples = process_file(infile)
    save_words(outfile,samples)