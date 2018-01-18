import logging
import re
import sys
from spacy.matcher import Matcher

from FeatureBuilder import extract_features
from temp_code.spc import read_sentences_from_annotated, nlp
from parse_tree_utils import get_parse_tree_path
from utils import PERSON, LOCTATION_STRS

NUMBER_NRG = 5

def format_token(token):
    return '{0}-{1}'.format(token.lower_, token.tag_)


def process_file(infile):
    samples=[]
    for sent_title, sent_str in read_sentences_from_annotated(infile):
        doc = nlp(unicode(sent_str.strip()))
        pairs = get_TRAIN_pairs(doc, sent_title)
        pair_samples = extract_features_pairs(doc, pairs)
        samples.extend(pair_samples)
    return samples



def extract_features_pairs(doc, pairs):
    samples=[]
    for pair, label in pairs:
        print(pair)
        if not pair[0] or not pair[1] or not pair[0].root.text.strip() or not pair[1].root.text.strip():
            continue
        features_dict = extract_features(pair, doc)
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

def get_TRAIN_pairs(doc,sent_title):
    sent_id, person_ent, rel, loc_en = sent_title.split("\t")
    replace_en(doc,person_ent,[PERSON])
    replace_en(doc, loc_en, LOCTATION_STRS)
    return get_pairs_supervised(doc,person_ent,loc_en)


def get_DEV_pairs(doc,sent_title):
    sent_id, person_ent, rel, loc_en = sent_title.split("\t")
    all_pairs = get_pairs_supervised(doc,person_ent,loc_en)
    return  all_pairs

def get_pairs_supervised(doc,person_ent,loc_en):
    pairs=[]
    pos_pairs = []
    neg_pairs = []
    for en_id1 in range(len(doc.ents)):
        for en_id2 in range(en_id1, len(doc.ents)):
            en1 = doc.ents[en_id1]
            en2 = doc.ents[en_id2]
            if (en1.label_ == PERSON and en2.label_ in LOCTATION_STRS) or \
                    (en1.label_ in LOCTATION_STRS and en2.label_ == PERSON):
                pairs.append((en1, en2))
    for ne1, ne2 in pairs:
        if (ne1.text == person_ent and ne2.text == loc_en) or \
                (ne1.text == loc_en and ne2.text == person_ent):
            pos_pairs.append(([ne1, ne2], 1))
        else:
            neg_pairs.append(([ne1, ne2], 0))
    print(len(pos_pairs), len(neg_pairs))
    return pos_pairs + neg_pairs[:]


def replace_en(doc,str,labels):
    gold_words  = re.split(' |-',str)
    sent_words = [w.text for w in doc]
    label = nlp.vocab.strings[labels[0]]
    try:
        start, end  = sent_words.index(gold_words[0]),sent_words.index(gold_words[len(gold_words)-1])+1
    except ValueError:
        logging.info("found bad title : "+str)
        return
    span = None
    for en in doc.ents:
        if en.start==start or en.end==end:
            span=en
            break
    if span :
        doc.ents = [e for e in doc.ents if e !=span]
        if  span.label_ in labels:
            label = span.label
    doc.ents += ((label, start, end),)




if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    samples= process_file(infile)
    save_words(outfile,samples)