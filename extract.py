import sys

from nltk import Tree, word_tokenize, pos_tag, ne_chunk

from spc import read_lines, nlp, save_file

LIVE_IN = "Live_In"

PERSON = 'PERSON'

LOCTATION_STRS = ['LOC','GPE']


def find_anch_ent_type(ents,word,child_types,entType,dis=20):
    if not (word.ent_type_ in child_types):
        return None
    ancs = word.ancestors
    for anc in ancs:
        if anc.ent_type_ in entType:
            if abs(word.i-anc.i)<dis:
                for ent in ents:
                    if ent.root == anc:
                        return ent

    return None


def predict(infile):
    predictions=[]
    LABELS  = [PERSON] + LOCTATION_STRS
    for sent_id,sent_str in read_lines(infile):
        a = ne_chunk(pos_tag(word_tokenize(sent_str)))
        nltkents = [en for en in a if type(en) == Tree]
        nlpline = nlp(sent_str)
        ents = []
        if len(nltkents) != len(nlpline.ents):
            continue
        for i, ent in enumerate(nlpline.ents):
            if (nltkents[i].label() not in LABELS) and (ent.label_ not in LABELS):
                continue
            if nltkents[i].label() == ent.label_:
                ents.append(ent)

        for ne in ents:


            w_relation = find_anch_ent_type(ents,ne.root,LOCTATION_STRS,[PERSON],10)
            if w_relation:
                predictions.append((sent_id, "\t".join([w_relation.text, LIVE_IN, ne.text, "( " + sent_str + " )"])))

            w_relation = find_anch_ent_type(ents,ne.root,PERSON,LOCTATION_STRS,2)
            if w_relation:
                predictions.append((sent_id, "\t".join([ne.text, LIVE_IN, w_relation.text, "( " + sent_str + " )"])))


    return predictions





if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    predictions = predict(infile)
    save_file(outfile,predictions)