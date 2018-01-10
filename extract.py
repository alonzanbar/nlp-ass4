import sys
from spc import read_lines, nlp, save_file

LIVE_IN = "Live_In"

PERSON = 'PERSON'

LOCTATION_STRS = ['LOC','GPE']


def find_anch_ent_type(sent,word,child_types,entType,dis=20):
    if not (word.ent_type_ in child_types):
        return None
    ancs = word.ancestors
    for anc in ancs:
        if anc.ent_type_ in entType:
            if abs(word.i-anc.i)<dis:
                for ent in sent.ents:
                    if ent.root == anc:
                        return ent

    return None


def predict(infile):
    predictions=[]
    for sent_id,sent_str in read_lines(infile):
        sent = nlp(sent_str)
        for ne in sent.ents:


            w_relation = find_anch_ent_type(sent,ne.root,LOCTATION_STRS,[PERSON],10)
            if w_relation:
                predictions.append((sent_id, "\t".join([w_relation.text, LIVE_IN, ne.text, "( " + sent_str + " )"])))

            w_relation = find_anch_ent_type(sent,ne.root,PERSON,LOCTATION_STRS,2)
            if w_relation:
                predictions.append((sent_id, "\t".join([ne.text, LIVE_IN, w_relation.text, "( " + sent_str + " )"])))


    return predictions





if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    predictions = predict(infile)
    save_file(outfile,predictions)