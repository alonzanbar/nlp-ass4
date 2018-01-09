import sys
from spc import read_lines, nlp, save_file

LIVE_IN = "Live_In"

PERSON = 'PERSON'

LOCTATION_STRS = ['LOC','GPE']


def traverse_person(word):
    tempwrod = word
    while(tempwrod.dep_!='ROOT'):
        tempwrod = tempwrod.head
        if tempwrod.ent_type_== ('%s' % PERSON):
            return tempwrod
        elif tempwrod.ent_type_:
            return None
    return None


def predict(infile):
    predictions=[]
    for sent_id,sent_str in read_lines(infile):
        sent = nlp(sent_str)
        for ne in sent.ents:
            if ne.root.ent_type_   in LOCTATION_STRS:
                w_relation = traverse_person(ne.root)
                if w_relation:
                    for ent in sent.ents:
                        if ent.root == w_relation:
                            predictions.append((sent_id, "\t".join([ent.text, LIVE_IN, ne.text])))
                            break

    return predictions





if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    predictions = predict(infile)
    save_file(outfile,predictions)