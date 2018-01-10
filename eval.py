import sys

from collections import defaultdict

import time

from spc import nlp, save_html


def eval(goldfile,predfile):
    gold_vecs,sum_gold= get_set_vec(goldfile)
    pred_vecs,sum_pred = get_set_vec(predfile)
    good =  0.0

    bad={}
    for set_id,pred_vec in pred_vecs.items():
        gold_vec=gold_vecs[set_id]
        for p in pred_vec:
            if p in gold_vec:
                good +=1
            else:
                bad[set_id]= p

    return  good/sum_pred , good/sum_gold,bad


def extract_pred_lines(file,lines):
    bad_lines=[]
    good_lines=[]
    for sent in open(file, 'r'):
        line = sent.strip('\n').split("\t")

        nlpline = nlp(unicode(sent[sent.find("(") + 1:sent.find(")")]))
        nlpline.user_data = {'title':"\t".join([line[0],line[1], line[2], line[3]])}


        if line[0] in lines:
            bad_lines.append(nlpline)
        else:
            good_lines.append(nlpline)
    return bad_lines,good_lines

def get_set_vec(goldfile):
    sentvec = defaultdict(set)
    sum_s=0.0
    for sent in open(goldfile,'r'):
        line = sent.strip('\n').split("\t")
        if (line[2]=="Live_In"):
            sentvec[line[0]].add("\t".join([line[0],line[1], line[2], line[3]]))
            sum_s+=1
    return sentvec,sum_s


if __name__ == '__main__':
    goldfile = sys.argv[1]
    predfile = sys.argv[2]
    prec,rec,b = eval (goldfile,predfile)
    f1 = prec*2 * rec / (prec+rec)
    timestr = time.strftime("%d%m%Y-%H%M%S")
    bad_lines,good_lines =extract_pred_lines(predfile,b.keys())
    save_html(bad_lines,"temp/bad.html")
    save_html(good_lines, "temp/good.html")

    with open("temp/prec"+timestr,"w") as f:
        f.write("F1: %.2f, precision: %.2f, recall: %.2f" % (f1,prec,rec))
        f.write("\n")
        f.write("\n".join(sorted(b.values())))
