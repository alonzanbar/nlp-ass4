import codecs 
import spacy 
import sys
from socket import *

from pathlib import Path
from spacy import displacy
from spacy.displacy import parse_ents

import render_ex
from render_ex import EntityRendererEX

nlp = spacy.load('en')

def read_lines(fname):
    for line in codecs.open(fname, encoding="utf8"):
        sent, sent_id = strip_and_split_sen(line)
        yield sent_id, sent


def strip_and_split_sen(line):
    sent_id, sent = line.strip().split("\t")
    sent = sent.replace("-LRB-", "(")
    sent = sent.replace("-RRB-", ")")
    return sent, sent_id


def read_sentences_from_annotated(fname,):
    for line in codecs.open(fname, encoding="utf8"):
        line1 = line.strip().split("\t")
        sent_title = "\t".join([line1[0], line1[1], line1[2], line1[3]])
        sent  = line[line.find("(") + 1:line.find(")")]
        sent = sent.replace("-LRB-", "(")
        sent = sent.replace("-RRB-", ")")
        yield sent_title, sent


def save_file(outfile,predictions):
    with open(outfile,"w") as f :
        for set_id,line in predictions:
            f.writelines(set_id+"\t"+line+"\n")


def save_html(lines,file):
    options = {'ents': ['PERSON', 'LOC', 'GPE']}
    entity_renderer = EntityRendererEX(options=options)
    parsed = [render_ex.parse_ents(doc, options) for doc in lines]
    html = entity_renderer.render(parsed)
    open(file, 'w').write(html)


if __name__=="__main__":
    lines =[]
    for i, st in enumerate(read_sentences_from_annotated(sys.argv[1])):
        sent_title , sent_str = st
        nlpline = nlp(sent_str)
        nlpline.user_data = {'title': sent_title}
        lines.append(nlpline)
        # for word in sent:
        #     head_id = str(word.head.i+1)        # we want ids to be 1 based
        #     if word == word.head:               # and the ROOT to be 0.
        #         assert(word.dep_=="ROOT"),word.dep_
        #         head_id = "0" # root
        #     print "\t".join([str(word.i+1), word.text, word.lemma_, word.tag_, word.pos_, head_id, word.dep_, word.ent_iob_, word.ent_type_])
        # print
        # print "#", Noun Chunks:
        # for np in sent.noun_chunks:
        #    print(np.text, np.root.text, np.root.dep_, np.root.head.text)
    # html = displacy.render(lines, style='dep', page=True)
    #
    # open("out/dep.html",'w').write(html)
    save_html(lines,"temp/min.html")

