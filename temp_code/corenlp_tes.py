from nltk import word_tokenize, pos_tag, ne_chunk, Tree
import sys

from spc import read_sentences_from_annotated, nlp
from pycorenlp import StanfordCoreNLP
import networkx as nx

LABELS = ['PERSON','LOC','GPE']

match = 0.0
tokens_len = 0.0
sentences=0.0
matched_sentences=0.0

for i, st in enumerate(read_sentences_from_annotated(sys.argv[1])):
    sentences +=1
    sent_title, sent_str = st

    nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))


    def get_stanford_annotations(text, port=9000,
                                 annotators='tokenize,ssplit,pos,lemma,depparse,parse'):
        output = nlp.annotate(text, properties={
            "timeout": "10000",
            "ssplit.newlineIsSentenceBreak": "two",
            'annotators': annotators,
            'outputFormat': 'json'
        })
        return output

    # Parse the text
    sent_str="Fort Hamilton , Brooklyn , N.Y. ;"
    annotations = get_stanford_annotations(sent_str, port=9000,
                                           annotators='tokenize,ssplit,pos,lemma,depparse,parse,ner')
    tokens = annotations['sentences'][0]['tokens']

    # Load Stanford CoreNLP's dependency tree into a networkx graph
    edges = []
    dependencies = {}
    for edge in annotations['sentences'][0]['basic-dependencies']:
        edges.append((edge['governor'], edge['dependent']))
        dependencies[(min(edge['governor'], edge['dependent']),
                      max(edge['governor'], edge['dependent']))] = edge

    graph = nx.Graph(edges)
    #a = nx.shortest_path(source=)
    # pprint(dependencies)
    # print('edges: {0}'.format(edges))

#     nlpline = nlp(unicode(sent_str.strip()))
#     a = ne_chunk(pos_tag(word_tokenize(sent_str)))
#     nltkents = [en for en in a if type(en)==Tree]
#     if len(nltkents) != len(nlpline.ents):
#         continue
#     match_bool=True
#     tokens_len += len(nltkents)
#     ents=[]
#     for i,ent in enumerate(nlpline.ents):
#         if (nltkents[i].label() not in LABELS) and  (ent.label_ not in LABELS):
#             continue
#         if nltkents[i].label() == ent.label_:
#             match+=1
#             ents.append(ent)
#
# print(match, (match/tokens_len))
# print("matched sentences: {0} , sentences: {1}".format(matched_sentences,sentences))