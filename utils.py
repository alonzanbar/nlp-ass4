import logging
import numpy as np

PERSON = 'PERSON'
LOCTATION_STRS = ['LOC','GPE']
logging.basicConfig(filename='example.log',level=logging.INFO)

def load_map(map_file_name):
    featuremap = {i:f for i,f in [p.strip('\n').split('\t')  for p in file(map_file_name) ]}
    return featuremap

def convert_features(fmap,feature_l):
    dense_fet_v = np.zeros(len(fmap))
    for feaure_text in feature_l:
        if feaure_text in fmap:
            f = int(fmap[feaure_text])
            dense_fet_v[f] = 1.0
    return [dense_fet_v]

def save_file(outfile,predictions):
    with open(outfile,"w") as f :
        for set_id,en1,rel,en2,sent_str in predictions:
            line = "\t".join([set_id,en1.text,rel, en2.text, "( " + sent_str + " )"])
            f.writelines(line+"\n")