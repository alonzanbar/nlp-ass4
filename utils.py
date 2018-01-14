import numpy as np


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