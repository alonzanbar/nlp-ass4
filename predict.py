

import scipy
import pickle
import sys
from ExtractFeatures import extract_sentence
from spc import read_sentences_from_annotated
from utils import load_map, convert_features

if __name__=="__main__":
    model_file_name=  sys.argv[1]
    map_file_name = sys.argv[2]
    length =  sys.argv[3] if len(sys.argv)>3 else -1
    length = int(length)
    model = model = pickle.load(open(model_file_name, 'rb'))
    featuremap = load_map(map_file_name)

    rev_featuremap = {v:k for k,v in featuremap.items()}
    match =all=cnt=0.0
    for title, sent in read_sentences_from_annotated('data/DEV.annotations'):
        if cnt>length and length!=-1 :
            break
        cnt+=1
        samples = extract_sentence(title, sent)
        for y,x in samples:
            feature_v = convert_features(featuremap,x)
            v = scipy.sparse.csr_matrix(feature_v)
            pred = model.predict(v)
            pred_str = rev_featuremap[str(int(pred[0]))]
            if pred_str=='1':
                match+=1 if str(y) == '1' else 0
                all+=1
    print(match/all)
    print(match , all)

    pass