import logging

import scipy
import pickle
import sys
from ExtractFeatures import get_DEV_pairs, extract_features_pairs
from temp_code.spc import read_sentences_from_annotated, nlp
from utils import load_map, convert_features, PERSON, save_file

LIVE_IN = "Live_In"

def get_en_type_from_pair(pair):
    for en in pair[0]:
        if en.label_ == PERSON:
            p_en = en
        else:
            loc_en = en
    return p_en,loc_en
def predict(model):
    predictions =[]
    rev_featuremap = {v:k for k,v in featuremap.items()}
    match = total_preds = total_true =  0.0
    for sent_title, sent_str in read_sentences_from_annotated(test_file_name):
        sent_id,en1,rel,en2 = sent_title.split('\t')
        if not rel == LIVE_IN:
            continue
        doc = nlp(unicode(sent_str.strip()))
        pairs = get_DEV_pairs(doc, sent_title)
        samples = extract_features_pairs(doc, pairs)

        for i,(y,x) in enumerate(samples):
            feature_v = convert_features(featuremap,x)
            v = scipy.sparse.csr_matrix(feature_v)
            pred = model.predict(v)
            pred_str = rev_featuremap[str(int(pred[0]))]
            if pred_str=='1': # if we predict current pair includes the relation
                p_en,loc_en = get_en_type_from_pair(pairs[i])
                predictions.append((sent_id, p_en, rel, loc_en, sent_str))
                if str(y)=='1': # if this pair was tagged with the relation
                    logging.info(sent_title + " " + sent_str)
                    match+=1
                total_preds+=1
            total_true+=1 if y==1 else 0
    return total_preds,total_true,match,predictions

if __name__=="__main__":
    model_file_name=  sys.argv[1]
    map_file_name = sys.argv[2]
    predict_file = sys.argv[3]
    test_file_name = sys.argv[4]
    length =  sys.argv[5] if len(sys.argv)>5 else -1
    length = int(length)
    model = model = pickle.load(open(model_file_name, 'rb'))
    featuremap = load_map(map_file_name)
    total_preds,total_true,match,predictions  = predict(model)
    prec = match / total_preds
    recall= match / total_true
    print(match / total_preds)
    print(match / total_true)
    print(2*prec*recall/(prec+recall))
    save_file(predict_file,predictions)
    pass