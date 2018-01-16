import scipy
import pickle
import sys
from ExtractFeatures import extract_sentence
from spc import read_sentences_from_annotated, save_file
from utils import load_map, convert_features



if __name__=="__main__":
    model_file_name=  sys.argv[1]
    map_file_name = sys.argv[2]
    predict_file = sys.argv[3]
    length =  sys.argv[4] if len(sys.argv)>4 else -1
    length = int(length)
    model = model = pickle.load(open(model_file_name, 'rb'))
    featuremap = load_map(map_file_name)
    predictions =[]
    rev_featuremap = {v:k for k,v in featuremap.items()}
    match = total_preds = total_true = cnt = 0.0
    for title, sent in read_sentences_from_annotated('data/DEV.annotations'):
        if cnt>length and length!=-1 :
            break
        cnt+=1
        samples = extract_sentence(title, sent.strip())
        for y,x in samples:
            feature_v = convert_features(featuremap,x)
            v = scipy.sparse.csr_matrix(feature_v)
            pred = model.predict(v)
            pred_str = rev_featuremap[str(int(pred[0]))]
            if pred_str=='1':
                if str(y)=='1':
                    predictions.append ((title,sent))
                    match+=1

                total_preds+=1
            total_true+=1 if y==1 else 0
            print(match / (total_preds+0.00001))
    prec = match / total_preds
    recall= match / total_true
    print(match / total_preds)
    print(match / total_true)
    print(2*prec*recall/(prec+recall))
    save_file(predict_file,predictions)
    pass