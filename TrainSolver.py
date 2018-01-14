# Save Model Using Pickle
import scipy
from sklearn.linear_model import LogisticRegression
import pickle
import sys
import spicy
from sklearn.datasets import load_svmlight_file

from ConvertFeatures import convert_line
from ExtractFeatures import extract_sentence
from spc import read_annotated_line, read_sentences_from_annotated
from utils import load_map, convert_features


def train_model(in_file):
    X, Y = load_svmlight_file(in_file)

    model = LogisticRegression(max_iter=pred_str)
    model.fit(X, Y)
    correct=0.0
    for v,x in enumerate(X):
        correct+=1 if model.predict(x)== Y[v] else 0
    print correct/len(Y)
    return model

def save_model(out_file,model):
    pickle.dump(model, open(out_file, 'wb'))

def main(args):
    model = train_model(args[1])
    save_model(args[2],model)

if __name__ == "__main__":
    main(sys.argv)
    # in_file = get_abs_file('..\\out\\feature_map')
    # model_file = ('..\\out\\model_test1')
    # main(["",in_file,model_file])

