from sklearn.preprocessing import StandardScaler
import nltk
from datasets import load_dataset
import numpy as np
import string
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk import pos_tag
import joblib

mappings = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
 'WP': 44, 'WP$': 45, 'WRB': 46 }

month_names = {
    'january', 'february', 'march', 'april', 'may', 'june', 
    'july', 'august', 'september', 'october', 'november', 'december'
}

day_names = {
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
}

def download_hugging_face_dataset():
    return load_dataset("conll2003")


def transform_labels(X):
    labels = []
    for id in X:
        for i in id['ner_tags']:
            if i > 0:
                labels.append(1)
            else:
                labels.append(0)
    return np.array(labels)


def featurize_data(X):
    feature_matrix = []
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    for id in X:
        for n in range(len(id['tokens'])):
            l = []
            i = id['tokens'][n]
            ### create your feature matrix here 
            if i[0].isupper(): ## first letter capital 
                l.append(1)
            else:
                l.append(0)

            if i.lower() in stopwords_set: ## word in stopwords set from nltk
                l.append(1)
            else:
                l.append(0)

            if i == i.upper():## full word in uppercase like acronymns - 'WHO'(organisation)
                l.append(1)
            else:
                l.append(0)

            if i == string.punctuation: ## punctuation sign or not 
                l.append(1)
            else:
                l.append(0)
            
            l.append(id['pos_tags'][n])
            if n != len(id['tokens'])-1:
                l.append(id['pos_tags'][n+1])
            else:
                l.append(7)

            if n != 0:
                l.append(id['pos_tags'][n-1])
            else:
                l.append(7)

            if i[-1] not in {'a','e','i','o','u'}:
                l.append(1)
            else:
                l.append(0)

            if i.lower() in month_names or i.lower() in day_names:
                l.append(1)
            else:
                l.append(0)

            ### end creation of your feature matrix 
            feature_matrix.append(l)
    feature_matrix = np.array(feature_matrix)

    ss = StandardScaler()
    standardized_feature_matrix = ss.fit_transform(feature_matrix)
    return standardized_feature_matrix




def main():
    dataset = download_hugging_face_dataset()
    train_dataset = dataset['train']
    valid_dataset = dataset['validation']
    test_dataset = dataset['test']

    features = featurize_data(train_dataset)
    labels = transform_labels(train_dataset)
    test_features = featurize_data(test_dataset)
    test_labels = transform_labels(test_dataset)
    valid_features = featurize_data(valid_dataset)
    valid_labels = transform_labels(valid_dataset)

    # print(features.shape)
    clf = SVC(kernel = 'linear')
    clf.fit(features, labels)
    joblib.dump(clf, 'svm_model_linear.joblib')

    # choice = input("Want to enter testing or stats or both (t or s or b): ")
    # if choice == 's' or choice == 'b':
    # clf = joblib.load('svm_model.joblib')
    predictions = clf.predict(valid_features)
    print(classification_report(y_true = valid_labels, y_pred=predictions))


if __name__ == "__main__":
    main()
