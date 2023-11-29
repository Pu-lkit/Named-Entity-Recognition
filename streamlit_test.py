import streamlit as st
from sklearn.preprocessing import StandardScaler
import nltk
import numpy as np
import string
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk import pos_tag
import joblib
from nltk.tokenize import word_tokenize

clf = joblib.load('svm_model_linear.joblib')

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

            # l.append(n/len(id['tokens']))
            # l.append(len(i))

            ### end creation of your feature matrix 
            feature_matrix.append(l)
    feature_matrix = np.array(feature_matrix)

    ss = StandardScaler()
    standardized_feature_matrix = ss.fit_transform(feature_matrix)
    return standardized_feature_matrix


def test(sentence):
    
    tokens = word_tokenize(sentence)
    pos = pos_tag(tokens)
    pos_tags = []
    for i in pos:
        if i[1] in mappings:
            pos_tags.append(mappings[i[1]])
        else:
            pos_tags.append(100)
    
    inp = [{'tokens' : tokens, 'pos_tags' : pos_tags}]
    X_feature = featurize_data(inp)
    pred = clf.predict(X_feature)

    ans = []

    for i in range(len(tokens)):
        ans.append(f"{tokens[i]}_{pred[i]} ")
    return ans


st.title("Name Entity Recognition Application")
user_input = st.text_area("Enter a sentence:")

if user_input:
    # Remove special characters except for necessary ones
    pos_tags = test(user_input)
    st.write("User Input Sentence:")
    st.markdown(f"> {user_input}")
    ans = test(user_input)
    tagged_sentence = ""
    # for i in ans:
    #     tagged_sentence += i

    for word in ans:
        tagged_sentence += f'<span style="background-color: white; padding: 4px; border-radius: 4px; color: black;">{word} </span> '
    st.write("Name Entity Recognition sentence:")
    st.markdown(tagged_sentence, unsafe_allow_html=True)
