# Named-Entity-Recognition
The following are the requirements : 
--- nltk version '3.8.1'
--- joblib version '1.3.2'
--- sklearn version '1.3.1'
--- numpy version '1.24.3'
--- datasets version '2.15.0'

--- nltk should be able to use pos_tag and stopwords 
(
if not able to use stopwords, try : 
import nltk
nltk.download('stopwords')
)

(
if not able to use pos_tag, try:
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
)

- The train.py file will train a model of linear SVM and save it.
- The streamlit_test.py when run using 'streamlit run streamlit_test.py' loads the trained model and is ready for NER. 
