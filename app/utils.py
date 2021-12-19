import numpy as np
import pandas as pd

import joblib
import string

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from scipy.sparse import csr_matrix

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.linear_model import Ridge

ENGLISH_STOP_WORDS = stopwords.words('english')

def tokenizer_prep(sentence):
    '''
    Returns list of words from a string with all punctuation removed.
    
    PARAMETERS:
    - sentence: str, input text.
    
    RETURNS:
    - listofwords: list, list of substring in the input text,
    excluding punctuation, separated by whitespace.
    
    '''
    # remove punctuation and set to lower case
    # english_stop_words = stopwords.words('english')
    for punctuation_mark in string.punctuation:
        sentence = sentence.replace(punctuation_mark,'').lower()

    # split sentence into words
    listofwords = sentence.split(' ')
    return listofwords


def tokenizer_stemming(sentence):
    '''
    Returns list of stemmed tokens from an input string.
    
    PARAMETERS:
    - sentence: str, input text.
    
    RETURNS:
    - listofstemmed_words: list, list of stemmed tokens 
    from input string
    
    '''
    stemmer = PorterStemmer()
    listofwords = tokenizer_prep(sentence) 
    listofstemmed_words = []
    
    # remove stopwords and any tokens that are just empty strings
    for word in listofwords:
        if (not word in ENGLISH_STOP_WORDS) and (word!=''):
            # Stem words
            stemmed_word = stemmer.stem(word)
            listofstemmed_words.append(stemmed_word)

    return listofstemmed_words

def convert_to_array(sparse_matrix):
    '''
    Converts sparse matrix to dense array
    
    PARAMETERS:
    - sparse_matrix: scipy.sparse.csr_matrix or numpy array
    
    RETURNS:
    - If sparse_matrix is not a scipy.sparse.csr_matrix,
      sparse_matrix is returned. Else, returns the dense array
      form of sparse_matrix.
    
    '''
    
    if type(sparse_matrix) == csr_matrix:
    
        return sparse_matrix.toarray()
    
    else:
        return sparse_matrix

def bound_predict(y_pred):
    '''
    Limits the predicted y values to stay within the range of 1 to 5.
    Predictions less than 1 will be reassigned to a prediction of 1.
    Predictions greater than 5 will be reassigned to a prediction of 5.
    
    PARAMETERS:
    - y_pred (pd.Series): Predicted y values
    
    RETURNS:
    - y_pred (pd.Series): Predicted y values no less than 1 nor greater than 5
    
    '''
    y_pred = np.where(y_pred > 5, 5, y_pred)
    y_pred = np.where(y_pred < 1, 1, y_pred)
    return y_pred

def word_count(input_text):
    '''
    Remove punction from input text and return the word count of the text.
    '''
    for punctuation_mark in string.punctuation:
        input_text = input_text.replace(punctuation_mark,'').lower()
        
    listofwords = input_text.split(' ')
    return len(listofwords)

# if __name__ == "__main__":
#     X_train = pd.read_csv('../data/X_train.csv')
#     y_train = pd.read_csv('../data/y_train.csv')
#     X_test = pd.read_csv('../data/X_test.csv')
#     y_test = pd.read_csv('../data/y_test.csv')
    
#     pipeline = joblib.load('best_linreg_pipeline.pkl')
#     joblib.dump(pipeline, 'pipeline.pkl')
