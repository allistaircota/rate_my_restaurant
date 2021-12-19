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

from utils import tokenizer_stemming, convert_to_array


ENGLISH_STOP_WORDS = stopwords.words('english')

X_train = pd.read_csv('../data/X_train.csv')
y_train = pd.read_csv('../data/y_train.csv')
numeric_columns = X_train.dtypes[X_train.dtypes != 'object'].index.to_list()

tfidf_stem_ct = ColumnTransformer([
    ('numeric', 'passthrough', numeric_columns),
    ('tfidf_stem_ct', TfidfVectorizer(tokenizer=tokenizer_stemming, max_features=3000, ngram_range=(1,2)), 'text')
])

linreg_pipeline = Pipeline([
    ('col_trans', tfidf_stem_ct),
    ('make_array', FunctionTransformer(convert_to_array, accept_sparse=True)),
    ('min-max-scaler', MinMaxScaler()),
    ('model', Ridge(alpha=5))
])

linreg_pipeline.fit(X_train, y_train)

joblib.dump(linreg_pipeline, 'linreg_pipeline.pkl')