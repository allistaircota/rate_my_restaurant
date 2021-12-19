from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from datetime import datetime


import string

import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer#, WordNetLemmatizer

from scipy.sparse import csr_matrix

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.linear_model import Ridge



from utils import tokenizer_stemming, convert_to_array, bound_predict, word_count

X_train = pd.read_csv('X_train.csv')
pipeline = joblib.load('linreg_pipeline.pkl')
categories = ['Sandwiches', 'Pizza', 'Bars',
'American (Traditional)', 'American (New)', 'Italian',
'Breakfast & Brunch', 'Coffee & Tea', 'Chinese', 'Seafood',
'Burgers', 'Fast Food', 'Salad', 'Cafes', 'Mexican', 'Bakeries',
'Japanese', 'Delis', 'Specialty Food']

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    
    if request.method == 'POST':
        df = pd.DataFrame(columns=X_train.columns.to_list(), dtype='float64')
        date = request.form['review_date']
        date = datetime.strptime(date, '%Y-%m-%dT%H:%M')
        df.loc[0, 'Year'] = date.year
        df.loc[0, 'Month'] = date.month
        df.loc[0, 'Day of Week'] = date.weekday()
        df.loc[0, 'Hour'] = date.hour
        df.loc[0, 'latitude'] = float(request.form['latitude'])
        df.loc[0, 'longitude'] = float(request.form['longitude'])
        df.loc[0, 'number_of_branches'] = float(request.form['number_of_branches'])
        df.loc[0, 'reviews_per_week'] = float(request.form['reviews_per_week'])
        df.loc[0, 'RestaurantsGoodForGroups'] = float(request.form['RestaurantsGoodForGroups'])
        df.loc[0, 'HasTV'] = float(request.form['HasTV'])
        df.loc[0, 'GoodForKids'] = float(request.form['GoodForKids'])
        df.loc[0, 'RestaurantsTakeOut'] = float(request.form['RestaurantsTakeOut'])
        df.loc[0, 'RestaurantsPriceRange2'] = float(request.form['RestaurantsPriceRange2'])
        df.loc[0, 'RestaurantsReservations'] = float(request.form['RestaurantsReservations'])
        df.loc[0, 'RestaurantsDelivery'] = float(request.form['RestaurantsDelivery'])
        df.loc[0, 'OutdoorSeating'] = float(request.form['OutdoorSeating'])
        df.loc[0, 'NoiseLevel'] = float(request.form['NoiseLevel'])
        df.loc[0, 'BusinessAcceptsCreditCards'] = float(request.form['BusinessAcceptsCreditCards'])
        df.loc[0, 'text'] = request.form['text']
        df.loc[0, 'review_length'] = word_count(df.loc[0, 'text'])

        if request.form['RestaurantsAttire'] == 'casual':
            df.loc[0, 'RestaurantsAttire_dressy'] = 0
            df.loc[0, 'RestaurantsAttire_formal'] = 0

        elif request.form['RestaurantsAttire'] == 'dressy':
            df.loc[0, 'RestaurantsAttire_dressy'] = 1
            df.loc[0, 'RestaurantsAttire_formal'] = 0

        else:
            df.loc[0, 'RestaurantsAttire_dressy'] = 0
            df.loc[0, 'RestaurantsAttire_formal'] = 1

        ### Alcohol
        if request.form['Alcohol'] == 'beer_and_wine':
            df.loc[0, 'Alcohol_full_bar'] = 0
            df.loc[0, 'Alcohol_none'] = 0

        elif request.form['Alcohol'] == 'full_bar':
            df.loc[0, 'Alcohol_full_bar'] = 1
            df.loc[0, 'Alcohol_none'] = 0

        else:
            df.loc[0, 'Alcohol_full_bar'] = 0
            df.loc[0, 'Alcohol_none'] = 1

        if request.form['WiFi'] == 'free':
            df.loc[0, 'WiFi_no'] = 0
            df.loc[0, 'WiFi_paid'] = 0

        elif request.form['WiFi'] == 'no':
            df.loc[0, 'WiFi_no'] = 1
            df.loc[0, 'WiFi_paid'] = 0

        else:
            df.loc[0, 'WiFi_no'] = 0
            df.loc[0, 'WiFi_paid'] = 1

        category_list = request.form.getlist('category')
        for category in categories:
            if category in category_list:
                df.loc[0, category] = 1
            else:
                df.loc[0, category] = 0

        
        prediction = bound_predict(pipeline.predict(df))[0][0]
        rounded_prediction = int(round(prediction, 0))

    return render_template('index.html',
    rounded_prediction=rounded_prediction)