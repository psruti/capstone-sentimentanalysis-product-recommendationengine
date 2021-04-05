import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import unicodedata
#from pycontractions import Contractions
#import contractions
#from contractions import contractions_dict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import itertools

# Decompress the file
import gzip
# Datetime
from datetime import datetime

# text preprocessing
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.toktok import ToktokTokenizer
import gensim
import re


import unicodedata
tokenizer = ToktokTokenizer()
nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)

## Modeling
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
#from catboost import CatBoostClassifier, Pool
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


## Warnings
import warnings
#from scipy import stats
warnings.filterwarnings('ignore')



df=pd.read_csv('sample30.csv')
df['Feedback'] = df.copy()['reviews_title']+" "+df.copy()['reviews_text']
df.drop(['reviews_title','reviews_text'],axis=1,inplace=True)
indexm=(round(100*(df.isnull().sum()/len(df.index)),2)>=40)
df_t = indexm.to_frame("columns")
cols_to_drop = df_t[df_t["columns"]==True].index.to_list()
#Dropping the columns which have null percentage more than 40
df.drop(columns=cols_to_drop,inplace=True)
# FILLING NULL VALUES IN manufacturer NAME WITH FIRST WORD FROM brand
df.loc[df.manufacturer.isnull(),'manufacturer'] = df.brand
# DROPPING NULL VALUE COLUMNS
df=df.dropna(subset=['reviews_doRecommend','reviews_username','reviews_date','Feedback'])
# DROP DUPLICATES
df.drop_duplicates(inplace = True)
#TEXT PROCESSING
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

# special_characters removal
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation_and_splchars(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_word = remove_special_characters(new_word, True)
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

stopword_list= stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
#stopword_list.remove('headphone')
#stopword_list.remove('headphones')
#stopword_list.remove('earbuds')
#stopword_list.remove('bud')
#stopword_list.remove('ear')
#stopword_list.remove('sony')
#stopword_list.remove('product')


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopword_list:
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words)
    return words

def lemmatize(words):
    lemmas = lemmatize_verbs(words)
    return lemmas

def normalize_and_lemmaize(input):
    sample = denoise_text(input)
    #sample = expand_contractions(sample)
    sample = remove_special_characters(sample)
    words = nltk.word_tokenize(sample)
    words = normalize(words)
    lemmas = lemmatize(words)
    return ' '.join(lemmas)

df['clean_feedback'] = df['Feedback'].map(lambda text: normalize_and_lemmaize(text))
def token (text):
    token = [w for w in nltk.word_tokenize(text)]
    return token

# To create token feature
df['token'] = df['clean_feedback'].apply(token)
# Function for creating a column to see the length of the review text
def length(text):
    length = len([w for w in nltk.word_tokenize(text)])
    return length

# Apply length function to create review length feature
df['review_length'] = df['Feedback'].apply(length)
df['user_sentiment'] = df['user_sentiment'].apply(lambda x: 1 if x =='Positive' else 0)

# Sentiment Analysis (CV-TF_IDF)
df2=df.copy()
# Drop unnecessary columns
df2 = df2.drop(['brand','categories','manufacturer','name','reviews_date','reviews_doRecommend','reviews_rating','review_length'], axis=1)
#df2['user_sentiment'] = df2['user_sentiment'].apply(lambda x: 1 if x =='Positive' else 0)
# Splitting the Data Set into Train and Test Sets
X = df2['clean_feedback']
y = df2['user_sentiment']

# Splitting Dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Function for converting the "classification report" results to a dataframe
def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred)

    avg = list(precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum()
    avg[-1] = total

    class_report_df['weighted avg'] = avg

    return class_report_df.T

# Function for adding explanatory columns and organizing all dataframe
def comparison_matrix(y_test, y_pred, label, vector):
    df = pandas_classification_report(y_test, y_pred)
    df['class']=['negative', 'positive', 'average']
    df['accuracy']= metrics.accuracy_score(y_test, y_pred)
    df['model'] = label
    df['vectorizer'] = vector
    df = df[['vectorizer', 'model', 'accuracy', 'class', 'precision', 'recall', 'f1-score', 'support']]
    return df

# Creating a function for applying different algorithms
def modeling(Model, X_train=count_vect_train, X_test=count_vect_test):
    """
    This function apply countVectorizer with machine learning algorithms.
    """

    # Instantiate the classifier: model
    model = Model

    # Fitting classifier to the Training set (all features)
    model.fit(X_train, y_train)

    global y_pred
    # Predicting the Test set results
    y_pred = model.predict(X_test)

    # Assign f1 score to a variable
    score = f1_score(y_test, y_pred, average='weighted')

    # Printing evaluation metric (f1-score)
    print("f1 score: {}".format(score))

# Create the word vector with TF-IDF Vectorizer
tfidf_vect = TfidfVectorizer(ngram_range=(1, 1))
tfidf_vect_train = tfidf_vect.fit_transform(X_train)
tfidf_vect_train = tfidf_vect_train.toarray()
tfidf_vect_test = tfidf_vect.transform(X_test)
tfidf_vect_test = tfidf_vect_test.toarray()

# Call the modeling function for random forest classifier with TF-IDF and print f1 score
modeling(RandomForestClassifier(n_estimators = 200,class_weight="balanced", random_state = 42),
         tfidf_vect_train, tfidf_vect_test)

# Assign y_pred to a variable for further process
y_pred_tfidf_rf = y_pred

# Compute and print the classification report
print(classification_report(y_test, y_pred_tfidf_rf))
#Recomendation Engine
## Item-Item Recomendation Engine

ratings=df.copy()
ratings.drop(labels=['brand','categories','manufacturer','reviews_date','reviews_doRecommend','user_sentiment','Feedback','token',
                     'clean_feedback','review_length'],axis=1,inplace=True)
ratings.drop_duplicates(inplace=True)
train, test = train_test_split(ratings, test_size = 0.30, random_state = 42)
df_product_features = train.pivot_table(index = 'reviews_username', columns = 'name', values ='reviews_rating').fillna(0)
dummy_train = train.copy()
dummy_test = test.copy()
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x:0 if x>=1 else 1)
dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x:1 if x>=1 else 0)
df_pivot = train.pivot_table(index = 'reviews_username', columns = 'name', values ='reviews_rating').T
mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T
from sklearn.metrics.pairwise import pairwise_distances

# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)
item_correlation[item_correlation<0]=0
item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)
# The final rating matrix used for finding the recommendation of users
item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
#Evaluation - Item Item
test.columns

common =  test[test.name.isin(train.name)]
common.shape
common_item_based_matrix = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T
item_correlation_df = pd.DataFrame(item_correlation)
item_correlation_df['name'] = df_subtracted.index
item_correlation_df.set_index('name',inplace=True)
list_name = common.name.tolist()

item_correlation_df.columns = df_subtracted.index.tolist()

item_correlation_df_1 =  item_correlation_df[item_correlation_df.index.isin(list_name)]

item_correlation_df_2 = item_correlation_df_1.T[item_correlation_df_1.T.index.isin(list_name)]

item_correlation_df_3 = item_correlation_df_2.T
item_correlation_df_3[item_correlation_df_3<0]=0

common_item_predicted_ratings = np.dot(item_correlation_df_3, common_item_based_matrix.fillna(0))
dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T.fillna(0)

common_item_predicted_ratings = np.multiply(common_item_predicted_ratings,dummy_test)
common_ = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T
from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_item_predicted_ratings.copy()
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))
l=common_.columns
df1_y=pd.DataFrame(y, columns =l )
# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))
rmse = (sum(sum((pd.DataFrame(common_.values - df1_y.values))**2))/total_non_nan)**0.5
print(rmse)


