from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import pickle
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

# Visualizations
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.colors as colors
#%matplotlib inline
from wordcloud import WordCloud, STOPWORDS

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
nlp = spacy.load('en_core_web_sm')#, parse=True, tag=True, entity=True)

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
from sklearn.metrics import recall_score
from gensim.models import Word2Vec
from tqdm import tqdm
import xgboost as xgb
from xgboost import XGBClassifier
#from sklearn.dummy import DummyClassifier

## Warnings
import warnings
from scipy import stats
warnings.filterwarnings('ignore')
import pickle
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("sample30.csv")
	df['Feedback'] = df.copy()['reviews_title'] + " " + df.copy()['reviews_text']
	df.drop(['reviews_title', 'reviews_text'], axis=1, inplace=True)
	indexm = (round(100 * (df.isnull().sum() / len(df.index)), 2) >= 40)
	df_t = indexm.to_frame("columns")
	cols_to_drop = df_t[df_t["columns"] == True].index.to_list()
	df.drop(columns=cols_to_drop, inplace=True)
	df.loc[df.manufacturer.isnull(), 'manufacturer'] = df.brand
	df = df.dropna(subset=['reviews_doRecommend', 'reviews_username', 'reviews_date', 'Feedback'])

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

	stopword_list = stopwords.words('english')
	stopword_list.remove('no')
	stopword_list.remove('not')

	# stopword_list.remove('headphone')
	# stopword_list.remove('headphones')
	# stopword_list.remove('earbuds')
	# stopword_list.remove('bud')
	# stopword_list.remove('ear')
	# stopword_list.remove('sony')
	# stopword_list.remove('product')

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
		# sample = expand_contractions(sample)
		sample = remove_special_characters(sample)
		words = nltk.word_tokenize(sample)
		words = normalize(words)
		lemmas = lemmatize(words)
		return ' '.join(lemmas)

	df['clean_feedback'] = df['Feedback'].map(lambda text: normalize_and_lemmaize(text))

	def token(text):
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

	def token(text):
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
	df['user_sentiment'] = df['user_sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)
	# Recomendation Engine
	## Item-Item Recomendation Engine

	ratings = df.copy()
	ratings.drop(labels=['brand', 'categories', 'manufacturer', 'reviews_date', 'reviews_doRecommend', 'user_sentiment',
						 'Feedback', 'token',
						 'clean_feedback', 'review_length'], axis=1, inplace=True)
	ratings.drop_duplicates(inplace=True)
	train, test = train_test_split(ratings, test_size=0.30, random_state=42)
	df_product_features = train.pivot_table(index='reviews_username', columns='name', values='reviews_rating').fillna(0)
	dummy_train = train.copy()
	dummy_test = test.copy()
	dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x >= 1 else 1)
	dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x >= 1 else 0)
	dummy_train = dummy_train.pivot_table(
		index='reviews_username',
		columns='name',
		values='reviews_rating'
	).fillna(1)
	df_pivot = train.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T
	mean = np.nanmean(df_pivot, axis=1)
	df_subtracted = (df_pivot.T - mean).T
	from sklearn.metrics.pairwise import pairwise_distances

	# Item Similarity Matrix
	item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
	item_correlation[np.isnan(item_correlation)] = 0
	print(item_correlation)
	item_correlation[item_correlation < 0] = 0
	item_predicted_ratings = np.dot((df_pivot.fillna(0).T), item_correlation)
	print(item_predicted_ratings.shape)
	print(dummy_train.shape)
	# The final rating matrix used for finding the recommendation of users
	item_final_rating = np.multiply(item_predicted_ratings, dummy_train)
	print(type(item_final_rating))
	print(item_final_rating.shape)



	df2 = df.copy()
	# Drop unnecessary columns
	df2 = df2.drop(
		['brand', 'categories', 'manufacturer', 'name', 'reviews_date', 'reviews_doRecommend', 'reviews_rating',
		 'review_length'], axis=1)
	#df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	# Features and Labels
	#df['label'] = df['class'].map({'ham': 0, 'spam': 1})
	X = df2['clean_feedback']
	y = df2['user_sentiment']
	# Splitting Dataset into train and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

	# Extract Feature With CountVectorizer
	#cv = CountVectorizer()
	#X = cv.fit_transform(X) # Fit the Data
	#from sklearn.model_selection import train_test_split
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	#from sklearn.naive_bayes import MultinomialNB

	#clf = MultinomialNB()
	#clf.fit(X_train,y_train)
	#clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		d = item_final_rating.loc[data].T.sort_values(by = data,ascending=False)[0:20]
		d1 = pd.merge(d, df, left_on='name', right_on='name', how='left')
		d_test = d1['clean_feedback']
		d_test = d1['clean_feedback']
		y_d_test = d1['user_sentiment']
		# tfidf_vect = TfidfVectorizer(ngram_range=(1, 1))
		# Create the word vector with TF-IDF Vectorizer
		tfidf_vect = TfidfVectorizer(ngram_range=(1, 1))
		tfidf_vect_train = tfidf_vect.fit_transform(X_train)
		tfidf_vect_train = tfidf_vect_train.toarray()
		tfidf_vect_test = tfidf_vect.transform(d_test)
		tfidf_vect_test = tfidf_vect_test.toarray()
		#load the model from disk
		loaded_model = pickle.load(open('pickle/model_randomforest_balance_tfidf.pkl', 'rb'))
		result = loaded_model.predict(tfidf_vect_test)
		result1 = pd.DataFrame(result, columns=['result_pred'])
		s2 = pd.concat([d1, result1], axis=1)
		reco_name = s2[s2['result_pred'] == 1]['name']
		reco_name.drop_duplicates(inplace=True)
		l = pd.merge(d, reco_name, left_on='name', right_on='name', how='inner')
		final_result = l['name'].values.tolist()
		# print(result)
		#vect = cv.transform(data).toarray()
		#my_prediction = clf.predict(vect)
	return render_template('result.html',len=len(final_result),name = data,prediction = final_result)



if __name__ == '__main__':
	app.run(debug=True)