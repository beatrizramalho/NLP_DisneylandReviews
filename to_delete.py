#!/usr/bin/env python
# coding: utf-8

#Import libraries:
import pandas as pd
import numpy as np

import re

import nltk
nltk.download('punkt')
nltk.download('wordnet') # wordnet is the most well known lemmatizer for english
nltk.download('stopwords')

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.probability import FreqDist

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
import seaborn as sns


# # Upload and Analysis data

#Upload data:

data = pd.read_csv("C:\\Users\\beatr\\Documents\\Beatriz\\Ironhack\\Projects\\NLP_DisneylandReviews\DisneylandReviews.csv", encoding='latin-1')

data.head()

#data = data.sample(n = 10000)

# Data Analysis:

len(data)

data.info()

data.isna().sum()

data["Rating"].value_counts()

data.groupby("Branch")["Rating"].count()

data.groupby(["Branch","Rating"])["Rating"].count()

data["Rating"].describe()

data["Rating"].value_counts(normalize=True)

plt.figure(figsize=(8,8))
sns.countplot(data['Rating'])
plt.title('Ratings Count in the dataset',fontsize=15)
plt.xlabel('Rating',fontsize=8)
plt.ylabel('Count',fontsize=8)


# # Treating the Reviews column

def clean_up(s):
    """
    Cleans up numbers, URLs, and special characters from a string.

    Args:
        s: The string to be cleaned up.

    Returns:
        A string that has been cleaned up.
    """
    
    final = s.lower()
    #print(final)
    
    final = re.sub("http:\S+", " ", final)
    #print(final)
    
    final = re.findall("[a-z]+", final)
    #print(final)
    
    return ' '.join(final)

data["Reviews_clean"] = data["Review_Text"].apply(clean_up)
data.head()

def tokenize(s):
    """
    Tokenize a string.

    Args:
        s: String to be tokenized.

    Returns:
        A list of words as the result of tokenization.
    """
    
    return nltk.word_tokenize(s)

data["Reviews_clean"] = data["Reviews_clean"].apply(tokenize)
data.head()

def stem_and_lemmatize(l):
    """
    Perform stemming and lemmatization on a list of words.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after being stemmed and lemmatized.
    """
    
    ps = PorterStemmer()
    stemmed = [ps.stem(w) for w in l]
    
    lemmatizer = WordNetLemmatizer() 
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
    
    return lemmatized

data["Reviews_clean"] = data["Reviews_clean"].apply(stem_and_lemmatize)
data.head()

variable = stopwords.words("english")

def remove_stopwords(l):
    """
    Remove English stopwords from a list of strings.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after stop words are removed.
    """
    removing = [w for w in l if not w in variable]
    return removing

data["Reviews_clean"] = data["Reviews_clean"].apply(remove_stopwords)
data.head()

def re_blob(row):
    return " ".join(row['Reviews_clean'])

data['Reviews_clean'] = data.apply(re_blob, axis=1)
data.head()

#from nltk.sentiment import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')

#sia = SentimentIntensityAnalyzer()

#def is_positive(review):
#    if sia.polarity_scores(review)["compound"] > 0: # when the compound score is greater than 0 the reviewer is positive
#        return 1 # 1 for positive
#    return 0 # 0 for negative or neutral

#data["sentiment"] = data["Reviews_clean"].apply(is_positive)

#data["sentiment"].value_counts()

#def intensity(review):
#    return (abs(sia.polarity_scores(review)["compound"])+1)**2 # to give more importance the intense reviews we are going to square it

#data['intensity'] = data["Reviews_clean"].apply(intensity)

#data["intensity"].value_counts()


# Bag of words

#Creating a Bag of words:

list_words = []

for row in data["Reviews_clean"]:
    for word in row:
        list_words.append(word)

list_words = nltk.FreqDist(list_words)
#print(list_words)

top_words = list(list_words.keys())[:5000]
#print(top_words)

#let's take only the most common 1000 words
bow_vect = CountVectorizer(max_features = 1000)

# fit creates one entry for each different word seen  
x = bow_vect.fit_transform(data['Reviews_clean']).toarray()

df = pd.DataFrame(x, columns = bow_vect.get_feature_names())
df.shape

df.head()

#df["sentiment_number"] = data["sentiment"]
#df["intensity_number"] = data["intensity"]

y = data["Rating"].copy()

def wordCloud_generator(data, title=None):
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='black',
                          min_font_size = 10
                         ).generate(" ".join(data.values))                      
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.title(title,fontsize=30)
    plt.show()
    
wordCloud_generator(data['Review_Text'], title="Top words in reviews")


def wordCloud_generator(data, title=None):
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='black',
                          min_font_size = 10
                         ).generate(" ".join(data.values))                      
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.title(title,fontsize=30)
    plt.show()
    
wordCloud_generator(data['Reviews_clean'], title="Top words in reviews after clean")


# # LogisticRegression Model

# Instantiating a LogisticRegression Model (this is classification)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0, max_iter=10000)

# Splitting the datasets into training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df, y, train_size = 0.8, random_state = 0)

# Fitting our model
model.fit(x_train, y_train)

predicted = model.predict(x_test)

# evaluate (y_test == predicted)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predicted)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predicted)

from sklearn.metrics import r2_score
r2_score(y_test, predicted)


# # RandomForestClassifier

# Instantiating a RandomForest

from sklearn.ensemble import RandomForestClassifier

# Split the data

x_train, x_test, y_train, y_test = train_test_split(df, y, train_size = 0.8, random_state = 0)

# define models
forest = RandomForestClassifier(random_state=0)

forest.fit(x_train, y_train)

predicted = forest.predict(x_test)

accuracy_score(y_test, predicted)


confusion_matrix(y_test, predicted)


r2_score(y_test, predicted)


# # Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)

predicted = nb.predict(x_test)

accuracy_score(y_test, predicted)

confusion_matrix(y_test, predicted)

r2_score(y_test, predicted)


# # xgboost

import xgboost as xgb
from sklearn.metrics import accuracy_score
#https://xgboost.readthedocs.io/en/latest/parameter.html

x_train, x_test, y_train, y_test = train_test_split(df, y, train_size = 0.8, random_state = 0)

# specify parameters via map
param = {
    'booster': 'gbtree'
    ,'max_depth': 3
    ,'learning_rate': 0.3
    ,'subsample': 0.5
    ,'sample_type': 'uniform'
    #,'objective': 'binary:hinge'
    #,'obejective:'binary:logistic'
    ,'rate_drop': 0.0
    ,'n_estimators': 2000
    ,'verbosity':3
    #,'nthread': 5
}

d_train = xgb.DMatrix(x_train, y_train)
d_test = xgb.DMatrix(x_test, y_test)

clf = xgb.train(param, d_train)

# make prediction
preds = clf.predict(d_test)

# print accuracy score
r2_score(y_test, preds)


# # Get a new text review 

dataframe_newreview = pd.DataFrame(columns = ['Review_Text'])

dataframe_newreview['Review_Text'] = [input("Can you give us your review?")]

dataframe_newreview["Reviews_clean"] = dataframe_newreview['Review_Text'].apply(clean_up)
dataframe_newreview.head()

dataframe_newreview["Reviews_clean"] = dataframe_newreview["Reviews_clean"].apply(tokenize)
dataframe_newreview.head()

dataframe_newreview["Reviews_clean"] = dataframe_newreview["Reviews_clean"].apply(stem_and_lemmatize)
dataframe_newreview.head()

dataframe_newreview["Reviews_clean"] = dataframe_newreview["Reviews_clean"].apply(remove_stopwords)
dataframe_newreview.head()

dataframe_newreview["Reviews_clean"] = dataframe_newreview.apply(re_blob, axis = 1)
dataframe_newreview.head()

dataframe_newreview.drop(["Review_Text"], axis = 1)

#get the columns and one row of our bag of words

new_review = df.iloc[0]
new_review.values[:] = 0

#from our new review split and add as a list
list_new_review = dataframe_newreview["Reviews_clean"].str.split(" ")

#interate over the new review to get the frequency of each word
for word in list_new_review[0]:
    if word in list(new_review.index):
        new_review[word] = +1

#transform the new review into a dataframe and reset the index
new_review = pd.DataFrame(new_review).T
#new_review.reset_index()

#Predict the review using our model
print(model.predict(new_review))

# # Get a speech review

import pyaudio
import wave
import speech_recognition as sr


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

filename = "output.wav"

# initialize the recognizer
r = sr.Recognizer()

# open the file
with sr.AudioFile(filename) as source:
    # listen for the data (load audio to memory)
    audio_data = r.record(source)
    # recognize (convert from speech to text)
    text = r.recognize_google(audio_data)
print(text)

dataframe_newreview = pd.DataFrame(columns = ['Review_Text'])

dataframe_newreview['Review_Text'] = [text]

dataframe_newreview["Reviews_clean"] = dataframe_newreview['Review_Text'].apply(clean_up)
dataframe_newreview.head()

dataframe_newreview["Reviews_clean"] = dataframe_newreview["Reviews_clean"].apply(tokenize)
dataframe_newreview.head()

dataframe_newreview["Reviews_clean"] = dataframe_newreview["Reviews_clean"].apply(stem_and_lemmatize)
dataframe_newreview.head()

dataframe_newreview["Reviews_clean"] = dataframe_newreview["Reviews_clean"].apply(remove_stopwords)
dataframe_newreview.head()

dataframe_newreview["Reviews_clean"] = dataframe_newreview.apply(re_blob, axis = 1)
dataframe_newreview.head()

dataframe_newreview.drop(["Review_Text"], axis = 1)

#get the columns and one row of our bag of words

new_review = df.iloc[0]
new_review.values[:] = 0

#from our new review split and add as a list
list_new_review = dataframe_newreview["Reviews_clean"].str.split(" ")

#interate over the new review to get the frequency of each word
for word in list_new_review[0]:
    if word in list(new_review.index):
        new_review[word] = +1

#transform the new review into a dataframe and reset the index
new_review = pd.DataFrame(new_review).T
#new_review.reset_index()

#Predict the review using our model
display(model.predict(new_review))

# # Pickle

# Save all the necessary elements in files to share with a user

import pickle

# save the model to disk
filename = 'Model_for_review.sav'
pickle.dump(model, open(filename, 'wb'))
print("FINISHED")

# some time later...
 
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(x_test, y_test)
#print(result)





