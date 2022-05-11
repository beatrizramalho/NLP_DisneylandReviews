#!/usr/bin/env python
# coding: utf-8

#Import libraries:
import pandas as pd

import re

import nltk
#nltk.download('punkt')
#nltk.download('wordnet') # wordnet is the most well known lemmatizer for english
#nltk.download('stopwords')

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.probability import FreqDist

import pyaudio
import wave
import speech_recognition as sr

import pickle

#Upload data:

df = pd.read_csv("C:\\Users\\beatr\\Documents\\Beatriz\\Ironhack\\Projects\\NLP_DisneylandReviews\df.csv")
df.drop(["Unnamed: 0"], axis = 1, inplace = True)


#Function to treat the review

variable = stopwords.words("english")

def clean_review(review):
    
    review_clean = review.lower()
   
    review_clean = re.sub("http:\S+", " ", review_clean)
    
    review_clean = re.findall("[a-z]+", review_clean)
    
    ps = PorterStemmer()
    stemmed = [ps.stem(w) for w in nltk.word_tokenize(' '.join(review_clean))]
    
    lemmatizer = WordNetLemmatizer() 
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
    
    row = [word for word in lemmatized if not word in variable]
   
    return " ".join(row) 


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



dataframe_newreview["Reviews_clean"] = dataframe_newreview['Review_Text'].apply(clean_review)

dataframe_newreview.drop(["Review_Text"], axis = 1)


# load the model from disk

filename = 'Model_for_review.sav'
model = pickle.load(open(filename, 'rb'))



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




