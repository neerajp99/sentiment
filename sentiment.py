import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

# Method to get data from each file 
def get_data(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text    

###################
# Train data
###################
train_pos = os.listdir('data/train/pos') #12500 files
train_neg = os.listdir('data/train/neg')

###################
# Test data
###################
test_pos = os.listdir('data/test/pos')
test_neg = os.listdir('data/test/neg')


#######################
# Data Pre-processing
#######################

train_data = {"text" : [], "sentiment": []}
test_data = {"text": [], "sentiment": []}
pathname = os.getcwd() 
for i in range(len(train_pos)):
    # Fetch positive and negative sentimental data from both train and test set
    train_pos_text = get_data(str(pathname + '/data/train/pos/' + train_pos[i]))
    train_neg_text = get_data(str(pathname + '/data/train/neg/' + train_neg[i]))
    test_pos_text = get_data(str(pathname + '/data/test/pos/' + test_pos[i]))
    test_neg_text = get_data(str(pathname + '/data/test/neg/' + test_neg[i]))
    
    # Append the text and sentiment to the new train dataset
    train_data['text'].append(train_pos_text)
    # Positive sentiment as 1
    train_data['sentiment'].append(1)
    train_data['text'].append(train_neg_text)
    # Negative sentiment as 0
    train_data['sentiment'].append(0)
    
    # Append the text and sentiment to the new test dataset
    test_data['text'].append(test_pos_text)
    # Positive sentiment as 1
    test_data['sentiment'].append(1)
    test_data['text'].append(test_neg_text)
    # Negative sentiment as 0
    test_data['sentiment'].append(0)
    

# Creating pandas datafram for train and test data    
train_data_df = pd.DataFrame(train_data)
test_data_df = pd.DataFrame(test_data)


# Creating csv files for merged test and train data
test_data_df.to_csv(os.getcwd() + '/data/test_data.csv', index=False, header=True) # Test data
train_data_df.to_csv(os.getcwd() + '/data/train_data.csv', index=False, header=True) # Train data


# Method for cleaning the dataframe 
def clean_text_content(text):
    # Remove special characters
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    # Remove square brackers 
    text = re.sub('\[[^]]*\][.;:!\'?,\"()\[\]] ', '', text)
    # Converting the text to lowercase 
    text = text.lower()
    # Remove break elements from the text 
    text = re.sub("(<br\s*/><br\s*/>)|(\-)|(\/)", '', text)
    return text

# Cleaing the text part of the datframes 
train_data_df['text'] = train_data_df['text'].apply(clean_text_content)
test_data_df['text'] = test_data_df['text'].apply(clean_text_content)
train_data_df['text'][80]

# Tokenize the words 
train_data_df['text'] = train_data_df['text'].apply(word_tokenize)
test_data_df['text'] = test_data_df['text'].apply(word_tokenize)


#############################
# Creating bag of words model
############################# 
normalised_array_train = train_data_df['text']
normalised_array_test = test_data_df['text']
vectorizer = CountVectorizer()
count_train_vectorizer = vectorizer.fit_transform(normalised_array_train)
count_test_vectorizer = vectorizer.transform(normalised_array_test)

# Read the unique words
vectorizer.get_feature_names()

#######################
# Multinomial NB Model
#######################

# Fitting a Multinomial Naive Bayes Model 
X_train = count_train_vectorizer
Y_train = train_data_df['sentiment'].values
X_test = count_test_vectorizer
Y_test = test_data_df['sentiment'].values

mnb = MultinomialNB()
# Predicted values of sentiments
Y_prediction = mnb.fit(X_train, Y_train).predict(X_test)
# Getting the accuracy of the model
print('Accuracy: ', accuracy_score(Y_test, Y_prediction))
# 0.82c

# Getting the confusion matrix
print('Confusion Matrix:\n', confusion_matrix(Y_test, Y_prediction))

# Precision score
print('Precision Score: ', precision_score(Y_test, Y_prediction))

# Recall score
print('Recall Score: ', recall_score(Y_test, Y_prediction))

# TRUE POSITIVE


#######################
# Gaussian NB Model
#######################

# For Gaussian naive bayes, take a dense graph
# Fitting a Naive Bayes Model 
X_train_ = count_train_vectorizer.toarray()
Y_train_ = train_data_df['sentiment'].values
X_test_ = count_test_vectorizer.toarray()
Y_test_ = test_data_df['sentiment']
gnb = GaussianNB()
# Predicted values of sentiments
Y_prediction_ = gnb.fit(X_train_, Y_train).predict(X_test_)

# Getting the accuracy of the model
print('Accuracy: ', accuracy_score(Y_test_, Y_prediction_))
# 0.67

# Getting the confusion matrix
print('Confusion Matrix:\n', confusion_matrix(Y_test_, Y_prediction_))

# Precision score
print('Precision Score: ', precision_score(Y_test_, Y_prediction_))

# Recall score
print('Recall Score: ', recall_score(Y_test_, Y_prediction_))

# TRUE POSITIVE