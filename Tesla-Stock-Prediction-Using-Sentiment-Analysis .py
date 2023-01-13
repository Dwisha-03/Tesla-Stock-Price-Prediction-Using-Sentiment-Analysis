#!/usr/bin/env python
# coding: utf-8

# # Data Science & Business Analytics Internship

# GRIP: The Sparks Foundation
# 
# Author - Dwisha Mehta
# 
# Task 1 - Stock Market Prediction using Numerical and Textual Analysis
# 
# In this task, I created a time series model using LSTM for TESLA stock performance prediction using numerical analysis of historical stock prices, and sentimental analysis of news headlines.
# 
# Datasets Used:
# 
# Historical stock prices :https://finance.yahoo.com/ (Tesla)
# 
# Textual news headlines : https://bit.ly/36fFPI6

# In[13]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


pip install vaderSentiment


# Import Libraries

# In[44]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
from textblob import TextBlob
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
import seaborn as sns


# In[15]:


# reading the data
stock_headlines = pd.read_csv('/kaggle/input/india-headlines-news-dataset/india-news-headlines.csv')
stock_price = pd.read_csv('/kaggle/input/tesla-stock-price/TSLA (1).csv')


# In[16]:


# filtering the stock_price data
stock_price = stock_price[stock_price['Date'] > '2020-12-31']


# Data Cleaning

# In[17]:


# Drop missing values
stock_price.isna().sum() # there are no missing values

# Drop duplicates
stock_price = stock_price.drop_duplicates()
stock_headlines = stock_headlines.drop_duplicates()

# Convert object column to the date column 
stock_price['Date'] = pd.to_datetime(stock_price['Date'], format = '%Y-%m-%d')

# filter out the important columns in the dataset
stock_price = stock_price.filter(['Date', 'Close', 'Open', 'High', 'Low', 'Volume'])

# sort the data in ascending order
stock_price.sort_index(ascending = True, axis = 0)

# set the Date column to the index column
stock_price.set_index('Date', inplace=True)


# In[18]:


stock_price['2021':'2023'].plot(subplots=True, figsize=(10,12))
plt.title('Tesla stock attributes from 2020 to 2023')


# In[19]:


stock_headlines


# Data Cleaning

# In[20]:


# Drop missing values
stock_headlines.isna().sum() # there are no missing values

# Drop duplicates
stock_headlines = stock_headlines.drop_duplicates()

# convert the column to date column
stock_headlines['publish_date'] = pd.to_datetime(stock_headlines['publish_date'].astype(str), format='%Y%m%d')


# In[21]:


# renaming columns
stock_headlines.rename(columns = {'headline_text':'Headlines'}, inplace = True)
stock_headlines.rename(columns = {'publish_date':'Date'}, inplace = True)
# filter out the important columns in the dataset
stock_headlines = stock_headlines.filter(['Date', 'Headlines'])

# set the Date column to the index column
stock_headlines.set_index('Date', inplace=True)


# In[22]:


#grouping the data by date column
stock_headlines = stock_headlines.groupby(['Date'])['Headlines'].apply(lambda x:' '.join(x))


# Combining both the datasets

# In[23]:


stock_data = pd.merge(stock_price, stock_headlines, left_index=True, right_index=True)


# In[24]:


stock_data


# NLP: Removing Punctuations

# In[25]:


import string
def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct
stock_data['Headlines_wo_punc']=stock_data['Headlines'].apply(lambda x: remove_punctuation(x))


# Convert to lower_case and remove extra spacing

# In[26]:


stock_data['Headlines_wo_punc'] = stock_data['Headlines_wo_punc'].apply(str.lower)
stock_data['Headlines_wo_punc'] = stock_data['Headlines_wo_punc'].replace(r'\s+', ' ', regex=True)


# In[27]:


stopword = nltk.corpus.stopwords.words('english')
print(stopword[:11])


# NLP: Removing Stopwords

# In[28]:


def remove_stopwords(text):
    text = nltk.word_tokenize(text)
    text=[word for word in text if word not in stopword]
    return text
stock_data['Headlines_wo_punc_stopwords']=stock_data['Headlines_wo_punc'].apply(lambda x: remove_stopwords(x))


# NLP: Lemmatizing

# In[29]:


import nltk

lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(tokens):
    lemmatized_tokens = [lemmatizer.lemmatize(text) for text in tokens]
    return ' '.join(lemmatized_tokens)

stock_data['Headlines_wo_punc_stopwords_lemmatized']=stock_data['Headlines_wo_punc_stopwords'].apply(lemmatize_text)


# Remove irrelevant features

# In[30]:


stock_data = stock_data.filter(['Close', 'Open', 'High', 'Low', 'Headlines_wo_punc_stopwords_lemmatized'])


# In[31]:


stock_data.rename(columns = {'Headlines_wo_punc_stopwords_lemmatized':'headlines'}, inplace = True)


# Removing numerical values from the news headlines

# In[32]:


stock_data['headlines'] = stock_data['headlines'].str.replace('\d+','None')


# Sentiment Analysis on News Headlines

# In[33]:


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return  TextBlob(text).sentiment.polarity


# In[34]:


stock_data['Subjectivity'] = stock_data['headlines'].apply(getSubjectivity)
stock_data['Polarity'] = stock_data['headlines'].apply(getPolarity)
stock_data
     


# In[36]:


sia = SentimentIntensityAnalyzer()

stock_data['Compound'] = [sia.polarity_scores(v)['compound'] for v in stock_data['headlines']]


# In[37]:


stock_data['Negative'] = [sia.polarity_scores(v)['neg'] for v in stock_data['headlines']]


# In[38]:


stock_data['Neutral'] = [sia.polarity_scores(v)['neu'] for v in stock_data['headlines']]


# In[39]:


stock_data['Positive'] = [sia.polarity_scores(v)['pos'] for v in stock_data['headlines']]
stock_data


# Exploratory Data Analysis

# In[42]:


stock_data.describe()


# In[45]:


plt.figure(figsize=(10,5))
sns.heatmap(stock_data.corr(),cmap='Blues',annot=True)


# In[40]:


plt.figure(figsize = (10,6))
stock_data['Polarity'].hist(color = 'purple')


# In[41]:


plt.figure(figsize = (10,6))
stock_data['Subjectivity'].hist(color = 'blue')


# In[48]:


plt.figure(figsize=(10,5))
stock_data['Close'].plot()
plt.xlabel('Date')
plt.ylabel('Close Price')


# In[49]:


plt.figure(figsize=(12,8))
stock_data[stock_data.index > '2014-01-01']['Close'].plot()
stock_data[stock_data.index > '2014-01-01'].rolling(window=7).mean()['Close'].plot()


# In[50]:


plt.figure(figsize=(12,8))
stock_data[stock_data.index > '2014-01-01']['Close'].plot()
stock_data[stock_data.index > '2014-01-01'].rolling(window=30).mean()['Close'].plot()


# In[51]:


stock_data = stock_data.drop(['headlines'], axis = 1)


# Data Preparation for Modelling

# In[52]:


time_step = 1
training_size = int(len(stock_data) * 0.80)
test_size = len(stock_data) - training_size
train_data, test_data = stock_data.iloc[0:training_size, :], stock_data.iloc[training_size - time_step:len(stock_data), :]


# In[53]:


def create_dataset(data, time_step = 1):
    features = [data["Close"]]
    feature_names = ["Close"]
    for i in range(1, time_step + 1):
        feature = "cp_shifted_{k}".format(k = i)
        feature_names.append(feature)
        temp = data["Close"].shift((-1 * i))
        features.append(temp);
        
    features.append(data["Compound"]) 
    feature_names.append("Compound")
    for i in range(1, time_step + 1):
        feature = "compound_shifted_{k}".format(k = i)
        feature_names.append(feature)
        temp = data["Compound"].shift((-1 * i))
        features.append(temp);
    
    
    df = pd.concat(features, axis=1)
    df.columns = feature_names
    df = df.shift(periods = time_step)
    df = df.dropna()
    return df


# In[54]:


X_train = create_dataset(train_data, time_step)
X_test = create_dataset(test_data, time_step)


# In[55]:


y_train = X_train["cp_shifted_{t}".format(t = time_step)]
y_test = X_test["cp_shifted_{t}".format(t = time_step)]


# In[56]:


X_train.drop(columns = ["cp_shifted_{t}".format(t = time_step)], inplace = True)
X_test.drop(columns = ["cp_shifted_{t}".format(t = time_step)], inplace = True)


# In[57]:


X_train.shape, X_test.shape


# Scaling the target variable and feature dataset

# In[58]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

X_train = scaler.fit_transform(np.array(X_train))
y_train = scaler.fit_transform(np.array(y_train).reshape((len(y_train), 1)))

X_test = scaler.fit_transform(np.array(X_test))
y_test = scaler.fit_transform(np.array(y_test).reshape((len(y_test), 1)))


# In[59]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# Dividing dataset into training and testing

# In[60]:


# Reshaping the feature dataset for feeding into the model
X_train = X_train.reshape (X_train.shape + (1,)) 
X_test = X_test.reshape(X_test.shape + (1,))

# Printing the re-shaped feature dataset
print('Shape of Training set X:', X_train.shape)
print('Shape of Test set X:', X_test.shape)


# Modelling Stock Data

# In[61]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation


# In[62]:


# Creating the model architecture
model=Sequential()
model.add(LSTM(60,return_sequences=True,activation='tanh',input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(120,return_sequences=True,activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(240,activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(1))

# Printing the model summary
model.summary()


# In[63]:


# Compiling the model
model.compile(loss='mse' , optimizer='adam')

# Fitting the model using the training dataset
model.fit(X_train, y_train, validation_split=0.2, epochs=25, batch_size=16, verbose=1)


# Predictions

# In[64]:


predictions = model.predict(X_test) 

# Unscaling the predictions
predictions = scaler.inverse_transform(np.array(predictions).reshape((len(predictions), 1)))

# Printing the predictions
print('Predictions:')
predictions[0:5]


# In[65]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[66]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[68]:


import math


# In[69]:


from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[70]:


math.sqrt(mean_squared_error(y_test,test_predict))


# Model evaluation

# In[71]:


# Calculating the training mean-squared-error
train_loss = model.evaluate(X_train, y_train, batch_size = 1)

# Calculating the test mean-squared-error
test_loss = model.evaluate(X_test, y_test, batch_size = 1)

# Printing the training and the test mean-squared-errors
print('Train Loss =', round(train_loss,4))
print('Test Loss =', round(test_loss,4))


# In[72]:


from sklearn import metrics


# In[73]:


root_mean_square_error = np.sqrt(np.mean(np.power((y_test - predictions),2)))
print('Root Mean Square Error =', round(root_mean_square_error,4))


# In[74]:


rmse = metrics.mean_squared_error(y_test, predictions)
print('Root Mean Square Error (sklearn.metrics) =', round(np.sqrt(rmse),4))


# Plotting the predictions against unseen data

# In[75]:


X_test = scaler.inverse_transform(np.array(X_test).reshape((len(X_test), 3)))

# Unscaling y_test, y_train
y_train = scaler.inverse_transform(np.array(y_train).reshape((len(y_train), 1)))
y_test = scaler.inverse_transform(np.array(y_test).reshape((len(y_test), 1)))


# In[76]:


# Plotting
plt.figure(figsize=(16,10))

plt.plot(predictions, label="Predicted Close Price")
plt.plot([row[0] for row in y_test], label="Testing Close Price")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
plt.show()

