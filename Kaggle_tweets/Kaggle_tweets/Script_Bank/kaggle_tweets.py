# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:15:14 2025

@author: n.nteits
"""
import kagglehub
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
import tqdm
import pprint
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV




train = pd.read_csv(r'C:\Users\Γιώργος Μπόζιακας\train.csv')
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

nltk.download('stopwords')

os.environ["OMP_NUM_THREADS"] = "1"

nltk.download('punkt')


nltk.data.path.append(r'C:/nltk_data')


train.isnull().sum()
print(train.groupby('target').count())
print(train.groupby('keyword').count())

train['text']=train['text'].str.lower()
train['keyword']=train['keyword'].str.lower()



def clean_text(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
        text = re.sub(r"\\W", " ", text)  # Remove non-word characters
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
        text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
        return text.strip()

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]
   
def text_stemmer(tokens):
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens
  
train['text']=train['text'].apply(clean_text)  
train['tokens'] = train['text'].apply(lambda x: x.split())
train['tokens'] = train['tokens'].apply(remove_stopwords)
train['tokens'] = train['tokens'].apply(text_stemmer)
train = train.sample(frac=1, random_state=1).reset_index(drop=True)

train['clean_text'] = train['tokens'].apply(lambda tokens: ' '.join(tokens))



logistic_model = LogisticRegression()
lasso_model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
lasso_model.__class__.__name__='lasso'
ridge_model = LogisticRegression(penalty='l2', C=0.1)
ridge_model.__class__.__name__='ridge'
dt_classifier = DecisionTreeClassifier(random_state=42)



models=[ MultinomialNB(),logistic_model,lasso_model,ridge_model,dt_classifier,svm.SVC()]




vectorizers=[CountVectorizer(binary=True)]
Y = train['target']
resultsdfac=pd.DataFrame()
for vectorizer in vectorizers:  
    X_train, X_test, Y_train, Y_test = train_test_split(train['clean_text'], Y, test_size=0.2, random_state=2)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    Y_train = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_train)
    Y_test = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_test)
    for model in models:
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        if model==lasso_model :
            resultsdfac.loc['lasso','Binary'] = accuracy_score(Y_test, Y_pred)
        elif model==ridge_model:
            resultsdfac.loc['ridge','Binary']= accuracy_score(Y_test, Y_pred)
        elif  model==logistic_model:
            resultsdfac.loc['logistic','Binary'] = accuracy_score(Y_test, Y_pred) 
        else:
            resultsdfac.loc[model.__class__.__name__,'Binary']= accuracy_score(Y_test, Y_pred) 


vectorizers=[CountVectorizer(binary=False),TfidfVectorizer()]
''',svm.SVC()'''
models=[ MultinomialNB(),logistic_model,lasso_model,ridge_model,dt_classifier,svm.SVC()]

Y = train['target']

for vectorizer in vectorizers:  
    X_train, X_test, Y_train, Y_test = train_test_split(train['clean_text'], Y, test_size=0.2, random_state=2)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    Y_train = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_train)
    Y_test = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_test)
    for model in models:
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        if model==lasso_model :
            resultsdfac.loc['lasso',vectorizer.__class__.__name__] = accuracy_score(Y_test, Y_pred)
        elif model==ridge_model:
            resultsdfac.loc['ridge',vectorizer.__class__.__name__]= accuracy_score(Y_test, Y_pred)
        elif  model==logistic_model:
            resultsdfac.loc['logistic',vectorizer.__class__.__name__] = accuracy_score(Y_test, Y_pred) 
        else:
            resultsdfac.loc[model.__class__.__name__,vectorizer.__class__.__name__]= accuracy_score(Y_test, Y_pred) 




def get_average_word2vec(tokens_list, w2vec, vector_size=300):
  
    # Filter out words not in the Word2Vec vocabulary
    valid_words = [w2vec.wv[word] for word in tokens_list if word in w2vec.wv]

    # If no valid words, return a zero vector
    if not valid_words:
        return np.zeros(vector_size)
    
    # Calculate the mean of valid word vectors
    tmp = np.vstack(valid_words)  # Stack vectors vertically
    result = np.mean(tmp, axis=0)  # Calculate mean across rows
    return result
    # If no valid words, return a zero vector of the desired size

models=[ logistic_model,lasso_model,ridge_model,dt_classifier,svm.SVC()]  
X_train, X_test, Y_train, Y_test = train_test_split(train['tokens'], Y, test_size=0.2, random_state=2)
w2vec = Word2Vec(sentences=X_train, vector_size=300, window=5, min_count=1, workers=6)


Train_trans=np.zeros((len(X_train),300))
Test_trans=np.zeros((len(X_test),300))
i=0
for idx in X_train.index :     
    Train_trans[i,:] = get_average_word2vec(X_train[idx], w2vec) 
    i=i+1
       
i=0
for idx in X_test.index :   
    
    Test_trans[i,:] = get_average_word2vec(X_test[idx], w2vec) 
    i=i+1
    
Y_train = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_train)
Y_test = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_test)
for model in models:
        model.fit(Train_trans, Y_train)
        Y_pred = model.predict(Test_trans)
        if model==lasso_model :
            resultsdfac.loc['lasso','word2vec'] = accuracy_score(Y_test, Y_pred)
        elif model==ridge_model:
            resultsdfac.loc['ridge','word2vec']= accuracy_score(Y_test, Y_pred)
        elif  model==logistic_model:
            resultsdfac.loc['logistic','word2vec'] = accuracy_score(Y_test, Y_pred) 
        else:
            resultsdfac.loc[model.__class__.__name__,'word2vec']= accuracy_score(Y_test, Y_pred) 

model= MultinomialNB()
param_grid = {
    'classifier__alpha': [0.001,0.01,0.1, 0.5, 1.0, 2.0, 5.0]  # Try different alpha values
}


pipeline = Pipeline([
    ('vectorizer', CountVectorizer(binary=True)),  # Convert text to numerical features
    ('classifier', BernoulliNB())     # Multinomial Naïve Bayes classifier
])

# Perform GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(train['text'], train['target'])

print("Best accuracy:", grid_search.best_score_)


#tf-iDF with 1-2grams
vectorizers = [TfidfVectorizer(ngram_range=(1, 2))]
models=[ MultinomialNB(),logistic_model,lasso_model,ridge_model,dt_classifier,svm.SVC()]

Y = train['target']
for vectorizer in vectorizers:  
    X_train, X_test, Y_train, Y_test = train_test_split(train['clean_text'], Y, test_size=0.2, random_state=2)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    Y_train = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_train)
    Y_test = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_test)
    for model in models:
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        if model==lasso_model :
            resultsdfac.loc['lasso','bi tfidf'] = accuracy_score(Y_test, Y_pred)
        elif model==ridge_model:
            resultsdfac.loc['ridge','bi tfidf']= accuracy_score(Y_test, Y_pred)
        elif  model==logistic_model:
            resultsdfac.loc['logistic','bi tfidf'] = accuracy_score(Y_test, Y_pred) 
        else:
            resultsdfac.loc[model.__class__.__name__,'bi tfidf']= accuracy_score(Y_test, Y_pred) 

def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            embeddings_index[word] = vector
    return embeddings_index

# Use the 100D embeddings
glove_path = r"./Python_examples/glove.6B.100d.txt"  # Adjust if needed
glove_embeddings = load_glove_embeddings(glove_path)

# Test: Get vector for "learning"
print(glove_embeddings.get("learning", "Word not found")[:5])

tweets = ["I love data science", "Machine learning is great", "NLP is fun", "Deep learning for NLP"]
labels = [1, 0, 1, 0]

# Convert tweets into embeddings (average word vectors)
def get_tweet_embedding(tweet, embeddings_dict, dim=100):
    words = tweet.split()
    vectors = [embeddings_dict[word] for word in words if word in embeddings_dict]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

X_train, X_test, Y_train, Y_test = train_test_split(train['clean_text'], Y, test_size=0.2, random_state=2)



Train_trans=np.zeros((len(X_train),100))
Test_trans=np.zeros((len(X_test),100))


i=0
for idx in X_train.index :     
    Train_trans[i,:] = np.array(get_tweet_embedding(X_train[idx], glove_embeddings, dim=100)) 
    i=i+1
       
i=0
for idx in X_test.index :   
    
    Test_trans[i,:] = np.array(get_tweet_embedding(X_test[idx], glove_embeddings, dim=100)) 
    i=i+1


models=[ logistic_model,lasso_model,ridge_model,dt_classifier,svm.SVC()]  
for model in models:
        model.fit(Train_trans, Y_train)
        Y_pred = model.predict(Test_trans)
        if model==lasso_model :
            resultsdfac.loc['lasso','Glove 100D'] = accuracy_score(Y_test, Y_pred)
        elif model==ridge_model:
            resultsdfac.loc['ridge','Glove 100D']= accuracy_score(Y_test, Y_pred)
        elif  model==logistic_model:
            resultsdfac.loc['logistic','Glove 100D'] = accuracy_score(Y_test, Y_pred) 
        else:
            resultsdfac.loc[model.__class__.__name__,'Glove 100D']= accuracy_score(Y_test, Y_pred) 


#300d
glove_path = r"./Python_examples/glove.6B.300d.txt"  # Adjust if needed
glove_embeddings = load_glove_embeddings(glove_path)

def get_tweet_embedding(tweet, embeddings_dict, dim=300):
    words = tweet.split()
    vectors = [embeddings_dict[word] for word in words if word in embeddings_dict]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

X_train, X_test, Y_train, Y_test = train_test_split(train['clean_text'], Y, test_size=0.2, random_state=2)



Train_trans=np.zeros((len(X_train),300))
Test_trans=np.zeros((len(X_test),300))


i=0
for idx in X_train.index :     
    Train_trans[i,:] = np.array(get_tweet_embedding(X_train[idx], glove_embeddings, dim=300)) 
    i=i+1
       
i=0
for idx in X_test.index :   
    
    Test_trans[i,:] = np.array(get_tweet_embedding(X_test[idx], glove_embeddings, dim=300)) 
    i=i+1


models=[ logistic_model,lasso_model,ridge_model,dt_classifier,svm.SVC()]  
for model in models:
        model.fit(Train_trans, Y_train)
        Y_pred = model.predict(Test_trans)
        if model==lasso_model :
            resultsdfac.loc['lasso','Glove 300D'] = accuracy_score(Y_test, Y_pred)
        elif model==ridge_model:
            resultsdfac.loc['ridge','Glove 300D']= accuracy_score(Y_test, Y_pred)
        elif  model==logistic_model:
            resultsdfac.loc['logistic','Glove 300D'] = accuracy_score(Y_test, Y_pred) 
        else:
            resultsdfac.loc[model.__class__.__name__,'Glove 300D']= accuracy_score(Y_test, Y_pred) 

#Universal Sentense Encoder
import tensorflow_hub as hub
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def get_tweet_embedding(tweet):
    return use_model([tweet]).numpy()[0]

X_train, X_test, Y_train, Y_test = train_test_split(train['clean_text'], Y, test_size=0.2, random_state=2)



Train_trans=np.zeros((len(X_train),512))
Test_trans=np.zeros((len(X_test),512))


i=0
for idx in X_train.index :     
    Train_trans[i,:] = np.array(get_tweet_embedding(X_train[idx]))
    i=i+1
       
i=0
for idx in X_test.index :   
    
    Test_trans[i,:] = np.array(get_tweet_embedding(X_test[idx]))
    i=i+1

models=[ logistic_model,lasso_model,ridge_model,dt_classifier,svm.SVC()]  
for model in models:
        model.fit(Train_trans, Y_train)
        Y_pred = model.predict(Test_trans)
        if model==lasso_model :
            resultsdfac.loc['lasso','USE'] = accuracy_score(Y_test, Y_pred)
        elif model==ridge_model:
            resultsdfac.loc['ridge','USE']= accuracy_score(Y_test, Y_pred)
        elif  model==logistic_model:
            resultsdfac.loc['logistic','USE'] = accuracy_score(Y_test, Y_pred) 
        else:
            resultsdfac.loc[model.__class__.__name__,'USE']= accuracy_score(Y_test, Y_pred) 

# Load the Transformer-based Universal Sentence Encoder
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

def get_tweet_embedding(tweet):
    return use_model([tweet]).numpy()[0]

X_train, X_test, Y_train, Y_test = train_test_split(train['clean_text'], Y, test_size=0.2, random_state=2)



Train_trans=np.zeros((len(X_train),512))
Test_trans=np.zeros((len(X_test),512))


i=0
for idx in X_train.index :     
    Train_trans[i,:] = np.array(get_tweet_embedding(X_train[idx]))
    i=i+1
       
i=0
for idx in X_test.index :   
    
    Test_trans[i,:] = np.array(get_tweet_embedding(X_test[idx]))
    i=i+1

models=[ logistic_model,lasso_model,ridge_model,dt_classifier,svm.SVC()]  
for model in models:
        model.fit(Train_trans, Y_train)
        Y_pred = model.predict(Test_trans)
        if model==lasso_model :
            resultsdfac.loc['lasso','USE_T'] = accuracy_score(Y_test, Y_pred)
        elif model==ridge_model:
            resultsdfac.loc['ridge','USE_T']= accuracy_score(Y_test, Y_pred)
        elif  model==logistic_model:
            resultsdfac.loc['logistic','USE_T'] = accuracy_score(Y_test, Y_pred) 
        else:
            resultsdfac.loc[model.__class__.__name__,'USE_T']= accuracy_score(Y_test, Y_pred) 

#Bert
from transformers import AutoTokenizer, AutoModel
import torch

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModel.from_pretrained("vinai/bertweet-base")

def get_tweet_embedding(tweet):
    tokens = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        output = model(**tokens)  # Get BERT output
    
    word_embeddings = output.last_hidden_state # Use CLS token embedding
    return torch.mean(word_embeddings, dim=1) 

X_train, X_test, Y_train, Y_test = train_test_split(train['clean_text'], Y, test_size=0.2, random_state=2)



Train_trans=np.zeros((len(X_train),768))
Test_trans=np.zeros((len(X_test),768))


i=0
for idx in X_train.index :     
    Train_trans[i,:] = np.array(get_tweet_embedding(X_train[idx]))
    i=i+1
       
i=0
for idx in X_test.index :   
    
    Test_trans[i,:] = np.array(get_tweet_embedding(X_test[idx]))
    i=i+1
    
models=[ logistic_model,lasso_model,ridge_model,dt_classifier,svm.SVC()]  
for model in models:
        model.fit(Train_trans, Y_train)
        Y_pred = model.predict(Test_trans)
        if model==lasso_model :
            resultsdfac.loc['lasso','Bert_mean'] = accuracy_score(Y_test, Y_pred)
        elif model==ridge_model:
            resultsdfac.loc['ridge','Bert_mean']= accuracy_score(Y_test, Y_pred)
        elif  model==logistic_model:
            resultsdfac.loc['logistic','Bert_mean'] = accuracy_score(Y_test, Y_pred) 
        else:
            resultsdfac.loc[model.__class__.__name__,'Bert_mean']= accuracy_score(Y_test, Y_pred) 

#param grid on svc
train = pd.read_csv(r'C:\Users\Γιώργος Μπόζιακας\train.csv')
model= svm.SVC()
param_grid = {
    "C": [0.05,0.1,0.3,0.2,0.5],         # Regularization strength
    "kernel": ["linear", "rbf"],  # Kernel types
    "gamma": ["scale", "auto"]   # Kernel coefficient for 'rbf'
}


use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

def get_tweet_embedding(tweet):
    return use_model([tweet]).numpy()[0]

train['text']=train['text'].apply(clean_text)  
train['tokens'] = train['text'].apply(lambda x: x.split())
train['tokens'] = train['tokens'].apply(remove_stopwords)
train['tokens'] = train['tokens'].apply(text_stemmer)
train = train.sample(frac=1, random_state=1).reset_index(drop=True)

train['clean_text'] = train['tokens'].apply(lambda tokens: ' '.join(tokens))

Train_trans=np.zeros((len(train),512))
Y=train['target']


i=0
for idx in train.index :     
    Train_trans[i,:] = np.array(get_tweet_embedding(train.loc[idx,'clean_text']))
    i=i+1
       



# Perform GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(Train_trans, Y)

print("Best accuracy:", grid_search.best_score_)
grid_search.best_params_
