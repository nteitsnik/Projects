# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:29:31 2025

@author: n.nteits
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:29:31 2025

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

nltk.download('stopwords')

os.environ["OMP_NUM_THREADS"] = "1"

nltk.download('punkt')


nltk.data.path.append(r'C:/nltk_data')



fake_news = pd.read_csv(r'C:\Users\Γιώργος Μπόζιακας\Fake_news_data\Fake.csv')
true_news = pd.read_csv(r'C:\Users\Γιώργος Μπόζιακας\Fake_news_data\True.csv')
fake_df=pd.DataFrame(data=fake_news._data)



fake_news.head(5)
true_news.head(5)

#null check
fake_news.isnull().sum()
true_news.isnull().sum()

#no missing data
fake_news['Class']=1
true_news['Class']=0

#Remove the Reueters tag and back from real news
true_news["text"] = true_news["text"].str.replace(r".*?\(Reuters\) -", "", regex=True)
#create a unified dataframe
cols=fake_news.columns.values
cols=np.append(cols,'Class')
print(cols)
news=pd.DataFrame(columns=cols)
news=pd.concat([fake_news,true_news])
'''
for i in range(len(news)):
    index = news[news['text'].str.contains('www.', case=False, na=False)].index
'''
news.shape

print(news.groupby('Class').count())


news.iloc[40000]


data=news[['text','Class']]
data.head()
'''
X = news.drop('Class',axis=1)
y = news['Class']
'''
#Clean text

#lowercase
data['text']=data['text'].str.lower()
data = data.reset_index(drop=True)
#remove special characters
for i in range(len(data)):
    
    data.loc[i,'text'] = re.sub(r'[^a-zA-Z0-9\s]', '', data.loc[i,'text'])
    data.loc[i,'text'] = re.sub('\[.*?\]','',data.loc[i,'text'])
    data.loc[i,'text'] = re.sub("\\W"," ",data.loc[i,'text'])
    data.loc[i,'text'] = re.sub('https?://\S+|www\.\S+','',data.loc[i,'text'])
    data.loc[i,'text'] = re.sub('<.*?>+',b'',data.loc[i,'text'])
    #data.loc[i,'text'] = re.sub('[%s]' % re.escape(string.punctuation),'',data.loc[i,'text'])
    data.loc[i,'text'] = re.sub('\w*\d\w*','',data.loc[i,'text'])




textdata=pd.DataFrame(columns=['text','Class'])
for i in range(len(data)):
    textdata.loc[i,'text'] = data.loc[i,'text']
    textdata.loc[i,'Class']=int(data.iloc[i, data.columns.get_loc('Class')])
   
textdata['tokens'] = textdata['text'].apply(lambda x: x.split())

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

textdata['tokens'] = textdata['tokens'].apply(remove_stopwords)
   
def text_stemmer(tokens):
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens
    

textdata['tokens'] = textdata['tokens'].apply(text_stemmer)

#shuffle
textdata = textdata.sample(frac=1, random_state=1).reset_index(drop=True)
textdata['clean_text'] = textdata['tokens'].apply(lambda tokens: ' '.join(tokens))


logistic_model = LogisticRegression()
lasso_model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')

ridge_model = LogisticRegression(penalty='l2', C=0.1)


vectorizers=[CountVectorizer(),TfidfVectorizer()]
''',svm.SVC()'''
models=[ MultinomialNB(),logistic_model,lasso_model,ridge_model]

Y = textdata['Class']
resultsdfac=pd.DataFrame()
for vectorizer in vectorizers:
    X = vectorizer.fit_transform(textdata['clean_text'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
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


