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


stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

nltk.download('stopwords')

nltk.download('punkt')


nltk.data.path.append(r'C:/nltk_data')



fake_news = pd.read_csv(r'C:\Users\Γιώργος Μπόζιακας\Fake_news_data\Fake.csv')
true_news = pd.read_csv(r'C:\Users\Γιώργος Μπόζιακας\Fake_news_data\True.csv')

#drop empty news
fake_news=fake_news.drop(fake_news[fake_news['text']==' '].index,axis=0)
true_news=true_news.drop(true_news[true_news['text']==' '].index,axis=0)







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

def clean_text(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
        text = re.sub(r"\\W", " ", text)  # Remove non-word characters
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
        text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
        return text.strip()
data['text']=data['text'].apply(clean_text)

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

#clean again
textdata=textdata.drop(textdata[textdata['clean_text']==''].index,axis=0)
textdata = textdata.reset_index(drop=True)



logistic_model = LogisticRegression()
lasso_model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
lasso_model.__class__.__name__='lasso'
ridge_model = LogisticRegression(penalty='l2', C=0.1)
ridge_model.__class__.__name__='ridge'
dt_classifier = DecisionTreeClassifier(random_state=42)



models=[ MultinomialNB(),logistic_model,lasso_model,ridge_model,dt_classifier,svm.SVC()]





vectorizers=[CountVectorizer(binary=True)]
Y = textdata['Class']
resultsdfac=pd.DataFrame()
for vectorizer in vectorizers:  
    X_train, X_test, Y_train, Y_test = train_test_split(textdata['clean_text'], Y, test_size=0.2, random_state=2)
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

Y = textdata['Class']

for vectorizer in vectorizers:  
    X_train, X_test, Y_train, Y_test = train_test_split(textdata['clean_text'], Y, test_size=0.2, random_state=2)
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



#Word2vec





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
X_train, X_test, Y_train, Y_test = train_test_split(textdata['tokens'], Y, test_size=0.2, random_state=2)
w2vec = Word2Vec(sentences=X_train, vector_size=300, window=5, min_count=1, workers=6)

Y_train.reset_index(drop=True)
Y_test.reset_index(drop=True)


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




resultsdfac.to_excel("results_stop.xlsx")  

#Hyperparameter tuning for SVM
'''
vectorizer=CountVectorizer(binary=True)
X = vectorizer.fit_transform(textdata['clean_text'])
model=svm.SVC()


Y = textdata['Class']
Y = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y)

param_distributions = {
    'C': loguniform(1e-3, 1e3),  # Regularization parameter
    'kernel': ['linear', 'rbf','sigmoid'],  # Kernel type
    'gamma': loguniform(1e-4, 1e1)  # Kernel coefficient for rbf
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=50,  # Number of random configurations to try
    scoring='accuracy',  # Or another appropriate metric
    cv=5,  # Number of cross-validation folds
    verbose=2,
    n_jobs=-1,  # Use all available processors
    random_state=42
)

random_search.fit(X, Y)

print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)
random_search.best_params_

with open("random_search.best_score_svm.txt", "w") as file:
    pprint.pprint(random_search.best_score_, stream=file)
'''



#ridge parameter tuning

log_reg = LogisticRegression()

# Define parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear', 'saga'],  

    'max_iter': [100, 200, 300]
}

random_search = RandomizedSearchCV(
    estimator=log_reg,
    param_distributions=param_grid,
    n_iter=50,  # Number of random configurations to try
    scoring='accuracy',  # Or another appropriate metric
    cv=5,  # Number of cross-validation folds
    verbose=2,
    n_jobs=-1,  # Use all available processors
    random_state=2
)




vectorizer=CountVectorizer(binary=True)
X = vectorizer.fit_transform(textdata['clean_text'])
Y = textdata['Class']
Y = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y)




random_search.fit(X, Y)


print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)

# Prompt

prompt=['BREAKING: Donald Trump Announces Plan to Colonize Mars Former. President Donald Trump unveiled an ambitious plan today, declaring his intention to lead the charge in colonizing Mars. Speaking at a rally, he stated, “No one’s ever done Mars like we’re going to do it. It’ll be tremendous, believe me.” Trump claimed his new initiative, "Trump Galactic," would establish "the biggest, most luxurious Martian city ever." Critics dismissed the plan as unrealistic, but supporters hailed it as visionary. SpaceX founder Elon Musk declined to comment, fueling speculation about potential collaboration.Stay tuned for developments on this out-of-this-world endeavor. ']
dftry=pd.DataFrame(data=prompt,columns=['text'])

dftry['text']=dftry['text'].apply(clean_text)
dftry['tokens'] = dftry['text'].apply(lambda x: x.split())
dftry['tokens'] = dftry['tokens'].apply(remove_stopwords)
dftry['tokens'] = dftry['tokens'].apply(text_stemmer)
dftry['clean_text'] = dftry['tokens'].apply(lambda tokens: ' '.join(tokens))

X_test = vectorizer.transform(dftry['clean_text'])
model.predict(X_test)


#---------------------------  

ridge_model = LogisticRegression( C=0.1)
ridge_model.__class__.__name__='ridge'
vectorizers=CountVectorizer(binary=True)    

Y_1 = textdata['Class']
X_1 = textdata['clean_text']
X = vectorizer.fit_transform(X_1)


Y = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_1)
ridge_model.fit(X, Y)    



prompt=['A former Colorado Bureau of Investigation DNA scientist appeared in court Thursday to face criminal charges over data tampering that authorities said raises questions about the validity of more than 500 cases.Problems with the scientist’s work were found in cases involving homicide, sexual assault, robbery and other crimes, according to a law enforcement affidavit.In at least two cases, both homicides, the defendants received lesser sentences under plea deals than they could have faced if they went to trial because prosecutors were afraid Yvonne “Missy” Woods’ involvement could lead to acquittals.Woods was described as a “star analyst” by a former colleague who was interviewed by investigators, but also one who worked too fast and was “not the most thorough,” according to an internal affairs report.Authorities haven’t found any evidence of wrongful convictions, but prosecutors across the state are continuing to review the impacted cases.“This gets to the heart of whether or not science can be trusted, whether or not law enforcement can be trusted and quite frankly whether the judicial system can be trusted,” Jefferson County judge Graham Peper said during the short hearing.Woods allegedly told investigators at one point that she had changed data to complete cases more quickly, according to an arrest affidavit.Woods faces 52 counts of forgery, 48 counts of attempting to influence a public servant and one count each of perjury and cybercrime, for alleged misconduct between 2008 and 2023.The fallout from the alleged misconduct is still unfolding.In the most recent case to be impacted, Michael Shannel Jefferson was sentenced last week to 32 years in prison in the home invasion killing of Roger Dean in 1985. Jefferson was identified as a suspect in the cold case in 2021 through DNA evidence.']

dftry=pd.DataFrame(data=prompt,columns=['text'])

dftry['text']=dftry['text'].apply(clean_text)
dftry['tokens'] = dftry['text'].apply(lambda x: x.split())
dftry['tokens'] = dftry['tokens'].apply(remove_stopwords)
dftry['tokens'] = dftry['tokens'].apply(text_stemmer)
dftry['clean_text'] = dftry['tokens'].apply(lambda tokens: ' '.join(tokens))

X_test = vectorizer.transform(dftry['clean_text'])
prediction=ridge_model.predict(X_test)

prompt=['President Vladimir Putin has grown increasingly concerned about distortions in Russia s wartime economy, just as Donald Trump pushes for an end to the Ukraine conflict, five sources with knowledge of the situation told Reuters.Russia s economy, driven by exports of oil, gas and minerals, grew robustly over the past two years despite multiple rounds of Western sanctions imposed after its invasion of Ukraine in 2022.But domestic activity has become strained in recent months by labour shortages and high interest rates introduced to tackle inflation, which has accelerated under record military spending.That has contributed to the view within a section of the Russian elite that a negotiated settlement to the war is desirable, according to two of the sources familiar with thinking in the Kremlin.Trump, who returned to office on Monday, has vowed to swiftly resolve the Ukraine conflict, Europe s biggest since World War Two. This week he has said more sanctions, as well as tariffs, on Russia are likely unless Putin negotiates, adding that Russia was heading for "big trouble" in the economy. A senior Kremlin aide said on Tuesday that Russia had so far received no specific proposals for talks.']


dftry=pd.DataFrame(data=prompt,columns=['text'])

dftry['text']=dftry['text'].apply(clean_text)
dftry['tokens'] = dftry['text'].apply(lambda x: x.split())
dftry['tokens'] = dftry['tokens'].apply(remove_stopwords)
dftry['tokens'] = dftry['tokens'].apply(text_stemmer)
dftry['clean_text'] = dftry['tokens'].apply(lambda tokens: ' '.join(tokens))

X_test = vectorizer.transform(dftry['clean_text'])
prediction=ridge_model.predict(X_test)

prompt=['In a stunning press announcement, former President Donald Trump declared a "war on Mars," claiming the planet poses a threat to Earth.Mars has been stealing Earth’s energy for years—bad energy, folks," Trump stated, without providing evidence. He proposed creating a new military branch, the "Galactic Space Force," to counter this supposed threat.Critics dismissed the claim as science fiction, while supporters praised Trump’s "bold vision." Trump also hinted at secret backing from Elon Musk but offered no details.NASA officials, caught off guard, declined to comment, with one insider joking, "We’re focused on science, not space wars."Trump promised the initiative would be "tremendous," though experts remain skeptical.']

dftry=pd.DataFrame(data=prompt,columns=['text'])

dftry['text']=dftry['text'].apply(clean_text)
dftry['tokens'] = dftry['text'].apply(lambda x: x.split())
dftry['tokens'] = dftry['tokens'].apply(remove_stopwords)
dftry['tokens'] = dftry['tokens'].apply(text_stemmer)
dftry['clean_text'] = dftry['tokens'].apply(lambda tokens: ' '.join(tokens))

X_test = vectorizer.transform(dftry['clean_text'])
prediction=ridge_model.predict(X_test)


prompt=['In a shocking revelation during a livestream, Kanye West unveiled his latest vision: creating a self-sustaining city called “Yeezy City” in the middle of the Nevada desert. The city, described by Kanye as “the future of human innovation,” will reportedly feature futuristic Yeezy-designed homes, a music production hub, and a museum dedicated entirely to Kanye’s career.Yeezy City will be a utopia where creativity knows no limits,” West stated. He also hinted at plans for a cryptocurrency called “YeezyCoin” to power the local economy.Critics have questioned the feasibility of the project, while fans have already started online petitions to move there. When asked about timelines, Kanye confidently replied, “We’re breaking ground next year. Elon’s already on board.”The announcement has sparked a social media frenzy, with many wondering if Yeezy City could become the next Silicon Valley—or just another Kanye dream.']
dftry=pd.DataFrame(data=prompt,columns=['text'])

dftry['text']=dftry['text'].apply(clean_text)
dftry['tokens'] = dftry['text'].apply(lambda x: x.split())
dftry['tokens'] = dftry['tokens'].apply(remove_stopwords)
dftry['tokens'] = dftry['tokens'].apply(text_stemmer)
dftry['clean_text'] = dftry['tokens'].apply(lambda tokens: ' '.join(tokens))

X_test = vectorizer.transform(dftry['clean_text'])
model.predict(X_test)

prompt=['via']
dftry=pd.DataFrame(data=prompt,columns=['text'])

dftry['text']=dftry['text'].apply(clean_text)
dftry['tokens'] = dftry['text'].apply(lambda x: x.split())
dftry['tokens'] = dftry['tokens'].apply(remove_stopwords)
dftry['tokens'] = dftry['tokens'].apply(text_stemmer)
dftry['clean_text'] = dftry['tokens'].apply(lambda tokens: ' '.join(tokens))

X_test = vectorizer.transform(dftry['clean_text'])
ridge_model.predict(X_test)
