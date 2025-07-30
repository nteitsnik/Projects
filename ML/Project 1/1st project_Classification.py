# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:30:14 2024

@author: n.nteits
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
import matplotlib as plt
# Seaborn for plotting and styling
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA



#Read data

pd.set_option('display.precision', 3)
Bc=sklearn.datasets.load_breast_cancer()
#Get Column names
cols=Bc.feature_names
tv=pd.DataFrame(data=Bc.target,columns=['Outcome'])


#Unify dataframe
df=pd.DataFrame(data=Bc.data,columns=cols)
#df['Outcome']=Bc.target
pd.set_option("display.max_columns", None)
#Peak
print(df.head(10))
print(df.tail(10))
#Dimensions of the dataset
print(df.shape)
print(df.dtypes)

#Summary Statistics
df.describe()

#Pearson Correlaction Check
df.corr(method='pearson')

#df.iloc[:-1,:-1].corr(method='pearson')


#Skewness
df.skew()

#Class Distribution
tv.groupby('Outcome').size()




num_cols = len(df.columns)
#Histograms
# Calculate the number of rows and columns for a square-like layout
cols = int(np.ceil(np.sqrt(num_cols)))  # Number of columns in the grid
rows = int(np.ceil(num_cols / cols))  # Number of rows needed

# Create a figure and a set of subplots
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8 * cols, 6 * rows))

# Flatten axes to easily iterate over them
axes = axes.flatten()

# Loop through each column and plot a histogram
for i, column in enumerate(df.columns):
    axes[i].hist(df[column], bins=10, edgecolor='black')
    axes[i].set_title(f"Histogram of {column}", fontsize=30)  # Set smaller font size for title
    

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust the layout for better spacing
plt.tight_layout()

# Save the plot as an image
#fig.savefig("histograms_square.png")

# Show the plot
plt.show()


#Densityplots

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8 * cols, 6 * rows))

# Flatten axes to easily iterate over them
axes = axes.flatten()

# Loop through each column and plot a histogram
for i, column in enumerate(df.columns):
    sns.kdeplot(df[column],ax=axes[i], fill=True, color="blue")
    axes[i].set_title(f"Density of {column}", fontsize=30)  # Set smaller font size for title
    

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust the layout for better spacing
plt.tight_layout()

# Save the plot as an image
#fig.savefig("histograms_square.png")

# Show the plot
plt.show()



#Boxplot
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8 * cols, 6 * rows))

# Flatten axes to easily iterate over them
axes = axes.flatten()

# Loop through each column and plot a histogram
for i, column in enumerate(df.columns):
    sns.boxplot(df[column],ax=axes[i], color="blue")
    axes[i].set_title(f"Boxplotof {column}", fontsize=30)  # Set smaller font size for title
    

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust the layout for better spacing
plt.tight_layout()

# Save the plot as an image
#fig.savefig("histograms_square.png")

# Show the plot
plt.show()


#Alltogether
f = plt.figure(figsize=(18,8))
sns.boxplot(data=df)


#Remove
description=df.describe()
description[description.index=='75%'].values-description[description.index=='25%'].values
description.loc['IQR',:]=description[description.index=='75%'].values-description[description.index=='25%'].values
description.loc['Upper Fence',:]=description[description.index=='75%'].values + 5* description[description.index=='IQR'].values
description.loc['Outlier flag',:]=description[description.index=='Upper Fence'].values < description[description.index=='max'].values
columns_to_drop = description.loc[:, description.loc['Outlier flag'] == True].columns

stats_df = df.drop(columns_to_drop, axis=1)
#All but the problematic
f = plt.figure(figsize=(18,8))
sns.boxplot(data=stats_df)

#The problematic
f = plt.figure(figsize=(18,8))
sns.boxplot(data=df[columns_to_drop])



# Plot correlation matrix
correlations = df.corr()
sns.heatmap(correlations)



##Pair Grid
g = sns.PairGrid(df, diag_sharey=False)
g.map_offdiag(sns.scatterplot)  # Scatter plots for off-diagonal
g.map_diag(sns.kdeplot) 

for ax in g.axes.flat:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust figure size
g.fig.set_size_inches(14, 14)

plt.show()


##Algorithms



#Split
kfold = KFold(n_splits=10)




X = np.array(df.values)
Y=np.array(tv).ravel()

#Search between models and scalers
scalers=[None,Normalizer(),MinMaxScaler(feature_range=(0, 1)),StandardScaler()]
models=[LogisticRegression(solver='liblinear'), DecisionTreeClassifier(), GaussianNB(),LinearDiscriminantAnalysis()]
resultsdfac=pd.DataFrame()
resultsstdac=pd.DataFrame()

resultsdff1=pd.DataFrame()
resultsstdf1=pd.DataFrame()



for model in models :
    for scaler in scalers :
        if scaler :
            X_1=scaler.fit_transform(X)
        else:
            X_1 = X
            
        r1 = cross_val_score(model, X_1, Y, cv=kfold,scoring='accuracy')      
        resultsdfac.loc[model.__class__.__name__,scaler.__class__.__name__]=r1.mean()  
        resultsstdac.loc[model.__class__.__name__,scaler.__class__.__name__]=r1.std() 
        
      
        r1 = cross_val_score(model, X_1, Y, cv=kfold,scoring='f1')      
        resultsdff1.loc[model.__class__.__name__,scaler.__class__.__name__]=r1.mean()  
        resultsstdf1.loc[model.__class__.__name__,scaler.__class__.__name__]=r1.std() 


#Logistic Regression seems to be the way to go 
'''
clf = RandomForestClassifier(n_estimators=20)
param_grid = {"max_depth": [5, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid_search.fit(X_1, Y)
cross_val_score(grid_search, X_1, Y, cv=kfold) 

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
report(grid_search.cv_results_)

'''
#Let's tune parameters for Logistic regression with standard scaler

log_reg = LogisticRegression()
param_grid = {
    'C': np.logspace(-3, 3, 10),
    'penalty': ['l1', 'l2' , 'none'],
    'class_weight': [None, 'balanced'],
    'max_iter': [100, 200, 500, 1000],
    
}
scaler=StandardScaler()
X_1=scaler.fit_transform(X)
random_search = RandomizedSearchCV(estimator=log_reg, param_distributions=param_grid, 
                                   n_iter=100, scoring='f1', cv=5)

random_search.fit(X_1, Y)
random_search.best_params_
random_search.best_score_
best_model = random_search.best_estimator_

##train test split and cv on the training data only?
##Corelation
threshold = 0.9
correlation_matrix = df.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
X_reduced = df.drop(columns=to_drop)

X_1=scaler.fit_transform(X_reduced)
random_search = RandomizedSearchCV(estimator=log_reg, param_distributions=param_grid,n_iter=100, scoring='f1', cv=5)

random_search.fit(X_1, Y)
random_search.best_params_
random_search.best_score_

##PCA

X_1=scaler.fit_transform(X)
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_1)

random_search.fit(X_pca, Y)
random_search.best_score_
