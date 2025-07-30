# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:13:44 2024

@author: n.nteits
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
# Seaborn for plotting and styling
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_diabetes

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from scipy.stats import uniform

#Load data
pd.set_option('display.precision', 3)

Bc=sklearn.datasets.load_diabetes()
cols=Bc.feature_names
tv=pd.DataFrame(data=Bc.target,columns=['Outcome'])
df=pd.DataFrame(data=Bc.data,columns=cols)

pd.set_option("display.max_columns", None)

##Peak

print(df.head(10))
print(df.tail(10))

#Dimensions of the dataset
print(df.shape)
print(df.dtypes)
print(tv.dtypes)

#Summary Statistics
df.describe()
tv.describe()

#Pearson Correlaction Check
df.corr(method='pearson')

#Skewness
df.skew()



#Histograms
# Calculate the number of rows and columns for a square-like layout
num_cols = len(df.columns)
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


#Plot of the target value
plt.hist(tv, bins=10, edgecolor='black')


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

#Altogether
f = plt.figure(figsize=(18,8))
sns.boxplot(data=df)

#Remove
description=df.describe()
description[description.index=='75%'].values-description[description.index=='25%'].values
description.loc['IQR',:]=description[description.index=='75%'].values-description[description.index=='25%'].values
description.loc['Upper Fence',:]=description[description.index=='75%'].values + 2.5* description[description.index=='IQR'].values
description.loc['Outlier flag',:]=description[description.index=='Upper Fence'].values < description[description.index=='max'].values
columns_to_drop = description.loc[:, description.loc['Outlier flag'] == True].columns

stats_df = df.drop(columns_to_drop, axis=1)
#All but the problematic
f = plt.figure(figsize=(18,8))
sns.boxplot(data=stats_df)

#the problematic
f = plt.figure(figsize=(18,8))
sns.boxplot(data=df[columns_to_drop])

# Plot correlation matrix
correlations = df.corr()
sns.heatmap(correlations)

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


#Data is on the same scale so no need for scaler selection
kfold = KFold(n_splits=10)




X = np.array(df.values)
Y=np.array(tv).ravel()

#Search between models and scalers
scalers=[None,Normalizer(),MinMaxScaler(feature_range=(0, 1)),StandardScaler()]
models=[LinearRegression(), Ridge(), Lasso(),ElasticNet(),KNeighborsRegressor(),DecisionTreeRegressor()]
resultsdfac=pd.DataFrame()
resultsstdac=pd.DataFrame()

resultsdff1=pd.DataFrame()
resultsstdf1=pd.DataFrame()

X = np.array(df.values)
Y=np.array(tv).ravel()

for model in models :
    for scaler in scalers :
        if scaler :
            X_1=scaler.fit_transform(X)
        else:
            X_1 = X
            
        r1 = cross_val_score(model, X_1, Y, cv=kfold,scoring='neg_mean_squared_error')      
        resultsdfac.loc[model.__class__.__name__,scaler.__class__.__name__]=r1.mean()  
        resultsstdac.loc[model.__class__.__name__,scaler.__class__.__name__]=r1.std() 
        
      
        
r1.mean()

#So Lasso with Standard Scales seems to be the best pick

param_grid = {
    'alpha': np.logspace(-6, 6, 13),      # Regularization strength
   
        # Tolerance for stopping criteria
    
   
}

lasso = Lasso()
scaler=StandardScaler()
X_1=scaler.fit_transform(X)
random_search = RandomizedSearchCV(lasso, param_distributions=param_grid,  cv=5,scoring='r2')
random_search.fit(X_1, Y)
random_search.best_score_
best_model = random_search.best_estimator_

param_grid = {
    'alpha': uniform(0, 1),      # Regularization strength
   
        # Tolerance for stopping criteria
    
   
}

lasso = Lasso()
scaler=StandardScaler()
X_1=scaler.fit_transform(X)
random_search = RandomizedSearchCV(lasso, param_distributions=param_grid,  cv=5,scoring='r2')
random_search.fit(X_1, Y)
random_search.best_score_
best_model = random_search.best_estimator_



#Correlaction


X_1=scaler.fit_transform(X)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_1)


random_search.fit(X_pca, Y)
random_search.best_score_


