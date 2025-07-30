import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GridSearchCV
import json
import random 
import os
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def save_classification_metrics(y_true, y_pred, save_path, average='weighted'):
    """
    Compute accuracy, precision, recall, and F1-score and save them as a JSON file.

    Args:
        y_true: Ground truth labels (array-like)
        y_pred: Predicted labels (array-like)
        save_path: Full path to the output JSON file
        average: Averaging method for precision/recall/f1 ('weighted', 'macro', etc.)
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "f1_score": f1_score(y_true, y_pred, average=average)
    }

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {save_path}")


random.seed(42)
np.random.seed(42)
scaler = MinMaxScaler()

folder_path = r"..\git_repos\DS_Test\IDS"

df_1 = pd.read_csv( os.path.join(folder_path,r'Data\UNSW_NB15_training-set.csv\UNSW_NB15_training-set.csv'),index_col=False )
df_2 = pd.read_csv(
    os.path.join(folder_path,r'Data\UNSW_NB15_testing-set.csv\UNSW_NB15_testing-set.csv'),
    index_col=False
)


df_all = pd.concat([df_1, df_2], ignore_index=True)

# Optionally shuffle to mix rows
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
df_all.drop(['attack_cat','id'],axis=1,inplace=True)


train_df, test_df = train_test_split(
    df_all,
    test_size=0.2,
    stratify=df_all["label"],
    random_state=42
)

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]
train_columns=X_train.columns

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

X_train_df = pd.DataFrame(X_train, columns=train_columns)
X_test_df  = pd.DataFrame(X_test,  columns=train_columns)

y_train_df = pd.Series(y_train, name="label")
y_test_df  = pd.Series(y_test,  name="label")


df_encoded_train = pd.get_dummies(X_train_df, columns=X_train_df.dtypes[X_train_df.dtypes== object].index)
df_encoded_train = df_encoded_train.astype({col: int for col in df_encoded_train.select_dtypes('bool').columns})

ls_train=y_train_df


df_encoded_train_scaled=pd.DataFrame(scaler.fit_transform(df_encoded_train),columns=df_encoded_train.columns)
train_columns = df_encoded_train.columns.tolist()





df_train_final=df_encoded_train_scaled.to_numpy()
y_train_labels = ls_train.to_numpy()



df_test_encoded = pd.get_dummies(
    X_test,
    columns=X_test.dtypes[X_test.dtypes == object].index
)

# Convert bools to int
df_test_encoded = df_test_encoded.astype({
    col: int for col in df_test_encoded.select_dtypes('bool').columns
})

# Drop 'id' column

# Reindex columns to match training columns
df_test_encoded = df_test_encoded.reindex(columns=train_columns, fill_value=0)

# Scale using the same scaler fitted on training data
df_test_encoded_scaled = pd.DataFrame(
    scaler.transform(df_test_encoded),
    columns=train_columns
)

# Optionally convert to NumPy array for model input
X_test_final = df_test_encoded_scaled.to_numpy()

# Convert labels to NumPy (if you have them)
y_test_final = y_test.to_numpy()


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(df_train_final, y_train_labels)

# 2️⃣ Create SelectFromModel
# This will select features with importance above the median
selector = SelectFromModel(
    rf,
    prefit=True,
    threshold="median"  # or e.g., "mean", or a float like 0.01
)

# 3️⃣ Transform the training data
X_train_selected = selector.transform(df_train_final)

feature_mask = selector.get_support()


#X_valid_selected = selector.transform(X_valid_final)
X_test_selected  = selector.transform(X_test_final)



from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification



base_estimator = DecisionTreeClassifier(max_depth=3)

# 4. Create AdaBoost classifier
ada = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=200,        # number of weak learners
    learning_rate=0.1,
    random_state=4
)

ada.fit(df_train_final, y_train_labels)





y_pred_ada = ada.predict(X_test_final)




save_classification_metrics(y_test_df, y_pred_ada, os.path.join(folder_path, r'Results\ada.json'), average='weighted')




ada.fit(X_train_selected, y_train_labels)

y_pred_ada_s = ada.predict(X_test_selected)


save_classification_metrics(y_test_df, y_pred_ada_s, os.path.join(folder_path, r'Results\ada_selected.json'), average='weighted')







import xgboost as xgb

param_grid = {
    "max_depth": [ 5,6,],
    "n_estimators": [ 200,300,400],
    "learning_rate": [ 0.15, 0.2,0.25],
    "subsample": [0.8, 1.0]
}

xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",  # or "multi:softmax" if multiclass
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)


grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    verbose=2,
    n_jobs=6
)

grid_search.fit(df_train_final, y_train_labels)


print("Best CV Accuracy:", grid_search.best_score_)

y_pred_gridxgb = grid_search.best_estimator_.predict(X_test_final)
print(grid_search.best_estimator_)
print(grid_search.best_params_)


save_classification_metrics(y_test_df, y_pred_gridxgb, os.path.join(folder_path, r'Results\xgb.json'), average='weighted')






grid_search.fit(X_train_selected, y_train_labels)


print("Best CV Accuracy:", grid_search.best_score_)

y_pred_gridxgb_s = grid_search.best_estimator_.predict(X_test_selected)
print(grid_search.best_estimator_)
print(grid_search.best_params_)



test_acc = accuracy_score(y_test_final, y_pred_gridxgb_s)

save_classification_metrics(y_test_df, y_pred_gridxgb_s, os.path.join(folder_path, r'Results\xgb_selected.json'), average='weighted')



from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier  # For FNN
from sklearn.linear_model import LogisticRegression

# Define base estimators
estimators = [
    ('rf', RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42,max_depth=6,n_estimators=300,learning_rate=0.2)),
    ('ada', AdaBoostClassifier(n_estimators=200,learning_rate=0.1, random_state=42)),  # Or use FNN below
    
    ('fnn', MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)),
     ('logreg', LogisticRegression(
        penalty='l1',
        solver='saga',
        C=0.01,
        max_iter=1000,
        random_state=42
    ))
]


# Define the meta-classifier (often a simple model like Logistic Regression)
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,  # 3-fold cross-validation for stacking
    n_jobs=-1
)

# Fit the stacking classifier
stacking_clf.fit(df_train_final, y_train_labels)

# Predict and evaluate
y_pred = stacking_clf.predict(X_test_final)
from sklearn.metrics import accuracy_score
print("Stacked model accuracy:", accuracy_score(y_test_final, y_pred))

save_classification_metrics(y_test_df, y_pred, os.path.join(folder_path, r'Results\Stacked.json'), average='weighted')

stacking_clf.fit(X_train_selected, y_train_labels)

# Predict and evaluate
y_pred = stacking_clf.predict(X_test_selected)
save_classification_metrics(y_test_df, y_pred, os.path.join(folder_path, r'Results\Stacked_selected.json'), average='weighted')


