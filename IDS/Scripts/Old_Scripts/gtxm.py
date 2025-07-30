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


train_df, temp_df = train_test_split(
    df_all,
    test_size=0.3,
    stratify=df_all["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]
train_columns=X_train.columns

# Same for test and val
X_val = val_df.drop(columns=["label"])
y_val = val_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

X_train_df = pd.DataFrame(X_train, columns=train_columns)
X_valid_df = pd.DataFrame(X_val, columns=train_columns)
X_test_df  = pd.DataFrame(X_test,  columns=train_columns)

y_train_df = pd.Series(y_train, name="label")
y_valid_df = pd.Series(y_val, name="label")
y_test_df  = pd.Series(y_test,  name="label")


df_encoded_train = pd.get_dummies(X_train_df, columns=X_train_df.dtypes[X_train_df.dtypes== object].index)
df_encoded_train = df_encoded_train.astype({col: int for col in df_encoded_train.select_dtypes('bool').columns})

ls_train=y_train_df


df_encoded_train_scaled=pd.DataFrame(scaler.fit_transform(df_encoded_train),columns=df_encoded_train.columns)
train_columns = df_encoded_train.columns.tolist()





df_train_final=df_encoded_train_scaled.to_numpy()
y_train_labels = ls_train.to_numpy()



df_valid_encoded = pd.get_dummies(
    X_val,
    columns=X_val.dtypes[X_val.dtypes == object].index
)

# Convert bools to int
df_valid_encoded = df_valid_encoded.astype({
    col: int for col in df_valid_encoded.select_dtypes('bool').columns
})

# Drop id

# Reindex to match train columns (important!)
df_valid_encoded = df_valid_encoded.reindex(columns=train_columns, fill_value=0)

# Scale using the same scaler fitted on train
df_valid_encoded_scaled = pd.DataFrame(
    scaler.transform(df_valid_encoded),
    columns=train_columns
)

# Optionally convert to NumPy array
X_valid_final = df_valid_encoded_scaled.to_numpy()



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


X_valid_selected = selector.transform(X_valid_final)
X_test_selected  = selector.transform(X_test_final)












X_train_tensor = torch.tensor(df_train_final, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_labels, dtype=torch.long)

X_valid_tensor = torch.tensor(X_valid_final, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid_df.to_numpy(), dtype=torch.long)

X_test_tensor = torch.tensor(X_test_final, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_final, dtype=torch.long)





dropout_rate=0.5
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)       # BatchNorm after first layer
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)       # BatchNorm after second layer
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, 8)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dropout1(x)

        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.dropout1(x)

        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.dropout1(x)

        x = F.relu(x)
        x = self.fc5(x)
        return x
    

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)    
val_dataset = TensorDataset(X_valid_tensor, y_valid_tensor) 
test_dataset=  TensorDataset(X_test_tensor, y_test_tensor) 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FeedForwardNN(input_dim=X_train_tensor.shape[1]).to(device)  
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 20



best_val_acc = 0.0  # Initialize best validation accuracy

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y_batch.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            outputs = model(X_val)
            loss = criterion(outputs, y_val)
            val_loss += loss.item() * y_val.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_val).sum().item()
            total += y_val.size(0)

    val_loss_epoch = val_loss / total
    val_acc_epoch = correct / total

    print(f"Epoch {epoch+1}, Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}")

    # Save model if validation accuracy improves
    if val_acc_epoch > best_val_acc:
        best_val_acc = val_acc_epoch
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Best model saved at epoch {epoch+1} with Val Acc: {best_val_acc:.4f}")



model.load_state_dict(torch.load("best_model.pth"))
model.to(device)
model.eval()



test_dataset = TensorDataset(X_test_tensor, y_test_tensor)    
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        outputs = model(X_test)
        loss = criterion(outputs, y_test)

        test_loss += loss.item() * y_test.size(0)  # sum over batch

        preds_FNN = outputs.argmax(dim=1)
        correct += (preds_FNN == y_test).sum().item()
        total += y_test.size(0)

avg_test_loss = test_loss / total
test_accuracy = correct / total


with open(os.path.join(folder_path,r'Results\FNN.json'), "w") as f:
    json.dump(test_accuracy , f, indent=4)




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








print("Test Accuracy:", accuracy_score(y_test_df, y_pred_ada))

with open(os.path.join(folder_path,r'Results\ada.txt'), "w") as f:
    f.write(json.dumps(accuracy_score(y_valid_df, y_pred_ada), indent=4))



import xgboost as xgb
'''

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=4
)

model.fit(
    X_train_selected,
    y_train_labels,
    eval_set=[(X_valid_selected, y_valid_df)],
    verbose=True
)


y_pred_xgb = model.predict(df_train_final)
valid_acc = accuracy_score(y_test_final, y_pred_xgb)

with open(os.path.join(folder_path,r'Results\xgb.json'), "w") as f:
    json.dump(valid_acc , f, indent=4)

'''
param_grid = {
    "max_depth": [ 5,6, 7,],
    "n_estimators": [50, 100, 200],
    "learning_rate": [ 0.1, 0.2,0.5],
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



test_acc = accuracy_score(y_test_final, y_pred_gridxgb)

with open(os.path.join(folder_path,r'Results\xgb_gs.json'), "w") as f:
    json.dump(test_acc , f, indent=4)

'''
from sklearn.svm import SVC



svm = SVC(kernel='rbf', C=1000.0, gamma='scale')  # You can try 'linear' or 'poly' as well
svm.fit(df_train_final, y_train_labels)
X_test,y_test=df_test_encoded_scaled,y_test


y_pred_svm = svm.predict(X_test_final)


svm_acc=accuracy_score(y_test, y_pred_svm)



with open(os.path.join(folder_path,r'Results\svm.json'), "w") as f:
    json.dump(svm_acc , f, indent=4)

'''

y_pred_ada
preds_FNN
y_pred_gridxgb



from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, svd_solver='full')  # keeps 95% variance


pca.fit(X_valid_final)  # or X_train_final if you use numpy array


pca_train=pca.transform(df_train_final)
pca_valid=pca.transform(X_valid_final)
pca_test=pca.transform(X_test_final)



grid_search.fit(pca_train, y_train_labels)

y_pred = grid_search.best_estimator_.predict(pca_test)
test_acc = accuracy_score(y_test_final, y_pred)

param_grid_ada = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1],
    'estimator__max_depth': [1, 2, 3],           # changed here
    'estimator__min_samples_split': [2, 4, 6]    # and here
}
adaboost = AdaBoostClassifier(estimator=DecisionTreeClassifier())

grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid_ada, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train_selected, y_train)



y_pred = ada.predict(pca_test)

test_acc = accuracy_score(y_test_final, y_pred)
