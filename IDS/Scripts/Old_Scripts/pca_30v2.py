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
import os

folder_path = r"..\git_repos\DS_Test\IDS"
df_train = pd.read_csv( os.path.join(folder_path,r'Data\UNSW_NB15_training-set.csv\UNSW_NB15_training-set.csv'),index_col=False )



scaler = MinMaxScaler()

df_train.drop(['attack_cat'],axis=1,inplace=True)
df_encoded_train = pd.get_dummies(df_train, columns=df_train.dtypes[df_train.dtypes== object].index)
df_encoded_train = df_encoded_train.astype({col: int for col in df_encoded_train.select_dtypes('bool').columns})

ls_train=df_encoded_train[['label']]

df_encoded_train=df_encoded_train.drop(['id','label'],axis=1)

df_encoded_train_scaled=pd.DataFrame(scaler.fit_transform(df_encoded_train),columns=df_encoded_train.columns)
train_columns = df_encoded_train.columns.tolist()

df_encoded_train_scaled=df_encoded_train_scaled.to_numpy()
y_train_labels = ls_train.to_numpy()

pca = PCA(n_components=30)  # choose the number of components
X_pca_train = pca.fit_transform(df_encoded_train_scaled)

X_train, X_valid, y_train, y_valid = train_test_split(X_pca_train, y_train_labels,stratify=y_train_labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).squeeze()

X_val_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_val_tensor = torch.tensor(y_valid, dtype=torch.long).squeeze()


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
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)   
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FeedForwardNN(input_dim=X_train_tensor.shape[1]).to(device)  
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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


df_test = pd.read_csv(
    os.path.join(folder_path,r'Data\UNSW_NB15_testing-set.csv\UNSW_NB15_testing-set.csv'),
    index_col=False
)

df_test.drop(['attack_cat'], axis=1, inplace=True)

# Get dummies
df_test_encoded = pd.get_dummies(
    df_test,
    columns=df_test.dtypes[df_test.dtypes == object].index
)

# Convert bools to int
df_test_encoded = df_test_encoded.astype({
    col: int for col in df_test_encoded.select_dtypes('bool').columns
})

# Save labels
y_test = df_test_encoded[['label']].to_numpy()

# Drop id and label
df_test_encoded = df_test_encoded.drop(['id', 'label'], axis=1)

# 🟢 Reindex to match train columns
df_test_encoded = df_test_encoded.reindex(columns=train_columns, fill_value=0)

# 🟢 Now scale
df_test_encoded_scaled = pd.DataFrame(
    scaler.transform(df_test_encoded),
    columns=train_columns
)

X_test = df_test_encoded_scaled.to_numpy()
X_pca_test = pca.transform(X_test)

X_test_tensor = torch.tensor(X_pca_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).squeeze()


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

        preds = outputs.argmax(dim=1)
        correct += (preds == y_test).sum().item()
        total += y_test.size(0)

avg_test_loss = test_loss / total
test_accuracy = correct / total

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

with open(os.path.join(folder_path,r'Results\FNN_PCA.json'), "w") as f:
    json.dump(test_accuracy,f , indent=4)


mm=str(model)
with open(os.path.join(folder_path,r'Results\model_architecture_PCA.txt'), "w") as f:
    f.write(mm)   


from sklearn.svm import SVC



df_test = pd.read_csv(
    os.path.join(folder_path,r'Data\UNSW_NB15_testing-set.csv\UNSW_NB15_testing-set.csv'),
    index_col=False
)

df_test.drop(['attack_cat'], axis=1, inplace=True)

# Get dummies
df_test_encoded = pd.get_dummies(
    df_test,
    columns=df_test.dtypes[df_test.dtypes == object].index
)

# Convert bools to int
df_test_encoded = df_test_encoded.astype({
    col: int for col in df_test_encoded.select_dtypes('bool').columns
})

# Save labels
y_test = df_test_encoded[['label']].to_numpy()

# Drop id and label
df_test_encoded = df_test_encoded.drop(['id', 'label'], axis=1)

# 🟢 Reindex to match train columns
df_test_encoded = df_test_encoded.reindex(columns=train_columns, fill_value=0)

# 🟢 Now scale
df_test_encoded_scaled = pd.DataFrame(
    scaler.transform(df_test_encoded),
    columns=train_columns
)

X_test = df_test_encoded_scaled.to_numpy()
X_pca_test = pca.transform(X_test)

X_train,y_train=X_pca_train,y_train_labels



svm = SVC(kernel='rbf', C=1000.0, gamma='scale')  # You can try 'linear' or 'poly' as well
svm.fit(X_train, y_train)
X_test,y_test=df_test_encoded_scaled,y_test



y_pred = svm.predict(X_pca_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
svm_acc=accuracy_score(y_test, y_pred)
params = svm.get_params()


with open(os.path.join(folder_path,r'Results\svm_PCA.json'), "w") as f:
    json.dump(svm_acc , f, indent=4)

with open(os.path.join(folder_path,r'Results\svm_params_PCA.txt'), "w") as f:
     f.write(json.dumps(params , indent=4))    

import xgboost as xgb


X_train, X_valid, y_train, y_valid = train_test_split(X_pca_train, y_train_labels,stratify=y_train_labels, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=True
)

y_pred = model.predict(X_pca_test)

valid_acc = accuracy_score(y_test, y_pred)


X_train, y_train =df_encoded_train_scaled,y_train_labels



param_grid = {
    "max_depth": [3,4, 5,6,7,8],
    "n_estimators": [25,50, 100, 200],
    "learning_rate": [ 0.2,0.5,1],
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
    cv=4,
    verbose=2,
    n_jobs=6
)

grid_search.fit(X_train, y_train)


print("Best CV Accuracy:", grid_search.best_score_)

y_pred = grid_search.best_estimator_.predict(X_test)
print(grid_search.best_estimator_)
print(grid_search.best_params_)


test_acc = accuracy_score(y_test, y_pred)



with open(os.path.join(folder_path,r'Results\xgb_best_params_PCA.txt'), "w") as f:
    f.write(json.dumps(grid_search.best_params_, indent=4))



with open(os.path.join(folder_path,r'Results\xgb_PCA.json'), "w") as f:
    json.dump(test_acc, f, indent=4)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification



base_estimator = DecisionTreeClassifier(max_depth=2)

# 4. Create AdaBoost classifier
ada = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=100,        # number of weak learners
    learning_rate=0.001,
    random_state=4
)


ada.fit(X_pca_train, y_train)



df_test = pd.read_csv(
    os.path.join(folder_path,r'Data\UNSW_NB15_testing-set.csv\UNSW_NB15_testing-set.csv'),
    index_col=False
)

df_test.drop(['attack_cat'], axis=1, inplace=True)

# Get dummies
df_test_encoded = pd.get_dummies(
    df_test,
    columns=df_test.dtypes[df_test.dtypes == object].index
)

# Convert bools to int
df_test_encoded = df_test_encoded.astype({
    col: int for col in df_test_encoded.select_dtypes('bool').columns
})

# Save labels
y_test = df_test_encoded[['label']].to_numpy()

# Drop id and label
df_test_encoded = df_test_encoded.drop(['id', 'label'], axis=1)

# 🟢 Reindex to match train columns
df_test_encoded = df_test_encoded.reindex(columns=train_columns, fill_value=0)

# 🟢 Now scale
df_test_encoded_scaled = pd.DataFrame(
    scaler.transform(df_test_encoded),
    columns=train_columns
)



X_test,y_test=df_test_encoded_scaled,y_test
X_pca_test = pca.transform(X_test)

y_pred = ada.predict(X_pca_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))

with open(os.path.join(folder_path,r'Results\ada_PCA.json'), "w") as f:
    json.dump(accuracy_score(y_test, y_pred), f, indent=4)
