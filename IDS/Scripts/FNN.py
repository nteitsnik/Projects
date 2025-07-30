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


train_df, temp_df = train_test_split(
    df_all,
    test_size=0.2,
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








device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



X_train_tensor = torch.tensor(df_train_final, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_labels, dtype=torch.long)

X_valid_tensor = torch.tensor(X_valid_final, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid_df.to_numpy(), dtype=torch.long)

X_test_tensor = torch.tensor(X_test_final, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_final, dtype=torch.long)





dropout_rate=0.2
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=2):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)       # BatchNorm after first layer
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.bn2 = nn.BatchNorm1d(128)       # BatchNorm after second layer
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 8)
        self.bn5 = nn.BatchNorm1d(8)
        self.fc6 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        

        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
       

        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.dropout1(x)

        x = F.relu(x)
        x = self.fc5(x)
        x = self.dropout1(x)

        x = F.relu(x)
        x = self.fc6(x)
        return x
    

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)    
val_dataset = TensorDataset(X_valid_tensor, y_valid_tensor) 
test_dataset=  TensorDataset(X_test_tensor, y_test_tensor) 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(y_test_final), shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


test_loss = 0.0
correct = 0
total = 0

all_preds = []
all_targets = []

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

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_test.cpu().numpy())

# Compute metrics
accuracy = correct / total
precision = precision_score(all_targets, all_preds, average='macro')
recall = recall_score(all_targets, all_preds, average='macro')
f1 = f1_score(all_targets, all_preds, average='macro')
avg_test_loss = test_loss / total


metrics_dict = {
    
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

# Print metrics to console
for k, v in metrics_dict.items():
    print(f"{k}: {v:.4f}")

# Save to JSON file
with open(os.path.join(folder_path, r'Results\FNN.json'), "w") as f:
    json.dump(metrics_dict, f, indent=4)
