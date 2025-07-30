import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
df_train = pd.read_csv( r'..\IDS\Data\UNSW_NB15_training-set.csv\UNSW_NB15_training-set.csv',index_col=False )

df_train.dtypes[df_train.dtypes!= object] 
df_train_numeric=df_train[df_train.dtypes[df_train.dtypes!= object].index]

df_non_numeric = df_train.columns[df_train.dtypes == object].tolist()
df_non_numeric.append('id')
ls=df_train_numeric[['id','label']]
scaler = MinMaxScaler()
X = df_train_numeric.iloc[:, 1:-1]

# Scale
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=10)  # Change to desired number of components
principal_components = pca.fit_transform(X_scaled)



# Rebuild DataFrame with same (sliced) column names
df_scaled = pd.DataFrame(principal_components)

working_df = df_scaled.join(ls)

working_df=working_df.drop(['id'],axis=1)

X = working_df.drop('label', axis=1)
y = working_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.001,0.1, 1, 10,100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

svm = SVC(kernel='rbf', C=2.0, gamma='scale')  # You can try 'linear' or 'poly' as well
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)


# Step 5: Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.001,0.1, 1, 10,100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X, y)

print("Best Parameters:", grid.best_params_)

import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler



class TabularDataset(Dataset):
    def __init__(self, df, label_col='label', transform=None):
        self.labels = df[label_col].reset_index(drop=True)
        feature_df = df.drop(columns=[label_col]).reset_index(drop=True)
        self.feature_columns = feature_df.columns.tolist()
        self.features = feature_df.values.astype('float32')
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels.iloc[idx]

        if self.transform:
            x_df = pd.DataFrame([x], columns=self.feature_columns)
            x = self.transform.transform(x_df)[0].astype('float32')

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y).long()
        return x, y
'''
scaler = MinMaxScaler()
df = pd.read_csv(r'C:\Users\aiane\IDS\Data\UNSW_NB15_training-set.csv\UNSW_NB15_training-set.csv', drop_cols=df_non_numeric, label_col='label',transform=scaler)
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for X_batch, y_batch in loader:
    print(X_batch.shape, y_batch.shape)
    break

'''




df = pd.read_csv( r'C:\Users\aiane\IDS\Data\UNSW_NB15_training-set.csv\UNSW_NB15_training-set.csv',index_col=False )

df.dtypes[df_train.dtypes!= object] 
df_numeric=df[df.dtypes[df.dtypes!= object].index]
df_numeric=df_numeric.drop(columns=['id'])
train_df, val_df = train_test_split(df_numeric, test_size=0.2, stratify=df['label'], random_state=42)
'''
df_non_numeric = df.columns[df.dtypes == object].tolist()
df_non_numeric.append('id')
'''
scaler = MinMaxScaler()
scaler.fit(train_df.drop(columns=['label']))

train_dataset = TabularDataset(train_df, transform=scaler)
val_dataset = TabularDataset(val_df, transform=scaler)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)       # BatchNorm after first layer
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)       # BatchNorm after second layer
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, output_dim)

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
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = len(train_dataset.feature_columns)
model = FeedForwardNN(input_dim=input_dim, hidden_dim=64, output_dim=2).to(device)  # adjust output_dim if you have more classes

# Example training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)



from torch.utils.data import random_split, DataLoader

num_epochs=50
# Suppose dataset is your full dataset instance
dataset_size = len(train_dataset)
val_ratio = 0.2  # 20% for validation
val_size = int(val_ratio * dataset_size)
train_size = dataset_size - val_size

# Split dataset randomly


# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



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

        running_loss += loss.item() * y_batch.size(0)  # sum loss for the batch

        # Get predictions
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
            X_val = X_val.to(device)      # Move input features to GPU
            y_val = y_val.to(device) 
            outputs = model(X_val)
            loss = criterion(outputs, y_val)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y_val).sum().item()
            total += y_val.size(0)
    print(f"Epoch {epoch+1}, Val Loss: {val_loss/total:.4f}, Val Acc: {correct/total:.4f}")