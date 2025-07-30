import kagglehub
import os
from collections import Counter
import torch.optim as optim    
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import logging
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download latest version
path = kagglehub.dataset_download("alex000kim/gc10det")

print("Path to dataset files:", path)

print(os.listdir(path))



from torch.utils.data import Dataset
from PIL import Image
import os


class GC10Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Collect all image paths and their labels
        for label_name in sorted(os.listdir(root_dir)):
            label_dir = os.path.join(root_dir, label_name)
            if os.path.isdir(label_dir):
                for fname in os.listdir(label_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.image_paths.append(os.path.join(label_dir, fname))
                        self.labels.append(int(label_name) - 1)  # label 1-10 → index 0-9

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


from torchvision import transforms
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

dataset = GC10Dataset(root_dir=str(path), transform=transform)
'''
##Random Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
'''


labels = [dataset[i][1] for i in range(len(dataset))]
labels = np.array(labels)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in splitter.split(np.zeros(len(labels)), labels):
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


import matplotlib.pyplot as plt
k=0
for i in range(500):
    image, label = train_dataset[i]  # your GC10Dataset instance
    if label == 9 :
        image = image.squeeze(0)   # Remove channel dimension: [1, H, W] → [H, W]

        plt.subplot(1, 6, k+1)
        k+=1
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
        if k==6 :
            break

plt.tight_layout()
plt.show()

all_labels = [label for _, label in dataset]

label_counts = Counter(all_labels)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) ##100x100

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2= nn.MaxPool2d(kernel_size=2, stride=2) ##50x50
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) ##25x25
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) ##12x12
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2) ##12x12
        
      


        # After 3 maxpool layers on 224x224 image, size becomes: 224 → 112 → 56 → 28
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% rate
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # [B, 16, 112, 112]
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # [B, 32, 56, 56]
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # [B, 64, 28, 28]
        x = self.pool4(F.relu(self.bn4(self.conv4(x)))) 
        x = self.pool5(F.relu(self.bn5(self.conv5(x)))) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)               # Apply dropout
        x = self.fc2(x)                      # Fully connected layer
        return x

 
model = SimpleCNN(num_classes=10)

model.eval()


print("fc1 weights:\n", model.fc1.weight.data)
print("\nfc2 weights:\n", model.fc2.weight.data)




criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-2
)




num_epochs = 50  # Adjust as needed
training_loss_log=[]
validation_loss_log=[]
acc_log=[]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
         
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_train_loss = running_loss / len(train_loader)
    training_loss_log.append(avg_train_loss)
    model.eval()
    val_loss = 0.0
    correct1 = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct1 += (predicted == labels).sum().item()

        accuracy = 100 * correct1 / total
        print(f"Validation Accuracy: {accuracy:.2f}%")    
        val_loss /= len(val_loader)
        accuracy = 100. * correct1 / len(val_loader.dataset)
        validation_loss_log.append(val_loss)
        acc_log.append(accuracy)
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

labels = np.array(all_labels)
preds = np.array(all_preds)

cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

from collections import defaultdict
correct_per_class = defaultdict(int)
total_per_class = defaultdict(int)
for true, pred in zip(labels, preds):
    total_per_class[true] += 1
    if true == pred:
        correct_per_class[true] += 1

classes = sorted(set(labels))
accuracy_per_class = {}
for cls in classes:
    acc = correct_per_class[cls] / total_per_class[cls]
    accuracy_per_class[cls] = acc
    print(f"Class {cls}: Accuracy = {acc:.2%}")        