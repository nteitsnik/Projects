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
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
import numpy as np
from collections import Counter
from torchvision import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = r'C:\Users\aiane\git_repo_metal\X-SDD-A-New-benchmark\X-SDD\datas'

print("Path to dataset files:", path)

print(os.listdir(path))


from torch.utils.data import Dataset
from PIL import Image
import os


class sddDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Get all label names sorted (e.g. ['cat', 'dog'])
        self.label_names = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )
        # Map label string to index: {'cat': 0, 'dog': 1, ...}
        self.label_to_idx = {label_name: i for i, label_name in enumerate(self.label_names)}

        # Collect all image paths and their labels
        for label_name in self.label_names:
            label_dir = os.path.join(root_dir, label_name)
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.image_paths.append(os.path.join(label_dir, fname))
                    self.labels.append(self.label_to_idx[label_name])  # integer label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        # Convert label to tensor (LongTensor for classification)
        label = torch.tensor(label, dtype=torch.long)

        return image, label
    

import torch
import torchvision.transforms as transforms

from PIL import Image

import torchvision.transforms.functional as TF



class PadTo200:
    def __call__(self, img_tensor):
        h, w = img_tensor.shape[-2], img_tensor.shape[-1]
        pad_h = 200 - h
        pad_w = 200 - w
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Image size is larger than 200x200, cannot pad")
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        return  F.pad(img_tensor, padding, value=0)

# Final transform
''''
# Final transform
transform = transforms.Compose([
    transforms.ToTensor(),  # PIL Image -> Tensor [C,H,W]
    PadTo200(),             # Pad to 200x200
])
'''
transform = transforms.Compose([
    transforms.Resize((200, 200)),  # Resize to 200x200
    transforms.ToTensor()
])


dataset = sddDataset(root_dir=str(path), transform=transform)

print(set(dataset.labels))  # List of class names

label_counts = Counter(dataset.labels)
for label_idx, count in label_counts.items():
    print(f"Label {label_idx} : {count} samples")

num_classes = len(label_counts)
total_samples = sum(label_counts.values())


class_weights = [ label_counts[i]/ total_samples for i in range(num_classes)]
class_weights = torch.tensor(class_weights, dtype=torch.float)
class_weights = class_weights / class_weights.sum()

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in splitter.split(np.zeros(len(dataset.labels)), dataset.labels):
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

img, label = train_dataset[0]  # Use your dataset
print(img.shape)



# If image is [1, H, W], squeeze it to [H, W]
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 3))
for i in range(6):
    img, label = train_dataset[i]
    if img.shape[0] == 1:
        img = img.squeeze(0)

    plt.subplot(1, 6, i + 1)
    plt.imshow(img.numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title(f"Label: {label}")
    plt.axis('off')

plt.tight_layout()
plt.show()




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

 


model = SimpleCNN(num_classes=6)
model.load_state_dict(torch.load("CNN_model.pth"))
model.eval()
new_num_classes=7

model.fc2 = nn.Linear(in_features=256, out_features=new_num_classes)
model.to(device)

'''print("fc1 weights:\n", model.fc1.weight.data)'''
print("\nfc2 weights:\n", model.fc2.weight.data)


for param in model.parameters():
    param.requires_grad = False

# Unfreeze the new last layer
for param in model.fc2.parameters():
    param.requires_grad = True



for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")    

'''list(model.fc1.parameters()) + list(model.fc2.parameters()),'''
criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(

    model.fc2.parameters(),
    lr=0.001,
    weight_decay=1e-2
)
scheduler = StepLR(optimizer, step_size=60, gamma=0.6)


num_epochs = 300  # Adjust as needed
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
    total1 = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total1 += labels.size(0)
            correct1 += (predicted == labels).sum().item()
        t_accuracy =100 * correct / total
        accuracy = 100 * correct1 / total1
        print(f"Training Accuracy: {t_accuracy:.2f}%")  
        print(f"Validation Accuracy: {accuracy:.2f}%")    
        val_loss /= len(val_loader)
        accuracy = 100. * correct1 / len(val_loader.dataset)
        validation_loss_log.append(val_loss)
        acc_log.append(accuracy)
    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
print(type(labels))


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