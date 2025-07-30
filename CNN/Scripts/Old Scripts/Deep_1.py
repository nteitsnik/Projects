import kagglehub

path = kagglehub.dataset_download("fantacher/neu-metal-surface-defects-data")

print("Path to dataset files:", path)

import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import numpy as np

##Get All the image file paths under the directory
image_dir = path
def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                image_files.append(os.path.join(root, file))
    return image_files


image_files = get_image_files(image_dir)




#Choose 5 random
sample_images = random.sample(image_files, 5)


#Plot them

fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for ax, image_path in zip(axes, sample_images):
    img = Image.open(image_path) # Convert to RGB
    
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(os.path.basename(image_path))
    
plt.show()




#Plot 1 in numeric form
img_array = np.array(Image.open(sample_images[0]) )

img_array.shape
img_array=img_array/255.0

'''
#Save Image to Check
np.set_printoptions(threshold=np.inf)

with open('image_raw_rgb.txt', 'w') as f:
    f.write(np.array2string(img_array, separator=', '))
'''
#Copy all the images to a train / Validation file 

folder_path = r'CNN\Data\Train' 
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder created at: {folder_path}")
else:
    print(f"Folder already exists at: {folder_path}")



source_file_train=path+'\\NEU Metal Surface Defects Data\\train'

train_image_files = get_image_files(source_file_train)




destination_folder = folder_path

for source_file in train_image_files:

    destination_file = os.path.join(destination_folder, os.path.basename(source_file))

    try:
        shutil.copy(source_file, destination_file)
        print(f"File copied to: {destination_file}")
    except Exception as e:
        print(f"Error: {e}")


folder_path = r'CNN\Data\Validation' 
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder created at: {folder_path}")
else:
    print(f"Folder already exists at: {folder_path}")



source_file_train=path+'\\NEU Metal Surface Defects Data\\Valid'

Valid_image_files = get_image_files(source_file_train)




destination_folder = folder_path

for source_file in Valid_image_files:

    destination_file = os.path.join(destination_folder, os.path.basename(source_file))

    try:
        shutil.copy(source_file, destination_file)
        print(f"File copied to: {destination_file}")
    except Exception as e:
        print(f"Error: {e}")


sizes=[]
img_mode=[]
for images in train_image_files :

    img = Image.open(image_path)

    sizes.append(img.size)
    img_mode.append(img.mode)

dimensions = np.array(sizes)
unique_dims, counts = np.unique(dimensions, axis=0, return_counts=True)
for dim, count in zip(unique_dims, counts):
    print(f"Size: {dim} -> Count: {count}")

img_mode_array=np.array(img_mode)
unique_lenghts, counts = np.unique(img_mode_array, axis=0, return_counts=True)
for dim, count in zip(unique_lenghts, counts):
    print(f"Type: {dim} -> Count: {count}")



#Fancy plot to be , ignore for now
'''
def plot_digit(digit, dem=200, font_size=12):
    max_ax = font_size * dem
    
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_xlim([0, max_ax])
    ax.set_ylim([0, max_ax])
    ax.axis('off')
    
    black = '#000000'
    
    # Create a blank image (white background)
    image = np.ones((dem, dem)) * 255
    
    # Fill the image with the pixel values
    for idx in range(dem):
        for jdx in range(dem):
            c = digit[jdx][idx] / 255.  # Convert pixel value to a scale between 0 and 1
            image[jdx][idx] = c * 255  # Apply it to the image
    
    # Use imshow for faster rendering
    ax.imshow(image, cmap='gray', extent=[0, max_ax, 0, max_ax])

    # Make the plot zoomable
    def on_zoom(event):
        # Get current axis limits
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        
        # Scroll up zooms in (event.button = 'up'), scroll down zooms out (event.button = 'down')
        if event.button == 'up':
            factor = 1.1  # Zoom in
        elif event.button == 'down':
            factor = 0.9  # Zoom out
        else:
            return
        
        # Update the axis limits based on the zoom factor
        ax.set_xlim([xlim[0] * factor, xlim[1] * factor])
        ax.set_ylim([ylim[0] * factor, ylim[1] * factor])
        
        # Redraw the plot with the new limits
        plt.draw()
    
    # Connect the zoom function to the mouse scroll
    fig.canvas.mpl_connect('scroll_event', on_zoom)
    
    # Show the plot
    plt.show()

'''
#Data PreProcessing
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    
    transforms.ToTensor(),  # Convert image to PyTorch tensor (scales values to [0, 1])
  
    
])



folder=r'CNN\Data\Train'



class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        """
        Args:
            image_folder (string): Path to the folder with images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []  # List to store the image file paths
        self.labels = []       # List to store the corresponding labels
        self.class_to_idx = {'Cr': 0, 'In': 1, 'Pa': 2,'PS':3,'RS':4,'Sc':5,}
        # Iterate through the folder and get image paths and labels
        for image_name in os.listdir(image_folder):
            if image_name.endswith(".bmp") :  # Assuming images are jpg/png
                image_path = os.path.join(image_folder, image_name)
                label = self.extract_label_from_filename(image_name)
                self.image_paths.append(image_path)
                self.labels.append(label)

    def extract_label_from_filename(self, filename):
        """
        Extract the label from the filename.
        Assumes that the label is in the filename (e.g., 'image_class1.jpg' -> class1).
        """
        # Example: 'image_class1.jpg' -> 'class1'
        label_str = filename.split('_')[0]  # Extract the label (split by underscore and take the last part)
        return label_str

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx])
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        # Get the label for the current image
        label_str = self.labels[idx]
        label = self.class_to_idx[label_str] 

        return image, torch.tensor(label, dtype=torch.long)  # Return image and label

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import logging

# Configure logging to only show info or higher level logs
logging.basicConfig(level=logging.INFO)

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
        
      


        # After 3 maxpool layers on 224x224 image, size becomes: 224 → 112 → 56 → 28
        self.fc1 = nn.Linear(16 * 25 * 25, 256)
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% rate
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # [B, 16, 112, 112]
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # [B, 32, 56, 56]
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # [B, 64, 28, 28]
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)               # Apply dropout
        x = self.fc2(x)                      # Fully connected layer
        return x



training_dataset=CustomImageDataset(image_folder=r'..\CNN\Data\Train', transform=transform)

training_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)


val_dataset = CustomImageDataset(image_folder=r'..\CNN\Data\Validation', transform=transform)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)



def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        init.xavier_uniform_(layer.weight)  # Xavier uniform for convolutional layers
        if layer.bias is not None:
            init.zeros_(layer.bias)  # Initialize bias to zero
    elif isinstance(layer, nn.Linear):
        init.xavier_uniform_(layer.weight)  # Xavier uniform for fully connected layers
        if layer.bias is not None:
            init.zeros_(layer.bias)









import torch.optim as optim    

model = SimpleCNN(num_classes=6).to(device)
model.apply(init_weights)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
num_epochs = 100
training_loss_log=[]
validation_loss_log=[]
acc_log=[]
best_acc=-1
best_model_path = "CNN_model.pth"


for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in training_loader:
        images = images.to(device)
        labels = labels.to(device)
       
        # Step 1: Zero the parameter gradients
        optimizer.zero_grad()

        # Step 2: Forward pass
        outputs = model(images)
        
        # Step 3: Compute loss
        loss = criterion(outputs, labels)
        
        # Step 4: Backward pass
        loss.backward()

        # Step 5: Update weights
        optimizer.step()

        # Logging
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    model.eval()  # switch to eval mode
    avg_train_loss = running_loss / len(training_loader)
    training_loss_log.append(avg_train_loss)

    val_loss = 0.0
    correct1 = 0
    total = 0
   

    with torch.no_grad():
        model.eval()   # disable gradient tracking for validation
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            
            
           
            total += labels.size(0)
            correct1 += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        accuracy = 100. * correct1 / len(val_loader.dataset)
        validation_loss_log.append(val_loss)
        acc_log.append(accuracy)
    if (accuracy > best_acc ) :
        best_acc = accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Saved new best model with val loss: {accuracy:.4f}")
    

plt.plot(training_loss_log, label='Train Loss')
plt.plot(validation_loss_log, label='Validation Loss')

plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Training Metrics Over Epochs')
plt.show()


plt.plot(acc_log, label='Accuracy')


plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.show()

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

from collections import Counter
import torch

label_counts = Counter()
for _, label in val_dataset:
    label_counts[label.item()] += 1

print(label_counts)