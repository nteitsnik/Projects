import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

def parse_celeba_attr_csv(attr_path):
    """
    Returns:
        attr_names: list of attribute names
        data: dict mapping filename -> list of attributes
    """
    data = {}
    with open(attr_path, "r") as f:
        lines = f.readlines()
    # First line is the header
    header = lines[0].strip().split(",")
    attr_names = header[1:]  # skip 'image_id'

    for line in lines[1:]:
        parts = line.strip().split(",")
        filename = parts[0]
        attr_values = [int(x) for x in parts[1:]]
        attr_values = [(v + 1) // 2 for v in attr_values]  # convert -1/1 to 0/1
        data[filename] = attr_values
    
    return attr_names, data

class CelebAImagesWithAttributes(Dataset):
    def __init__(self, img_dir, attr_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Parse attribute file
        self.attr_names, self.attr_data = parse_celeba_attr_csv(attr_path)
        
        # Get intersection of images in folder and attribute data keys (filenames)
        img_files = set(os.listdir(img_dir))
        self.img_names = sorted([f for f in img_files if f in self.attr_data])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        attrs = self.attr_data[img_name]
        attrs = torch.tensor(attrs, dtype=torch.float32) # convert to floats if needed
        
        return image, attrs

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
])



dataset = CelebAImagesWithAttributes(
    img_dir=r'..\DL\Celebs\archive(1)\img_align_celeba\img_align_celeba',
    attr_path=r'..\DL\Celebs\archive(1)\list_attr_celeba.csv',
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example iteration
for images, features in dataloader:
    print(images.shape)   # torch.Size([32, 3, 128, 128])
    print(len(features))  # 32, each element is list of attributes
    break
import matplotlib.pyplot as plt
import numpy as np

# Get a single batch
loader = DataLoader(dataset, batch_size=6, shuffle=True)

# Get one batch
images, attrs = next(iter(loader))

# Convert attrs to numpy (if it's a list)
attrs = np.array(attrs)

# Plot 6 images in a row
fig, axes = plt.subplots(1, 6, figsize=(18, 5))

for i, ax in enumerate(axes):
    # Permute image from [C,H,W] -> [H,W,C] for plotting
    img = images[i].permute(1, 2, 0).numpy()
    
    # If you have normalized images, unnormalize here
    # img = img * 0.5 + 0.5  # Uncomment if you used Normalize(mean=0.5, std=0.5)

    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Attrs: {attrs[i][:5]}")  # Show first 5 attributes

plt.tight_layout()
plt.show()




import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, attr_dim, latent_dim):
        super().__init__()
        # CNN for image
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3,padding= 1),
            nn.BatchNorm2d(16),             # 64x64 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32,kernel_size=3,padding= 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3,padding= 1),# 16x16 
            nn.BatchNorm2d(64),
          
            nn.MaxPool2d(2, 2),
          

            nn.Flatten()
        )
        # Calculate flattened CNN output size (depends on input size)
        dummy = torch.zeros(1, 3, 128, 128)
        cnn_out_dim = self.cnn(dummy).shape[1]
        

        # FC for attributes (shallow)
        self.fc_attr = nn.Sequential(
            nn.Linear(attr_dim, 16),
            nn.ReLU()
        )

        # Combine CNN and attr embeddings and produce mu and logvar
        self.fc_mu = nn.Linear(cnn_out_dim + 16, latent_dim)
        self.fc_logvar = nn.Linear(cnn_out_dim + 16, latent_dim)

    def forward(self, x, c):
        x_feat = self.cnn(x)      # [batch, cnn_out_dim]
        
        c_feat = self.fc_attr(c)  # [batch, 32]
        combined = torch.cat([x_feat, c_feat], dim=1)  # concat
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar
    

class Decoder(nn.Module):
    def __init__(self, attr_dim, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + attr_dim, 4*4*256),  # expand to feature map size
            nn.ReLU()
        )
        # Transpose conv layers to upsample back to 64x64
        self.deconv = nn.Sequential(
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4->8
    nn.ReLU(),
    nn.BatchNorm2d(128),

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 8->16
    nn.ReLU(),
    nn.BatchNorm2d(64),

    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 16->32
    nn.ReLU(),
    nn.BatchNorm2d(32),

    nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 32->64
    nn.ReLU(),
    nn.BatchNorm2d(16),

    nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),     # 64->128
    nn.Sigmoid()
    )
        
    def forward(self, z, c):
        # Concatenate latent and condition
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)  # reshape into feature map
        x = self.deconv(x)
        
        return torch.sigmoid(x)   
    

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (elementwise)
    loss = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return loss + 20*KLD, loss, KLD    



def train_cvae(
    encoder,
    decoder,
    dataloader,
    optimizer,
    device,
    num_epochs=10
):
    encoder.train()
    decoder.train()

    
    total_loss = 0.0
    total_bce = 0.0
    total_kld = 0.0

    for batch_idx, (images, attrs) in enumerate(dataloader):
        images = images.to(device)
        attrs = attrs.to(device)

        optimizer.zero_grad()

            # Encode
        mu, logvar = encoder(images, attrs)

            # Sample z
        z = reparameterize(mu, logvar)

            # Decode
        recon_images = decoder(z, attrs)

            # Compute loss
        loss, bce, kld = loss_function(recon_images, images, mu, logvar)

            # Backprop
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce += bce.item()
        total_kld += kld.item()

        if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.2f} "
                    f"BCE: {bce.item():.2f} "
                    f"KLD: {kld.item():.2f}"
                )

    avg_loss = total_loss / len(dataloader.dataset)
    avg_bce = total_bce / len(dataloader.dataset)
    avg_kld = total_kld / len(dataloader.dataset)

    print(
            f"===> Epoch [{epoch+1}] Average loss: {avg_loss:.4f} "
            f"BCE: {avg_bce:.4f} KLD: {avg_kld:.4f}"
        )
    return avg_loss, avg_bce, avg_kld

def validate_cvae(
    encoder,
    decoder,
    dataloader,
    device
):
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_kld = 0.0

    all_recons = []
    all_inputs = []

    with torch.no_grad():
        for images, attrs in dataloader:
            images = images.to(device)
            attrs = attrs.to(device)

            # Encode
            mu, logvar = encoder(images, attrs)

            # Reparameterize
            z = reparameterize(mu, logvar)

            # Decode
            recon_images = decoder(z, attrs)

            # Compute loss
            loss, bce, kld = loss_function(recon_images, images, mu, logvar)

            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

            # Optionally collect for visualization
            all_inputs.append(images.cpu())
            all_recons.append(recon_images.cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    avg_bce = total_bce / len(dataloader.dataset)
    avg_kld = total_kld / len(dataloader.dataset)

    print(
        f"===> Validation Average loss: {avg_loss:.4f} "
        f"BCE: {avg_bce:.4f} KLD: {avg_kld:.4f}"
    )

    # Concatenate all batches
    all_inputs = torch.cat(all_inputs)
    all_recons = torch.cat(all_recons)

    return avg_loss, avg_bce, avg_kld, all_inputs, all_recons        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


num_epochs1 = 5
learning_rate = 1e-4

encoder = Encoder(attr_dim=40, latent_dim=64).to(device)
decoder = Decoder(attr_dim=40, latent_dim=64).to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=learning_rate
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,  # set to 0 if you have issues on Windows
    pin_memory=True,
    
   
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)



for epoch in range(0, num_epochs1 + 1):
    train_loss, train_bce, train_kld = train_cvae(
        encoder, decoder, train_loader, optimizer, device
    )

    val_loss, val_bce, val_kld, val_inputs, val_recons = validate_cvae(
        encoder, decoder, val_loader, device
    )

    print(
        f"Epoch {epoch+1}: "
        f"Train Loss {train_loss:.4f} (BCE {train_bce:.4f}, KLD {train_kld:.4f}) | "
        f"Val Loss {val_loss:.4f} (BCE {val_bce:.4f}, KLD {val_kld:.4f})"
    )
















img = Image.open('path_to_image.jpg')
print(img.size)


latent_dim = 128
attr_dim = 40

# Create dummy inputs
dummy_z = torch.randn(1, latent_dim)
dummy_c = torch.randn(1, attr_dim)

# Forward pass
output = decoder(dummy_z, dummy_c)
print(dataset[0][0].shape)


def run_inference(encoder, decoder, image, attr, device):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)  # add batch dim if single image
        attr = attr.to(device).unsqueeze(0)    # add batch dim if single attr vector

        mu, logvar = encoder(image, attr)
        z = reparameterize(mu, logvar)
        recon_image = decoder(z, attr)
        
    return recon_image.squeeze(0) 

def show_image(tensor_img):
    # If image is [C, H, W], convert to [H, W, C]
    img = tensor_img.cpu().permute(1, 2, 0).numpy()
    # If image was normalized, you might need to unnormalize here
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def show_two_images(img1, img2, title1='Image 1', title2='Image 2'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Convert and show first image
    axes[0].imshow(img1.cpu().permute(1, 2, 0).numpy())
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    # Convert and show second image
    axes[1].imshow(img2.cpu().permute(1, 2, 0).numpy())
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    plt.show()

train_dataset[0][0]
reconstructed = run_inference(encoder, decoder, train_dataset[0][0], train_dataset[0][1], device)
show_image(train_dataset[0][0])
show_image(reconstructed)

show_two_images(train_dataset[0][0], reconstructed, 'Original', 'Reconstructed')