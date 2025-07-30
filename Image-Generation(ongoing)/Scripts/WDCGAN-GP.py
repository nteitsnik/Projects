import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader, random_split
import os
from PIL import Image
import torchvision.utils as vutils
from pathlib import Path
save_dir = Path(r"C:\Users\aiane\git_repos\DS_Test\VAE\Results\WDCGAN_GP_Results")
save_dir.mkdir(parents=True, exist_ok=True)


transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),  # Ensure square crop
    transforms.ToTensor(),       # Convert to [0,1]
    transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1] for 3 channels
])

# === CelebA Dataset ===

class CelebAImagesWithAttributes(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Parse attribute file
       
        
        # Get intersection of images in folder and attribute data keys (filenames)
        self.img_names = sorted([
            f for f in os.listdir(img_dir) 
            if f.lower().endswith(('.jpg', '.png'))
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
   
        
        return image



dataset = CelebAImagesWithAttributes(
    img_dir=r'C:\Users\aiane\DL\Celebs\archive(1)\img_align_celeba\img_align_celeba',
    
    transform=transform
)


# === DataLoader ===
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# === DataLoaders ===


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)




class Generator128(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_map_size=64):
        super(Generator128, self).__init__()
        self.net = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),

            # (feature_map_size*16) x 4 x 4
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),

            # (feature_map_size*8) x 8 x 8
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),

            # (feature_map_size*4) x 16 x 16
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size , 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size ),
            nn.ReLU(True),


            # (feature_map_size) x 64 x 64
            nn.ConvTranspose2d(feature_map_size, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1]

            # Output size: img_channels x 128 x 128
        )

    def forward(self, x):
        return self.net(x)






class Critic128(nn.Module):
    def __init__(self, img_channels=3, feature_map_size=64):
        super(Critic128, self).__init__()

        # Typical DCGAN discriminator architecture:
        # Downsample 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        # Channels grow: img_channels -> feature_map_size -> feature_map_size*2 -> ... -> feature_map_size*8

        self.net = nn.Sequential(
            # Input: (batch, 3, 128, 128)
            nn.Conv2d(img_channels, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (batch, feature_map_size, 64, 64)
            nn.Conv2d(feature_map_size, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (batch, feature_map_size*2, 32, 32)
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (batch, feature_map_size*4, 16, 16)
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (batch, feature_map_size*8, 8, 8)
    

            # (batch, feature_map_size*16, 4, 4)
            nn.Conv2d(feature_map_size * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # Output shape: (batch, 1, 1, 1)
          
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1, 1).squeeze(1)  # flatten to (batch,)
        

class GAN(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_map_size=64, device="cpu"):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.device = device

        # Generator
        self.generator = Generator128(
            latent_dim=latent_dim,
            img_channels=img_channels,
            feature_map_size=feature_map_size
        ).to(device)

        # Discriminator
        self.discriminator = Critic128(
            img_channels=img_channels,
            feature_map_size=feature_map_size
        ).to(device)

    def sample_noise(self, batch_size):
        # Generate latent vectors (z)
        return torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)

    def generate_fake_images(self, batch_size):
        z = self.sample_noise(batch_size)
        return self.generator(z)

    def discriminate(self, images):
        return self.discriminator(images)        
    

    
          # Your generator class
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

latent_dim = 100
feature_map_size = 64
num_epochs = 20
lr_d = 1e-4
lr_g = 1e-5
beta1 = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"

# === GAN model ===
gan = GAN(latent_dim=latent_dim, feature_map_size=feature_map_size, device=device)
generator = gan.generator
discriminator = gan.discriminator
# Weights init
generator.apply(weights_init) 
discriminator.apply(weights_init) 

def generator_loss_wgan(fake_scores):
    return -fake_scores.mean()

def critic_loss_wgan(real_scores, fake_scores):
    return fake_scores.mean() - real_scores.mean()

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1, device=real_samples.device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

critic_iterations = 5
num_epochs = 50
lr_g = 1e-4
lr_d = 1e-4
beta1 = 0.0
beta2 = 0.9
lambda_gp = 10  # gradient penalty weight
d_losses = []
g_losses = []

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

for epoch in range(num_epochs):
    tqdm_loader = tqdm(dataloader)
    for i, real_images in enumerate(tqdm(dataloader)):
        
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        for _ in range(critic_iterations):
            noise = gan.sample_noise(batch_size)
            fake_images = generator(noise).detach()

            real_scores = discriminator(real_images)
            fake_scores = discriminator(fake_images)

            gp = compute_gradient_penalty(discriminator, real_images, fake_images)

            D_loss = critic_loss_wgan(real_scores, fake_scores) + lambda_gp * gp

            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()
            d_losses.append(D_loss.item())

    # Print losses
        noise = gan.sample_noise(batch_size)
        fake_images = generator(noise)
        fake_scores = discriminator(fake_images)

        G_loss = generator_loss_wgan(fake_scores)

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

    # Print losses
    print(f"[Epoch {epoch+1}/{num_epochs}] D_loss: {D_loss.item():.4f} | G_loss: {G_loss.item():.4f}")
    g_losses.append(G_loss.item())
    # Save generated samples
    with torch.no_grad():
        fixed_noise = gan.sample_noise(16)
        samples = generator(fixed_noise)
        samples = (samples + 1) / 2  # assuming outputs in [-1, 1]

        grid = vutils.make_grid(samples, nrow=4)
        ndarr = grid.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
        image = Image.fromarray(ndarr)
        image.save(rf"C:\Users\aiane\git_repos\DS_Test\VAE\Results\WGAN_GP_{epoch}.png")


def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, path="gan_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, path)            

def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, path="gan_checkpoint_1.pth", device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    epoch = checkpoint['epoch']
    print(f"✅ Loaded checkpoint from {path}, resuming from epoch {epoch}")
    return epoch    


save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch+1)