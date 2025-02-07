import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset

# Custom dataset class for loading images from a directory
class CustomDataset(Dataset):
    """
    A custom dataset to load images from a specified directory.

    Parameters:
        data_dir (str): Path to the dataset folder.
        transform (callable, optional): Transformation to apply to the images.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.img_list[idx])
        image = Image.open(img_name).convert("RGB")  # Ensure images are RGB format

        if self.transform:
            image = self.transform(image)

        return image

# Generator model definition
class Generator(nn.Module):
    """
    A fully connected Generator network for GAN.

    Architecture:
    - Input: Latent noise vector (latent_dim)
    - Several fully connected layers with LeakyReLU activations and BatchNorm
    - Output: Flattened image tensor reshaped into (channels, height, width)
    """
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),  # Output layer
            nn.Tanh()  # Normalize pixel values to [-1, 1]
        )
        self.img_shape = img_shape

    def forward(self, z):
        """
        Forward pass of the generator.

        Parameters:
            z (torch.Tensor): Input latent vector of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Generated image tensor reshaped to (batch_size, *img_shape).
        """
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)  # Reshape flattened output to image dimensions
        return img

# Discriminator model definition
class Discriminator(nn.Module):
    """
    A fully connected Discriminator network for GAN.

    Architecture:
    - Input: Flattened image tensor
    - Several fully connected layers with LeakyReLU activations
    - Output: Single value representing the probability of the image being real
    """
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability of image being real
        )

    def forward(self, img):
        """
        Forward pass of the discriminator.

        Parameters:
            img (torch.Tensor): Input image tensor of shape (batch_size, *img_shape).

        Returns:
            torch.Tensor: Probability of the image being real.
        """
        img_flat = img.view(img.size(0), -1)  # Flatten image into 1D vector
        validity = self.model(img_flat)
        return validity

# Function to train the GAN
def train_gan(device, generator, discriminator, dataloader, latent_dim, epochs, lr, b1, b2):
    """
    Trains the GAN using the provided generator and discriminator.

    Parameters:
        device (torch.device): The device to run the model on (CPU or GPU).
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        dataloader (DataLoader): DataLoader for real images.
        latent_dim (int): Dimension of the input noise vector.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        b1 (float): Adam optimizer beta1 parameter.
        b2 (float): Adam optimizer beta2 parameter.
    """
    adversarial_loss = nn.BCELoss()  # Binary cross-entropy loss
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(epochs):
        for imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            
            # Real and fake labels
            real_labels = torch.ones(imgs.size(0), 1).to(device)
            fake_labels = torch.zeros(imgs.size(0), 1).to(device)

            # Move models to the correct device
            generator = generator.to(device)
            discriminator = discriminator.to(device)

            # Train the Discriminator
            discriminator_optimizer.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs.view(imgs.size(0), -1)), real_labels)
            real_loss.backward()

            # Generate fake images
            z = torch.randn(imgs.size(0), latent_dim).to(device)
            fake_imgs = generator(z)
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake_labels)
            fake_loss.backward()

            discriminator_optimizer.step()  # Update discriminator weights

            # Train the Generator
            generator_optimizer.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim).to(device)
            gen_imgs = generator(z)
            gen_loss = adversarial_loss(discriminator(gen_imgs.view(gen_imgs.size(0), -1)), real_labels)
            gen_loss.backward()
            generator_optimizer.step()  # Update generator weights

# Function to prepare data loader
def prepare_data(data_path, batch_size):
    """
    Prepares the data loader for training.

    Parameters:
        data_path (str): Path to the dataset folder.
        batch_size (int): Number of images per batch.

    Returns:
        DataLoader: Dataloader for the dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Hyperparameters
latent_dim = 100  # Dimension of the input noise vector
img_shape = (3, 256, 256)  # Image dimensions (channels, height, width)
lr = 0.0002  # Learning rate for Adam optimizer
b1 = 0.5  # Beta1 hyperparameter for Adam optimizer
b2 = 0.999  # Beta2 hyperparameter for Adam optimizer
batch_size = 64  # Number of images per batch
epochs = 100  # Number of training epochs

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Generator and Discriminator models
generator = Generator(latent_dim, img_shape).to(device)
discriminator = Discriminator(img_shape).to(device)

# Load dataset
faces_dataset_path = os.getcwd() + '/celeba_hq_256/'  # Path to dataset
dataloader = prepare_data(faces_dataset_path, batch_size)

# Train the GAN
train_gan(device, generator, discriminator, dataloader, latent_dim, epochs, lr, b1, b2)

# Save the trained generator model
torch.save(generator.state_dict(), "generator.pth")