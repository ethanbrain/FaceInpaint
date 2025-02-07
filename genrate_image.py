import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

# Configuration parameters
batch_size = 16         # Number of images to generate in a batch
num_images = 15         # Number of images to save
latent_dim = 100        # Dimensionality of the latent space (input noise vector)
device = torch.device("cpu")  # Device for computation (can be changed to 'cuda' if GPU is available)
img_shape = (3, 256, 256)  # Shape of generated images (channels, height, width)
model_path = "model.pth"  # Path to the pre-trained generator model

class Generator(nn.Module):
    """
    A simple fully connected Generator network for image synthesis.
    
    Architecture:
    - Takes a latent vector as input.
    - Passes through several fully connected layers with LeakyReLU activations.
    - Applies Batch Normalization to stabilize training.
    - Outputs an image tensor reshaped according to img_shape using a Tanh activation.
    """

    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),  # Output layer
            nn.Tanh()  # Normalize output to range [-1, 1]
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
        img = self.model(z)  # Pass through the network
        img = img.view(img.size(0), *self.img_shape)  # Reshape output to image dimensions
        return img

# Generate a batch of random latent vectors
z = torch.randn(batch_size, latent_dim).to(device)

# Load the pre-trained generator model
generator = Generator(latent_dim, img_shape)
generator.load_state_dict(torch.load(model_path))  # Load saved model weights
generator.eval()  # Set model to evaluation mode

# Generate images using the trained generator
generated_images = generator(z)

# Create output directory if it doesn't exist
os.makedirs("generated_images", exist_ok=True)

# Save the first 'num_images' generated images
for i in range(num_images):
    save_image(generated_images[i, :, :, :], 
               f"generated_images/generated_image_{i+1}.png", 
               normalize=True)  # Normalize images for better visualization