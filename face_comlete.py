import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
import cv2

# ------------------------------
# Generator Model Definitions
# ------------------------------

class Generator(nn.Module):
    """
    Standard Generator model similar to AdvancedGenerator but with a smaller latent space mapping.
    """
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, z):
        """
        Forward pass for generating an image.
        
        Parameters:
            z (torch.Tensor): Latent space vector.
        
        Returns:
            torch.Tensor: Generated image.
        """
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# ------------------------------
# Image Processing Functions
# ------------------------------

def find_white_mask(image_path, lower_white=np.array([240, 240, 240]), upper_white=np.array([255, 255, 255])):
    """
    Detects white regions in the input image and generates a mask.

    Parameters:
        image_path (str): Path to the input image.
        lower_white (numpy.array): Lower threshold for white color in BGR format.
        upper_white (numpy.array): Upper threshold for white color in BGR format.

    Returns:
        numpy.array: Boolean mask where white regions are True.
    """
    input_image = cv2.imread(image_path)
    mask = cv2.inRange(input_image, lower_white, upper_white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return mask.astype(bool)

def find_skin_mask(image_path):
    """
    Detects skin regions in the image using color thresholding.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        numpy.array: Skin mask as a binary image.
    """
    input_image = cv2.imread(image_path)
    edges = cv2.Canny(input_image, 100, 200)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    skin_mask = cv2.inRange(input_image_hsv, lower_skin, upper_skin)
    final_skin_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=edges)
    return final_skin_mask

def correct_generated_color(input_image, generated_image, skin_mask):
    """
    Corrects the color of generated images by preserving the original skin tones.

    Parameters:
        input_image (numpy.array): Original image.
        generated_image (torch.Tensor): Generated image.
        skin_mask (numpy.array): Mask of detected skin regions.

    Returns:
        numpy.array: Color-corrected generated image.
    """
    generated_np = (generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2.0 * 255.0
    generated_np = generated_np[:input_image.shape[0], :input_image.shape[1], :]
    
    output_image = np.copy(generated_np)
    output_image[skin_mask] = input_image[skin_mask]

    return output_image

def replace_white_regions(input_image_path, output_image_path, generator, latent_dim):
    """
    Replaces white regions in an image with a generated texture using a trained generator model.

    Parameters:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the processed output image.
        generator (nn.Module): Trained generator model.
        latent_dim (int): Latent space dimension.
    """
    device = torch.device("cpu")
    input_image = cv2.imread(input_image_path)
    
    # Generate a mask for white regions
    skin_mask = find_white_mask(input_image_path)

    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        generated_image = generator(z)

    # Convert generated image to NumPy array
    generated_np = (generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2.0 * 255.0
    generated_np = cv2.cvtColor(generated_np, cv2.COLOR_BGR2RGB)

    output_image = np.copy(input_image)
    shift_amount = 10  # Amount of shifting to blend textures

    height, width, _ = input_image.shape

    # Replace white regions with generated texture
    for y in range(height - shift_amount):
        output_image[y, skin_mask[y]] = generated_np[y + shift_amount, skin_mask[y]]

    cv2.imwrite(output_image_path, output_image)

# ------------------------------
# Main Execution
# ------------------------------

# Set up paths and parameters
base_dir = os.getcwd()
input_image_path = base_dir + '/input/image.png'
output_image_path = base_dir + '/output/3_image.jpg'

latent_dim = 100
img_shape = (3, 256, 256)

# Load trained Generator
generator = Generator(latent_dim, img_shape)
generator.load_state_dict(torch.load('model.pth'))
generator.eval()

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Process all images in the input folder
for name in os.listdir(base_dir + '/input'):
    input_image_path = base_dir + f'/input/{name}'
    output_image_path = base_dir + f'/output/3_{name}'
    replace_white_regions(input_image_path, output_image_path, generator, latent_dim)
