import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import torch.nn as nn
import pywt
from skimage import color, transform
import os

# ------------------------------
# Generator Model Definition
# ------------------------------
class Generator(nn.Module):
    """
    Generator model using fully connected layers.

    Parameters:
        latent_dim (int): Dimension of the latent space.
        img_shape (tuple): Shape of the generated image (channels, height, width).
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
        Forward pass to generate an image from a random noise vector.
        """
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# ------------------------------
# Image Processing Functions
# ------------------------------
def find_white_mask(image_path, lower_white=np.array([240, 240, 240]), upper_white=np.array([255, 255, 255])):
    """
    Detects white regions in an image and returns a binary mask.

    Parameters:
        image_path (str): Path to the input image.
        lower_white (numpy.array): Lower bound for white color in BGR.
        upper_white (numpy.array): Upper bound for white color in BGR.

    Returns:
        numpy.array: Boolean mask where white regions are True.
    """
    input_image = cv2.imread(image_path)
    mask = cv2.inRange(input_image, lower_white, upper_white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return mask.astype(bool)

def replace_white_regions(input_image_path, generator, latent_dim):
    """
    Replaces white regions in an image using a trained GAN-based generator.

    Parameters:
        input_image_path (str): Path to the input image.
        generator (Generator): Pre-trained Generator model.
        latent_dim (int): Dimension of the latent vector.

    Returns:
        numpy.array: Image with white regions replaced by generated content.
    """
    device = torch.device("cpu")
    input_image = cv2.imread(input_image_path)
    white_mask = find_white_mask(input_image_path)

    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        generated_image = generator(z)

    # Convert generated image from PyTorch tensor to NumPy array
    generated_np = (generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2.0 * 255.0
    generated_np = cv2.cvtColor(generated_np.astype(np.uint8), cv2.COLOR_BGR2RGB)

    output_image = np.copy(input_image)
    shift_amount = 10  # Shift pixels for blending effect
    height, width, _ = input_image.shape

    # Replace white regions with generated texture
    for y in range(height - shift_amount):
        output_image[y, white_mask[y]] = generated_np[y + shift_amount, white_mask[y]]

    return output_image

# ------------------------------
# Image Retrieval Functions
# ------------------------------
def preprocess_image(image_path, size=(128, 128)):
    """
    Preprocesses an image by converting it to grayscale and resizing.

    Parameters:
        image_path (str): Path to the image.
        size (tuple): Target size for resizing.

    Returns:
        numpy.array: Preprocessed grayscale image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image_resized

def compute_wavelet_coeffs(image, wavelet='db1', level=2):
    """
    Computes the wavelet transform coefficients of an image.

    Parameters:
        image (numpy.array): Grayscale image.
        wavelet (str): Type of wavelet used.
        level (int): Number of decomposition levels.

    Returns:
        list: Wavelet coefficients.
    """
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs

def compare_coeffs(coeffs1, coeffs2):
    """
    Computes the Euclidean distance between two sets of wavelet coefficients.

    Returns:
        float: Sum of squared differences.
    """
    distance = 0
    for c1, c2 in zip(coeffs1, coeffs2):
        for subband1, subband2 in zip(c1, c2):
            distance += np.sum((subband1 - subband2) ** 2)
    return distance

def find_closest_image(target_image_path, dataset_folder, wavelet='db1', level=2):
    """
    Finds the closest matching image in a dataset using wavelet feature comparison.

    Parameters:
        target_image_path (str): Path to the inpainted image.
        dataset_folder (str): Path to the dataset directory.

    Returns:
        str: Filename of the closest image.
    """
    target_image = preprocess_image(target_image_path)
    target_coeffs = compute_wavelet_coeffs(target_image, wavelet, level)

    closest_image = None
    min_distance = float('inf')

    for image_name in os.listdir(dataset_folder):
        image_path = os.path.join(dataset_folder, image_name)
        image = preprocess_image(image_path)
        image_coeffs = compute_wavelet_coeffs(image, wavelet, level)

        distance = compare_coeffs(target_coeffs, image_coeffs)
        if distance < min_distance:
            min_distance = distance
            closest_image = image_name

    return closest_image

# ------------------------------
# GUI Application
# ------------------------------
class InpaintSearchApp:
    """
    Tkinter GUI for image inpainting and retrieval.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Retrieval Inpainted Occluded Image")
        self.root.configure(bg='#2e2e2e')

        # Frame for buttons
        button_frame = tk.Frame(root, bg='#2e2e2e')
        button_frame.pack(pady=10)

        # Buttons
        self.upload_button = tk.Button(button_frame, text="Upload", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, padx=10)

        self.inpaint_button = tk.Button(button_frame, text="Inpaint", command=self.inpaint_image)
        self.inpaint_button.pack(side=tk.LEFT, padx=10)

        self.search_button = tk.Button(button_frame, text="Search", command=self.search_image)
        self.search_button.pack(side=tk.LEFT, padx=10)

        # Load pre-trained generator
        self.latent_dim = 100
        self.img_shape = (3, 256, 256)
        self.generator = Generator(self.latent_dim, self.img_shape)
        self.generator.load_state_dict(torch.load('model.pth'))
        self.generator.eval()

    def upload_image(self):
        self.uploaded_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    
    def inpaint_image(self):
        self.inpainted_image = replace_white_regions(self.uploaded_image_path, self.generator, self.latent_dim)

    def search_image(self):
        closest_image_name = find_closest_image(self.uploaded_image_path, 'Labrator/0-RetrievalTester')
        print(f"Closest image found: {closest_image_name}")

if __name__ == "__main__":
    root = tk.Tk()
    app = InpaintSearchApp(root)
    root.mainloop()