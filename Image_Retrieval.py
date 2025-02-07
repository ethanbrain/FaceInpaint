import numpy as np
import pywt
from skimage import io, color
from skimage.transform import resize
import os

def preprocess_image(image_path, size=(128, 128)):
    """
    Reads an image from the given path, converts it to grayscale if necessary, 
    resizes it to the specified dimensions, and applies anti-aliasing.

    Parameters:
        image_path (str): Path to the image file.
        size (tuple): Target size for resizing (width, height).

    Returns:
        np.ndarray: Preprocessed grayscale image.
    """
    image = io.imread(image_path)
    if image.ndim == 3:  # Convert RGB to grayscale if needed
        image = color.rgb2gray(image)
    image_resized = resize(image, size, anti_aliasing=True)
    return image_resized

def compute_wavelet_coeffs(image, wavelet='db1', level=2):
    """
    Computes the discrete wavelet transform (DWT) coefficients of an image.

    Parameters:
        image (np.ndarray): Grayscale image.
        wavelet (str): Type of wavelet to use (default is 'db1' - Daubechies wavelet).
        level (int): Decomposition level.

    Returns:
        list: Wavelet decomposition coefficients.
    """
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs

def compare_coeffs(coeffs1, coeffs2):
    """
    Computes the squared Euclidean distance between two sets of wavelet coefficients.

    Parameters:
        coeffs1 (list): Wavelet coefficients of the first image.
        coeffs2 (list): Wavelet coefficients of the second image.

    Returns:
        float: Sum of squared differences between coefficients.
    """
    distance = 0
    for c1, c2 in zip(coeffs1, coeffs2):  # Iterate through decomposition levels
        for subband1, subband2 in zip(c1, c2):  # Compare subbands
            distance += np.sum((subband1 - subband2) ** 2)
    return distance

def find_closest_image(target_image_path, dataset_folder, wavelet='db1', level=2):
    """
    Finds the most visually similar image to the target image from a dataset using 
    wavelet transform-based feature comparison.

    Parameters:
        target_image_path (str): Path to the target image.
        dataset_folder (str): Directory containing the dataset images.
        wavelet (str): Type of wavelet to use for feature extraction.
        level (int): Decomposition level.

    Returns:
        str: Filename of the closest matching image in the dataset.
    """
    target_image = preprocess_image(target_image_path)
    target_coeffs = compute_wavelet_coeffs(target_image, wavelet, level)

    closest_image = None
    min_distance = float('inf')  # Initialize with a very high value

    for image_name in os.listdir(dataset_folder):  # Iterate through dataset images
        image_path = os.path.join(dataset_folder, image_name)
        image = preprocess_image(image_path)
        image_coeffs = compute_wavelet_coeffs(image, wavelet, level)
        
        distance = compare_coeffs(target_coeffs, image_coeffs)
        if distance < min_distance:  # Update closest image if a smaller distance is found
            min_distance = distance
            closest_image = image_name

    return closest_image

# Define target image and dataset folder
target_image_path = 'output/00000.png'
dataset_folder = 'Labrator/0-RetrievalTester'

# Find the closest matching image in the dataset
closest_image = find_closest_image(target_image_path, dataset_folder)
print(f'The closest image found is: {closest_image}')
