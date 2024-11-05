# !pip install pywavelets scikit-image numpy

import numpy as np
import pywt
from skimage import io, color
from skimage.transform import resize
import os

def preprocess_image(image_path, size=(128, 128)):
    image = io.imread(image_path)
    if image.ndim == 3:
        image = color.rgb2gray(image)
    image_resized = resize(image, size, anti_aliasing=True)
    return image_resized

def compute_wavelet_coeffs(image, wavelet='db1', level=2):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs

def compare_coeffs(coeffs1, coeffs2):
    distance = 0
    for c1, c2 in zip(coeffs1, coeffs2):
        for subband1, subband2 in zip(c1, c2):
            distance += np.sum((subband1 - subband2) ** 2)
    return distance

def find_closest_image(target_image_path, dataset_folder, wavelet='db1', level=2):
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

target_image_path = 'output/00000.png'
dataset_folder = 'Labrator/0-RetrievalTester'

closest_image = find_closest_image(target_image_path, dataset_folder)
print(f'The closest image found is: {closest_image}')
