import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import torch.nn as nn
import pywt
from skimage import color, transform
import os

# Define the Generator class from your initial code
class Generator(nn.Module):
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
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# def find_white_mask(image_path, threshold=200):
#     input_image = cv2.imread(image_path)
#     input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
#     _, binary_mask = cv2.threshold(input_image_gray, threshold, 255, cv2.THRESH_BINARY)
#     binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
#     return binary_mask.astype(bool)
def find_white_mask(image_path, lower_white=np.array([240, 240, 240]), upper_white=np.array([255, 255, 255])):
    # Read the input image
    input_image = cv2.imread(image_path)
    
    # Convert the image to the HSV color space
    input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    
    # Create a mask for white color
    mask = cv2.inRange(input_image, lower_white, upper_white)
    
    # Perform a morphological close operation to close small holes in the binary mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    
    return mask.astype(bool)

def replace_white_regions(input_image_path, generator, latent_dim):
    device = torch.device("cpu")
    input_image = cv2.imread(input_image_path)
    skin_mask = find_white_mask(input_image_path)
    
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        generated_image = generator(z)
    
    generated_np = (generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2.0 * 255.0
    generated_np = cv2.cvtColor(generated_np.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    output_image = np.copy(input_image)
    shift_amount = 10
    height, width, _ = input_image.shape

    for y in range(height - shift_amount):
        output_image[y, skin_mask[y]] = generated_np[y + shift_amount, skin_mask[y]]

    return output_image

def preprocess_image(image_path, size=(128, 128)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
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

class InpaintSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Retrival Inpainted Occluded Image")
        self.root.configure(bg='#2e2e2e')
        
        # Adding frames for better layout
        button_frame = tk.Frame(root, bg='#2e2e2e')
        button_frame.grid(row=0, column=0, padx=10, pady=10, columnspan=2)
        
        image_frame = tk.Frame(root, bg='#2e2e2e')
        image_frame.grid(row=1, column=0, padx=10, pady=10, columnspan=2)
        
        self.upload_button = tk.Button(button_frame, text="Upload", command=self.upload_image, bg='#4CAF50', fg='white', font=('Helvetica', 12, 'bold'))
        self.upload_button.grid(row=0, column=0, padx=10, pady=10)

        self.inpaint_button = tk.Button(button_frame, text="Inpaint", command=self.inpaint_image, bg='#4CAF50', fg='white', font=('Helvetica', 12, 'bold'))
        self.inpaint_button.grid(row=0, column=1, padx=10, pady=10)

        self.search_button = tk.Button(button_frame, text="Search", command=self.search_image, bg='#4CAF50', fg='white', font=('Helvetica', 12, 'bold'))
        self.search_button.grid(row=0, column=2, padx=10, pady=10)

        self.image_box1 = tk.Label(image_frame, bg='white', width=256, height=256)
        self.image_box1.grid(row=1, column=0, padx=10, pady=10)

        self.image_box2 = tk.Label(image_frame, bg='white', width=256, height=256)
        self.image_box2.grid(row=1, column=1, padx=10, pady=10)

        self.uploaded_image_path = ""
        self.inpainted_image = None

        # Load the generator model
        self.latent_dim = 100
        self.img_shape = (3, 256, 256)
        self.generator = Generator(self.latent_dim, self.img_shape)
        self.generator.load_state_dict(torch.load('model.pth'))
        self.generator.eval()

        # Set default images
        default_image = Image.open("person.jpg").resize((256, 256), Image.Resampling.LANCZOS)
        self.default_img_tk = ImageTk.PhotoImage(default_image)
        self.image_box1.config(image=self.default_img_tk)
        self.image_box2.config(image=self.default_img_tk)

    def upload_image(self):
        self.uploaded_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.uploaded_image_path:
            image = Image.open(self.uploaded_image_path)
            image.thumbnail((256, 256))
            photo = ImageTk.PhotoImage(image)
            self.image_box1.config(image=photo)
            self.image_box1.image = photo

    def inpaint_image(self):
        if not self.uploaded_image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        
        self.inpainted_image = replace_white_regions(self.uploaded_image_path, self.generator, self.latent_dim)
        inpainted_image_rgb = cv2.cvtColor(self.inpainted_image, cv2.COLOR_BGR2RGB)
        inpainted_image_pil = Image.fromarray(inpainted_image_rgb)
        inpainted_image_pil.thumbnail((256, 256))
        photo = ImageTk.PhotoImage(inpainted_image_pil)
        self.image_box1.config(image=photo)
        self.image_box1.image = photo

    def search_image(self):
        if self.inpainted_image is None:
            messagebox.showwarning("Warning", "Please inpaint an image first.")
            return
        
        # Save the inpainted image to a temporary path
        inpainted_image_path = "temp_inpainted_image.png"
        cv2.imwrite(inpainted_image_path, self.inpainted_image)

        closest_image_name = find_closest_image(inpainted_image_path, 'Labrator/0-RetrievalTester')
        if closest_image_name:
            closest_image_path = os.path.join('Labrator/0-RetrievalTester', closest_image_name)
            closest_image = Image.open(closest_image_path)
            closest_image.thumbnail((256, 256))
            photo = ImageTk.PhotoImage(closest_image)
            self.image_box2.config(image=photo)
            self.image_box2.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = InpaintSearchApp(root)
    root.mainloop()

# 71-52