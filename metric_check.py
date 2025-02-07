# region SingleMetricCheck
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(original_image_path, inpainted_image_path):
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    inpainted_image = cv2.imread(inpainted_image_path, cv2.IMREAD_COLOR)
    
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    inpainted_gray = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2GRAY)
    
    ssim_score = ssim(original_gray, inpainted_gray)
    psnr_score = psnr(original_image, inpainted_image)
    
    return ssim_score, psnr_score

original_image_path = "Orginal_Images/02.png"
inpainted_image_path = "output/v3completed_image1.png"
ssim_score, psnr_score = calculate_metrics(original_image_path, inpainted_image_path)
print("SSIM:", ssim_score)
print("PSNR:", psnr_score)
# endregion
# -------------------------------------------------------------------------------------
# region DatasetMetricCheck
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(original_image_path, inpainted_image_path):
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    inpainted_image = cv2.imread(inpainted_image_path, cv2.IMREAD_COLOR)
    
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    inpainted_gray = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2GRAY)
    
    ssim_score = ssim(original_gray, inpainted_gray)
    psnr_score = psnr(original_image, inpainted_image)
    
    return ssim_score, psnr_score

original_images_dir = 'Labrator/1-Original_Images'
inpainted_images_dir = 'Labrator/4-Hat/2-Inpainted_Images'

total_ssim = 0.0
total_psnr = 0.0
num_images = 0

for i in range(101):
    filename = f"{i:05d}.jpg"
    original_image_path = os.path.join(original_images_dir, filename)
    inpainted_image_path = os.path.join(inpainted_images_dir, filename)
    
    if os.path.exists(original_image_path) and os.path.exists(inpainted_image_path):
        ssim_score, psnr_score = calculate_metrics(original_image_path, inpainted_image_path)
        total_ssim += ssim_score
        total_psnr += psnr_score
        num_images += 1
    else:
        print(f"Skipping {filename} as it does not exist in both directories.")

if num_images > 0:
    average_ssim = total_ssim / num_images
    average_psnr = total_psnr / num_images
    print("Average SSIM:", average_ssim)
    print("Average PSNR:", average_psnr)
else:
    print("No images found for comparison.")
# endregion
# -------------------------------------------------------------------------------------
# region RenameImages
import os

directory = 'Labrator/4-Hat/2-Inpainted_Images'

files = os.listdir(directory)

for filename in files:
    if filename.startswith("completed_") and filename.endswith(".jpg"):
        new_filename = filename.replace("completed_", "")
        
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        
        os.rename(old_filepath, new_filepath)

print("Renaming completed.")
# endregion
# -------------------------------------------------------------------------------------
# Analysis Result
# ==========================================
# NOSE & MOUTH 
    # Average SSIM: 0.9051388337425199
    # Average PSNR: 23.435345248042935
# Glasses
    # Average SSIM: 0.9110651638794877
    # Average PSNR: 23.31528043508007
#  Hat
    # Average SSIM: 0.8724905113847955
    # Average PSNR: 20.819711019502144
# ==========================================