import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

batch_size = 16
num_images = 15
latent_dim = 100
device = torch.device("cpu")
img_shape = (3, 256, 256) 
model_path = "model.pth"

class Generator(nn.Module):
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
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


z = torch.randn(batch_size, latent_dim).to(device)

generator = Generator(latent_dim, img_shape)
generator.load_state_dict(torch.load(model_path))
generator.eval()

generated_images = generator(z)

os.makedirs("generated_images", exist_ok=True)
for i in range(num_images):
    save_image(generated_images[i, :, :, :], f"generated_images/generated_image_{i+1}.png", normalize=True)