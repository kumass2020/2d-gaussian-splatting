import os
import torchvision.transforms as transforms
from torchvision.datasets import LSUN
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# Define directories
real_images_dir = '../real_images'
os.makedirs(real_images_dir, exist_ok=True)

# Transform to resize images and convert them to tensors
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load the LSUN Church dataset
lsun_dataset = LSUN(root='data', classes=['church_outdoor_train'], transform=transform)
dataloader = DataLoader(lsun_dataset, batch_size=1, shuffle=True)

# Save a subset of real images
num_real_images = 1000
for i, (img, _) in enumerate(tqdm(dataloader, desc="Saving real images")):
    if i >= num_real_images:
        break
    img = transforms.ToPILImage()(img.squeeze(0))
    img.save(os.path.join(real_images_dir, f'image_{i}.png'))
