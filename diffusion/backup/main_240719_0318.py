import torch
from diffusers import DDPMPipeline
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from diffusers import DDPMScheduler, UNet2DModel
import os
from tqdm import tqdm

# model_id = "google/ddpm-cat-256"
# model_id = "google/ddpm-cifar10-32"
model_id = "google/ddpm-ema-church-256"

scheduler = DDPMScheduler.from_pretrained(model_id)
model = UNet2DModel.from_pretrained(model_id).to("cuda")

scheduler.set_timesteps(50)

print(scheduler.timesteps)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")

# Function to generate and save images
def generate_and_save_images(model, scheduler, noise, num_images=100, output_dir='output_dm'):
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(num_images), desc="Generating images"):
        input = noise

        for t in scheduler.timesteps:
            with torch.no_grad():
                noisy_residual = model(input, t).sample
            previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
            input = previous_noisy_sample

        image = (input / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype("uint8"))
        image.save(f"{output_dir}/image_{i}.png")

# Function to plot 25 images in a 5x5 grid
def plot_images_grid(prefix='output_dm/image_', num_images=25):
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(num_images):
        img = mpimg.imread(f'{prefix}{i}.png')
        ax = axs[i // 5, i % 5]
        ax.imshow(img)
        ax.axis('off')

    # Adjust layout to remove any excess margins
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    plt.show()

# Assuming `model`, `scheduler`, and `noise` are predefined
# Generate and save 100 images with progress bar
generate_and_save_images(model, scheduler, noise, num_images=100)

# Plot 25 images in a 5x5 grid
# plot_images_grid(num_images=25)
plot_images_grid(num_images=25)