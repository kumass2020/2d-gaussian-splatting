import torch
from diffusers import DDPMPipeline
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from diffusers import DDPMScheduler, UNet2DModel
import os
from tqdm import tqdm
from torchvision import transforms

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
def generate_and_save_images(model, scheduler, noise=None, noise_dir=None, num_images=100, sample_size=256,
                             output_dir='output_dm'):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    use_predefined_noise = noise is not None


    for i in tqdm(range(num_images), desc="Generating images"):
        if use_predefined_noise:
            input = noise.to(device)
        else:
            noise_path = os.path.join(noise_dir, f'noise_{i + 1:05d}.png')
            noise_image = Image.open(noise_path)
            if noise_image.mode != 'RGB':
                noise_image = noise_image.convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((sample_size, sample_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            input = transform(noise_image).unsqueeze(0).to(device)

            if use_normalization:
                # Calculate mean and std of the input tensor
                mean = input.mean([0, 2, 3])
                std = input.std([0, 2, 3])
                normalize = transforms.Normalize(mean, std)
                input = normalize(input[0]).unsqueeze(0).to(device)

        # Verifying the normalization
        input_numpy = input.squeeze(0).permute(1, 2, 0).cpu().numpy()
        print(
            f"\nImage {i}: Mean after normalization: {input_numpy.mean()}, Std after normalization: {input_numpy.std()}")

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
def plot_images_grid(output_dir='output_dm', num_images=25):
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(num_images):
        img = mpimg.imread(f'{output_dir}/image_{i}.png')
        ax = axs[i // 5, i % 5]
        ax.imshow(img)
        ax.axis('off')

    # Adjust layout to remove any excess margins
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    plt.show()


# Assuming `model`, `scheduler`, and `noise` (if used) are predefined
# Option to use predefined noise or read noise maps from a directory
use_predefined_noise = False  # Change to False to read noise maps from directory
use_normalization = True
noise = torch.randn(1, 3, 256, 256) if use_predefined_noise else None  # Example noise, adjust as needed
noise_dir = 'output/240718-2337/train/ours_30000/vis'  # Set this to the directory containing noise maps

# Generate and save 100 images with progress bar
generate_and_save_images(model, scheduler, noise=noise, noise_dir=noise_dir, num_images=100, sample_size=256)

# Plot 25 images in a 5x5 grid
plot_images_grid(num_images=25)