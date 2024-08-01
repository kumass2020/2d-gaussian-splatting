import torch
from diffusers import DDPMPipeline
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from diffusers import DDPMScheduler, UNet2DModel

model_id = "google/ddpm-cat-256"
# model_id = "google/ddpm-cifar10-32"

# scheduler = DDPMScheduler.from_pretrained(model_id)
scheduler = DDPMScheduler.from_config(model_id)
model = UNet2DModel.from_pretrained(model_id).to("cuda")

# scheduler.set_timesteps(50)

# print(scheduler.timesteps)

sample_size = model.config.sample_size
noisy_sample = torch.randn((1, 3, sample_size, sample_size), device="cuda")

with torch.no_grad():
    noisy_residual = model(sample=noisy_sample, timestep=2).sample

less_noisy_sample = scheduler.step(
    model_output=noisy_residual, timestep=2, sample=noisy_sample
).prev_sample
less_noisy_sample.shape

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = Image.fromarray(image_processed[0])
    print(f"\nImage at step {i}")
    image_pil.save(f"output_dm/image_{i}.png")
    img = mpimg.imread(f'output_dm/image_{i}.png')

    # Create a figure and axes with no padding
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(img)
    ax.axis('off')  # Turn off axis numbers and ticks

    # Remove the frame around the image
    plt.gca().set_frame_on(False)
    plt.show()

model.to("cuda")
noisy_sample = noisy_sample.to("cuda")

import tqdm

sample = noisy_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
  # 1. predict noise residual
  with torch.no_grad():
      residual = model(sample, t).sample

  # 2. compute less noisy image and set x_t -> x_t-1
  sample = scheduler.step(residual, t, sample).prev_sample

  # 3. optionally look at image
  if (i + 1) % 50 == 0:
      display_sample(sample, i + 1)
