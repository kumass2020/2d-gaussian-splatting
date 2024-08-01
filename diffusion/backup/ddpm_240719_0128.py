import torch
from diffusers import DDPMPipeline
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from diffusers import DDPMScheduler, UNet2DModel

model_id = "google/ddpm-cat-256"
# model_id = "google/ddpm-cifar10-32"

scheduler = DDPMScheduler.from_pretrained(model_id)
model = UNet2DModel.from_pretrained(model_id).to("cuda")

scheduler.set_timesteps(50)

print(scheduler.timesteps)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")

input = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample

image = (input / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
image.save("output_dm/image.png")

################# matplot
# Read the image
img = mpimg.imread('output_dm/image.png')

# Create a figure and axes with no padding
fig, ax = plt.subplots()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.imshow(img)
ax.axis('off')  # Turn off axis numbers and ticks

# Remove the frame around the image
plt.gca().set_frame_on(False)
plt.show()