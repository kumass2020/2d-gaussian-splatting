import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# `image` is an RGB PIL.Image
image = Image.open('../output/240718-2337/train/ours_30000/gt/00000.png')
images = pipe("make the truck made of glass", image=image).images

images[0].save("output_dm/image.png")

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