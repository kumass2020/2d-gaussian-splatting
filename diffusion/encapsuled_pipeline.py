import torch
from diffusers import DDPMPipeline
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256").to("cuda")
image = ddpm(num_inference_steps=25).images[0]
# image.show()
image.save("output_dm/image.png")

################# PIL
# # Open the image file
# img = Image.open("output_dm/image.png")
#
# # Display the image
# img.show()


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


################## opencv
# import cv2
#
# # Read the image
# img = cv2.imread('output_dm/image.png')
#
# # Display the image in a window
# cv2.imshow('Image', img)
#
# # Wait for a key press and close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()
