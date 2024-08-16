import os
import matplotlib.pyplot as plt
from PIL import Image
import random

# Function to get all images that start with 'iter40000' from a directory
def get_images(directory, prefix='iter40000'):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(prefix)]

# Function to plot images in a 4x4 grid with no margins and maintaining aspect ratio
def plot_images(image_paths, output_path='output_plot/outlier.png'):
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate the overall aspect ratio based on the first image
    first_img = Image.open(image_paths[0])
    img_aspect = first_img.width / first_img.height

    # Set the figure size based on the aspect ratio
    fig_width = 10  # You can adjust this value
    fig_height = fig_width / (img_aspect * 4 / 4)  # Adjust height according to aspect ratio
    fig, axes = plt.subplots(4, 4, figsize=(fig_width, fig_height))

    for ax, img_path in zip(axes.flatten(), image_paths):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')  # Hide axes

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(fig)

# Directory containing the images
directory = '/home1/hoho/sync/2d-gaussian-splatting/output_ig2g/garden_240809-051224/'
prefix = 'iter40000'

# Get all images starting with 'iter40000'
image_paths = get_images(directory, prefix)

# Sample 16 images randomly
sampled_image_paths = random.sample(image_paths, 16)

# Plot and save the images
plot_images(sampled_image_paths, output_path='../output_plot/random_noise.png')
