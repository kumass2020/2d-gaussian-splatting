import os
import cv2
import numpy as np
from skimage.metrics import mean_squared_error as mse


def load_images(directory, prefix):
    """Load images from a directory with a specific prefix."""
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.startswith(prefix) and filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            images.append(image)
    return images


def apply_fft(image):
    """Apply FFT to the image and return the magnitude spectrum."""
    fft_image_shifted = np.fft.fftshift(np.fft.fft2(image))
    magnitude_spectrum = np.log1p(np.abs(fft_image_shifted))
    return magnitude_spectrum


def generate_random_noise_image(shape):
    """Generate a random noise image with N(0,1) distribution."""
    random_noise_image = np.random.normal(0, 1, shape)
    return random_noise_image


def calculate_average_similarity(gt_images, rendered_images, random_noise_images):
    """Calculate average MSE similarity between images in the frequency domain."""
    rendered_mse_total = 0
    random_mse_total = 0
    num_images = len(gt_images)

    for gt_image, rendered_image, random_noise_image in zip(gt_images, rendered_images, random_noise_images):
        gt_magnitude = apply_fft(gt_image)
        rendered_magnitude = apply_fft(rendered_image)
        random_magnitude = apply_fft(random_noise_image)

        rendered_mse = mse(gt_magnitude, rendered_magnitude)
        random_mse = mse(gt_magnitude, random_magnitude)

        rendered_mse_total += rendered_mse
        random_mse_total += random_mse

    avg_rendered_mse = rendered_mse_total / num_images
    avg_random_mse = random_mse_total / num_images

    return avg_rendered_mse, avg_random_mse


def process_datasets(dataset_paths):
    """Process all datasets and print average similarity scores."""
    for dataset in dataset_paths:
        gt_dir = os.path.join(dataset, 'gt')
        rendered_dir = os.path.join(dataset, 'vis')

        gt_images = load_images(gt_dir, "")
        rendered_images = load_images(rendered_dir, "noise_")
        random_noise_images = [generate_random_noise_image(img.shape) for img in rendered_images]

        avg_rendered_mse, avg_random_mse = calculate_average_similarity(gt_images, rendered_images, random_noise_images)

        print(f"Dataset: {dataset}")
        print(f"Average MSE (GT vs Rendered Noise): {avg_rendered_mse}")
        print(f"Average MSE (GT vs Random Noise): {avg_random_mse}\n")

prefix_date = "../output_render/240823-0157/"

# List of dataset directories
dataset_paths = [
    "bicycle",
    "bonsai",
    "counter",
    "flowers",
    "garden",
    "kitchen",
    "room",
    "stump",
    "treehill",
]

postfix = "/test/ours_20000"

dataset_paths = [prefix_date + dataset + postfix for dataset in dataset_paths]

# Process all datasets
process_datasets(dataset_paths)
