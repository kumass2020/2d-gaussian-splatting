import os
import cv2
import numpy as np
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim
from scipy.signal import correlate2d, correlate
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import nn

# Your InstructPix2Pix class should be imported here.
from diffusion.orig.ip2p import InstructPix2Pix

# Initialize the InstructPix2Pix model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pix2pix_model = InstructPix2Pix(device=device, num_train_timesteps=1000, ip2p_use_full_precision=False)

# Loading and processing functions
def load_images(directory, prefix):
    """Load images from a directory with a specific prefix and scale them to [0, 255]."""
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.startswith(prefix) and filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)  # Scale to [0, 255]
            images.append(image)
    return images

def apply_fft(image):
    """Apply FFT to the image and return the magnitude spectrum."""
    fft_image_shifted = np.fft.fftshift(np.fft.fft2(image))
    magnitude_spectrum = np.log1p(np.abs(fft_image_shifted))
    return magnitude_spectrum

def convert_images_to_latents(images):
    """Convert images to latents using InstructPix2Pix VAE."""
    images_tensor = torch.from_numpy(np.array(images)).float().to(device) / 255.0
    images_tensor = images_tensor.unsqueeze(1).repeat(1, 3, 1, 1)  # Repeat grayscale to 3 channels
    images_tensor = images_tensor.to(pix2pix_model.auto_encoder.dtype)
    latents = pix2pix_model.imgs_to_latent(images_tensor)
    latents = latents.detach().cpu().numpy()
    return latents

def low_pass_filter(image, cutoff_frequency):
    """Apply a low-pass filter to each channel to keep only low-frequency components."""
    # If the image has a channel dimension (e.g., (channels, height, width))
    if len(image.shape) == 3:
        channels, rows, cols = image.shape
        filtered_image = np.zeros_like(image)

        for c in range(channels):
            # Apply low-pass filter for each channel
            fft_image = np.fft.fftshift(np.fft.fft2(image[c]))
            mask = np.zeros((rows, cols), np.uint8)
            crow, ccol = rows // 2, cols // 2
            mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 1
            filtered_fft = fft_image * mask
            filtered_image[c] = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))

    # If the image is 2D (no channel dimension)
    else:
        rows, cols = image.shape
        fft_image = np.fft.fftshift(np.fft.fft2(image))
        mask = np.zeros((rows, cols), np.uint8)
        crow, ccol = rows // 2, cols // 2
        mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 1
        filtered_fft = fft_image * mask
        filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))

    return filtered_image


def high_pass_filter(image, cutoff_frequency):
    """Apply a high-pass filter to each channel to keep only high-frequency components."""
    # If the image has a channel dimension (e.g., (channels, height, width))
    if len(image.shape) == 3:
        channels, rows, cols = image.shape
        filtered_image = np.zeros_like(image)

        for c in range(channels):
            # Apply high-pass filter for each channel
            fft_image = np.fft.fftshift(np.fft.fft2(image[c]))
            mask = np.ones((rows, cols), np.uint8)
            crow, ccol = rows // 2, cols // 2
            mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0
            filtered_fft = fft_image * mask
            filtered_image[c] = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))

    # If the image is 2D (no channel dimension)
    else:
        rows, cols = image.shape
        fft_image = np.fft.fftshift(np.fft.fft2(image))
        mask = np.ones((rows, cols), np.uint8)
        crow, ccol = rows // 2, cols // 2
        mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0
        filtered_fft = fft_image * mask
        filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))

    return filtered_image


def generate_random_noise_image(shape):
    """Generate a random noise image with N(0,1) distribution and scale it to [0, 255]."""
    random_noise_image = np.random.normal(0, 1, shape)
    random_noise_image = cv2.normalize(random_noise_image, None, 0, 255, cv2.NORM_MINMAX)
    random_noise_image = random_noise_image.astype(np.uint8)
    return random_noise_image

def calculate_average_similarity(gt_images, rendered_images, random_noise_images, mode='all', cutoff_frequency=30):
    """Calculate average similarity metrics between images in the frequency domain."""
    rendered_mse_total = 0
    random_mse_total = 0
    rendered_ssim_total = 0
    random_ssim_total = 0
    rendered_cross_corr_total = 0
    random_cross_corr_total = 0
    rendered_cosine_sim_total = 0
    random_cosine_sim_total = 0

    num_images = len(gt_images)

    # Convert images and noises to latents using the VAE
    gt_latents = convert_images_to_latents(gt_images)
    rendered_latents = convert_images_to_latents(rendered_images)
    random_noise_latents = convert_images_to_latents(random_noise_images)

    for gt_latent, rendered_latent, random_noise_latent in zip(gt_latents, rendered_latents, random_noise_latents):
        if mode == 'low_frequency':
            gt_latent = low_pass_filter(gt_latent, cutoff_frequency)
            rendered_latent = low_pass_filter(rendered_latent, cutoff_frequency)
            random_noise_latent = low_pass_filter(random_noise_latent, cutoff_frequency)
        elif mode == 'high_frequency':
            gt_latent = high_pass_filter(gt_latent, cutoff_frequency)
            rendered_latent = high_pass_filter(rendered_latent, cutoff_frequency)
            random_noise_latent = high_pass_filter(random_noise_latent, cutoff_frequency)

        # Apply FFT and get magnitude spectrum
        gt_magnitude = apply_fft(gt_latent)
        rendered_magnitude = apply_fft(rendered_latent)
        random_magnitude = apply_fft(random_noise_latent)

        # MSE
        rendered_mse = mse(gt_magnitude, rendered_magnitude)
        random_mse = mse(gt_magnitude, random_magnitude)

        # SSIM - specify the data_range parameter
        data_range = np.max(gt_magnitude) - np.min(gt_magnitude)  # Adjust based on the image's range
        rendered_ssim = ssim(gt_magnitude, rendered_magnitude, data_range=data_range, channel_axis=0)
        random_ssim = ssim(gt_magnitude, random_magnitude, data_range=data_range, channel_axis=0)

        # Cross-Correlation
        rendered_cross_corr = np.max(correlate(gt_magnitude, rendered_magnitude, mode='valid'))
        random_cross_corr = np.max(correlate(gt_magnitude, random_magnitude, mode='valid'))

        # Cosine Similarity
        rendered_cosine_sim = cosine_similarity(gt_magnitude.flatten().reshape(1, -1),
                                                rendered_magnitude.flatten().reshape(1, -1))[0][0]
        random_cosine_sim = cosine_similarity(gt_magnitude.flatten().reshape(1, -1),
                                              random_magnitude.flatten().reshape(1, -1))[0][0]

        # Accumulate the metrics
        rendered_mse_total += rendered_mse
        random_mse_total += random_mse
        rendered_ssim_total += rendered_ssim
        random_ssim_total += random_ssim
        rendered_cross_corr_total += rendered_cross_corr
        random_cross_corr_total += random_cross_corr
        rendered_cosine_sim_total += rendered_cosine_sim
        random_cosine_sim_total += random_cosine_sim

    # Calculate averages
    avg_rendered_mse = rendered_mse_total / num_images
    avg_random_mse = random_mse_total / num_images
    avg_rendered_ssim = rendered_ssim_total / num_images
    avg_random_ssim = random_ssim_total / num_images
    avg_rendered_cross_corr = rendered_cross_corr_total / num_images
    avg_random_cross_corr = random_cross_corr_total / num_images
    avg_rendered_cosine_sim = rendered_cosine_sim_total / num_images
    avg_random_cosine_sim = random_cosine_sim_total / num_images

    return {
        "avg_rendered_mse": avg_rendered_mse,
        "avg_random_mse": avg_random_mse,
        "avg_rendered_ssim": avg_rendered_ssim,
        "avg_random_ssim": avg_random_ssim,
        "avg_rendered_cross_corr": avg_rendered_cross_corr,
        "avg_random_cross_corr": avg_random_cross_corr,
        "avg_rendered_cosine_sim": avg_rendered_cosine_sim,
        "avg_random_cosine_sim": avg_random_cosine_sim,
    }


def process_datasets(dataset_paths, mode='all', cutoff_frequency=30):
    """Process all datasets and print average similarity scores."""
    for dataset in dataset_paths:
        gt_dir = os.path.join(dataset, 'gt')
        rendered_dir = os.path.join(dataset, 'vis')

        gt_images = load_images(gt_dir, "")
        rendered_images = load_images(rendered_dir, "noise_")
        random_noise_images = [generate_random_noise_image(img.shape) for img in rendered_images]

        results = calculate_average_similarity(gt_images, rendered_images, random_noise_images, mode, cutoff_frequency)

        print(f"Dataset: {dataset} (Mode: {mode})")
        print(f"MSE (Rendered Noise): {results['avg_rendered_mse']}")
        print(f"MSE (Random Noise): {results['avg_random_mse']}")
        print(f"SSIM (Rendered Noise): {results['avg_rendered_ssim']}")
        print(f"SSIM (Random Noise): {results['avg_random_ssim']}")
        print(f"Cross-Correlation (Rendered Noise): {results['avg_rendered_cross_corr']}")
        print(f"Cross-Correlation (Random Noise): {results['avg_random_cross_corr']}")
        print(f"Cosine Similarity (Rendered Noise): {results['avg_rendered_cosine_sim']}")
        print(f"Cosine Similarity (Random Noise): {results['avg_random_cosine_sim']}")
        print("\n")


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

with torch.no_grad():
    # # Run for low-frequency comparison
    # print("Low-Frequency Comparison")
    # process_datasets(dataset_paths, mode='low_frequency', cutoff_frequency=30)

    # Run for high-frequency comparison
    print("High-Frequency Comparison")
    process_datasets(dataset_paths, mode='high_frequency', cutoff_frequency=30)

    # # Run for all frequencies comparison (original)
    # print("All-Frequency Comparison")
    # process_datasets(dataset_paths, mode='all')

