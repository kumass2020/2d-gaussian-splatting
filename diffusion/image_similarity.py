import os
import cv2
import numpy as np
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim
from scipy.signal import correlate2d
from sklearn.metrics.pairwise import cosine_similarity


def load_images(directory, prefix):
    """Load images from a directory with a specific prefix."""
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.startswith(prefix) and filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            images.append(image)
    return images


def generate_random_noise_image(shape):
    """Generate a random noise image with N(0,1) distribution and scale it to [0, 255]."""
    random_noise_image = np.random.normal(0, 1, shape)
    # Scale the noise to [0, 255]
    random_noise_image = cv2.normalize(random_noise_image, None, 0, 255, cv2.NORM_MINMAX)
    random_noise_image = random_noise_image.astype(np.uint8)
    return random_noise_image


def calculate_average_similarity(gt_images, rendered_images, random_noise_images):
    """Calculate average similarity metrics between images in the image space."""
    rendered_mse_total = 0
    random_mse_total = 0
    rendered_ssim_total = 0
    random_ssim_total = 0
    rendered_cross_corr_total = 0
    random_cross_corr_total = 0
    rendered_cosine_sim_total = 0
    random_cosine_sim_total = 0

    num_images = len(gt_images)

    for gt_image, rendered_image, random_noise_image in zip(gt_images, rendered_images, random_noise_images):
        # MSE
        rendered_mse = mse(gt_image, rendered_image)
        random_mse = mse(gt_image, random_noise_image)

        # SSIM
        data_range = np.max(gt_image) - np.min(gt_image)
        rendered_ssim = ssim(gt_image, rendered_image, data_range=data_range)
        random_ssim = ssim(gt_image, random_noise_image, data_range=data_range)

        # Cross-Correlation
        rendered_cross_corr = np.max(correlate2d(gt_image, rendered_image, mode='valid'))
        random_cross_corr = np.max(correlate2d(gt_image, random_noise_image, mode='valid'))

        # Cosine Similarity
        rendered_cosine_sim = cosine_similarity(gt_image.flatten().reshape(1, -1),
                                                rendered_image.flatten().reshape(1, -1))[0][0]
        random_cosine_sim = cosine_similarity(gt_image.flatten().reshape(1, -1),
                                              random_noise_image.flatten().reshape(1, -1))[0][0]

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


def process_datasets(dataset_paths):
    """Process all datasets and print average similarity scores."""
    for dataset in dataset_paths:
        gt_dir = os.path.join(dataset, 'gt')
        rendered_dir = os.path.join(dataset, 'vis')

        gt_images = load_images(gt_dir, "")
        rendered_images = load_images(rendered_dir, "noise_")
        random_noise_images = [generate_random_noise_image(img.shape) for img in rendered_images]

        results = calculate_average_similarity(gt_images, rendered_images, random_noise_images)

        print(f"Dataset: {dataset}")
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

# Process all datasets
process_datasets(dataset_paths)
