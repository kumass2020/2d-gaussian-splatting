import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim
from scipy.signal import correlate2d
from sklearn.metrics.pairwise import cosine_similarity

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
    return magnitude_spectrum, fft_image_shifted

def low_pass_filter(image, cutoff_frequency):
    """Apply a low-pass filter to keep only low-frequency components."""
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # center
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 1

    fft_image = np.fft.fftshift(np.fft.fft2(image))
    filtered_fft = fft_image * mask
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))

    return filtered_image

def high_pass_filter(image, cutoff_frequency):
    """Apply a high-pass filter to keep only high-frequency components."""
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # center
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0

    fft_image = np.fft.fftshift(np.fft.fft2(image))
    filtered_fft = fft_image * mask
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))

    return filtered_image

def generate_random_noise_image(shape):
    """Generate a random noise image with N(0,1) distribution and scale it to [0, 255]."""
    random_noise_image = np.random.normal(0, 1, shape)
    random_noise_image = cv2.normalize(random_noise_image, None, 0, 255, cv2.NORM_MINMAX)
    random_noise_image = random_noise_image.astype(np.uint8)
    return random_noise_image

def calculate_errors(gt_image, compared_image, error_type):
    """Calculate error map based on the selected error type."""
    if error_type == 'mse':
        error_map = (gt_image - compared_image) ** 2
    elif error_type == 'ssim':
        error_map = 1 - ssim(gt_image, compared_image, data_range=gt_image.max() - gt_image.min(), full=True)[1]
    elif error_type == 'cosine':
        gt_flat = gt_image.flatten().reshape(1, -1)
        compared_flat = compared_image.flatten().reshape(1, -1)
        cosine_sim = cosine_similarity(gt_flat, compared_flat)[0][0]
        error_map = 1 - cosine_sim
        error_map = np.full_like(gt_image, error_map * 255)  # Fill the error map with the cosine error value
    else:
        raise ValueError("Unsupported error type")

    return cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Calculate and visualize errors in the frequency domain using MSE, SSIM, and Cosine Similarity
def visualize_frequency_domain(gt_image, rendered_image, random_image, mode, output_dir):
    """Visualize and save frequency domain plots with different error metrics."""
    # Apply filters based on mode
    if mode == 'low_frequency':
        gt_image = low_pass_filter(gt_image, cutoff_frequency=30)
        rendered_image = low_pass_filter(rendered_image, cutoff_frequency=30)
        random_image = low_pass_filter(random_image, cutoff_frequency=30)
    elif mode == 'high_frequency':
        gt_image = high_pass_filter(gt_image, cutoff_frequency=30)
        rendered_image = high_pass_filter(rendered_image, cutoff_frequency=30)
        random_image = high_pass_filter(random_image, cutoff_frequency=30)

    # FFT and Magnitude Spectrum
    gt_magnitude, gt_fft = apply_fft(gt_image)
    rendered_magnitude, rendered_fft = apply_fft(rendered_image)
    random_magnitude, random_fft = apply_fft(random_image)

    # Visualize Magnitude Spectrum
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.log1p(np.abs(gt_fft)), cmap='gray')
    axes[0].set_title("GT Frequency Magnitude")
    axes[1].imshow(np.log1p(np.abs(rendered_fft)), cmap='gray')
    axes[1].set_title("Rendered Noise Frequency Magnitude")
    axes[2].imshow(np.log1p(np.abs(random_fft)), cmap='gray')
    axes[2].set_title("Random Noise Frequency Magnitude")
    plt.savefig(os.path.join(output_dir, f'frequency_magnitude_{mode}.png'))
    plt.show()

    # Calculate errors in the frequency domain
    mse_rendered = np.square(np.abs(gt_fft - rendered_fft))
    mse_random = np.square(np.abs(gt_fft - random_fft))

    ssim_rendered = 1 - ssim(np.abs(gt_fft), np.abs(rendered_fft), data_range=np.max(gt_fft) - np.min(gt_fft))
    ssim_random = 1 - ssim(np.abs(gt_fft), np.abs(random_fft), data_range=np.max(gt_fft) - np.min(gt_fft))

    cosine_rendered = 1 - cosine_similarity(np.abs(gt_fft).flatten().reshape(1, -1),
                                            np.abs(rendered_fft).flatten().reshape(1, -1))[0][0]
    cosine_random = 1 - cosine_similarity(np.abs(gt_fft).flatten().reshape(1, -1),
                                          np.abs(random_fft).flatten().reshape(1, -1))[0][0]

    cosine_rendered_map = np.full_like(np.abs(gt_fft), cosine_rendered * 255)
    cosine_random_map = np.full_like(np.abs(gt_fft), cosine_random * 255)

    # Visualize errors
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # MSE Visualization
    axes[0, 0].imshow(np.log1p(mse_rendered), cmap='hot')
    axes[0, 0].set_title("MSE Error GT vs Rendered Noise")
    axes[0, 1].imshow(np.log1p(mse_random), cmap='hot')
    axes[0, 1].set_title("MSE Error GT vs Random Noise")

    # SSIM Visualization
    axes[1, 0].imshow(np.log1p(np.abs(gt_fft) * (1 - ssim_rendered)), cmap='hot')
    axes[1, 0].set_title("SSIM Error GT vs Rendered Noise")
    axes[1, 1].imshow(np.log1p(np.abs(gt_fft) * (1 - ssim_random)), cmap='hot')
    axes[1, 1].set_title("SSIM Error GT vs Random Noise")

    # Cosine Similarity Visualization
    axes[2, 0].imshow(np.log1p(cosine_rendered_map), cmap='hot')
    axes[2, 0].set_title("Cosine Similarity Error GT vs Rendered Noise")
    axes[2, 1].imshow(np.log1p(cosine_random_map), cmap='hot')
    axes[2, 1].set_title("Cosine Similarity Error GT vs Random Noise")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'frequency_errors_{mode}.png'))
    plt.show()

def visualize_image_space_errors(gt_image, rendered_image, random_image, error_types, output_dir, mode):
    """Visualize and save error maps in image space for different error types."""
    # Apply the correct filters to the images for low/high frequency
    if mode == 'low_frequency':
        gt_image = low_pass_filter(gt_image, cutoff_frequency=30)
        rendered_image = low_pass_filter(rendered_image, cutoff_frequency=30)
        random_image = low_pass_filter(random_image, cutoff_frequency=30)
    elif mode == 'high_frequency':
        gt_image = high_pass_filter(gt_image, cutoff_frequency=30)
        rendered_image = high_pass_filter(rendered_image, cutoff_frequency=30)
        random_image = high_pass_filter(random_image, cutoff_frequency=30)

    fig, axes = plt.subplots(len(error_types), 5, figsize=(20, 5 * len(error_types)))

    for i, error_type in enumerate(error_types):
        error_rendered_map = calculate_errors(gt_image, rendered_image, error_type)
        error_random_map = calculate_errors(gt_image, random_image, error_type)

        axes[i, 0].imshow(gt_image, cmap='gray')
        axes[i, 0].set_title(f"GT Image ({mode})")
        axes[i, 1].imshow(rendered_image, cmap='gray')
        axes[i, 1].set_title(f"Rendered Noise ({mode})")
        axes[i, 2].imshow(random_image, cmap='gray')
        axes[i, 2].set_title(f"Random Noise ({mode})")
        axes[i, 3].imshow(error_rendered_map, cmap='hot')
        axes[i, 3].set_title(f"Error GT vs Rendered ({error_type.upper()})")
        axes[i, 4].imshow(error_random_map, cmap='hot')
        axes[i, 4].set_title(f"Error GT vs Random ({error_type.upper()})")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'image_space_errors_{mode}.png'))
    plt.show()

# Main function to sample and visualize
def process_and_visualize(datasets, mode='all', error_types=None, random_number=None):
    """Sample one image from each dataset, process it, and visualize errors."""
    global random_noise_image

    if error_types is None:
        error_types = ['mse', 'ssim', 'cosine']  # Default to all error types

    for dataset_path in datasets:
        output_dir = os.path.join(dataset_path, f"output_{mode}")
        os.makedirs(output_dir, exist_ok=True)

        gt_dir = os.path.join(dataset_path, 'gt')
        rendered_dir = os.path.join(dataset_path, 'vis')

        # Load one sample image
        gt_images = load_images(gt_dir, "")
        rendered_images = load_images(rendered_dir, "noise_")

        # Sample random image index if not provided
        if random_number is None:
            random_number = np.random.randint(0, len(gt_images))

        gt_image = gt_images[random_number]
        rendered_image = rendered_images[random_number]
        if random_noise_image is None:
            random_noise_image = generate_random_noise_image(gt_image.shape)

        # Visualize in frequency domain
        visualize_frequency_domain(gt_image, rendered_image, random_noise_image, mode, output_dir)

        # Visualize in image space with different error types
        visualize_image_space_errors(gt_image, rendered_image, random_noise_image, error_types, output_dir, mode)

prefix_date = "../output_render/240823-0157/"
datasets = [
    os.path.join(prefix_date, "bicycle/test/ours_20000"),
    # os.path.join(prefix_date, "bonsai/test/ours_20000"),
    # os.path.join(prefix_date, "counter/test/ours_20000"),
    # os.path.join(prefix_date, "flowers/test/ours_20000"),
    # os.path.join(prefix_date, "garden/test/ours_20000"),
    # os.path.join(prefix_date, "kitchen/test/ours_20000"),
    # os.path.join(prefix_date, "room/test/ours_20000"),
    # os.path.join(prefix_date, "stump/test/ours_20000"),
    # os.path.join(prefix_date, "treehill/test/ours_20000"),
]

plt.rc('font', size=13)  # 기본 폰트 크기
plt.rc('axes', labelsize=10)  # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=10)  # x축 눈금 폰트 크기
plt.rc('ytick', labelsize=10)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=10)  # 범례 폰트 크기
plt.rc('figure', titlesize=13)  # figure title 폰트 크기

# Select a random image index
random_number = np.random.randint(0, len(load_images(os.path.join(datasets[0], 'gt'), "")))
random_noise_image = None

# Example Usage for multiple datasets with the same image index across all modes
process_and_visualize(datasets, mode='all', error_types=['mse', 'ssim', 'cosine'], random_number=random_number)
process_and_visualize(datasets, mode='low_frequency', error_types=['mse', 'ssim', 'cosine'], random_number=random_number)
process_and_visualize(datasets, mode='high_frequency', error_types=['mse', 'ssim', 'cosine'], random_number=random_number)
