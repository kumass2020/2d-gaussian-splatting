import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Assuming InstructPix2Pix is imported from your previous code
from diffusion.orig.ip2p import InstructPix2Pix

# Load the InstructPix2Pix model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ip2p = InstructPix2Pix(device=device, num_train_timesteps=1000)


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Assuming InstructPix2Pix is imported from your previous code
from diffusion.orig.ip2p import InstructPix2Pix

# Load the InstructPix2Pix model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ip2p = InstructPix2Pix(device=device, num_train_timesteps=1000)


# Loading and processing functions
def load_images(directory, prefix):
    """Load RGB images from a directory with a specific prefix."""
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.startswith(prefix) and filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)  # Load as RGB
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)  # Scale to [0, 255]
            images.append(image)
    return images


def apply_fft(latent):
    """Apply FFT to the latent tensor and return the magnitude spectrum."""
    latent = latent.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and move to CPU
    fft_latent_shifted = np.fft.fftshift(np.fft.fftn(latent, axes=(-2, -1)))  # Apply FFT on the last two dimensions (H, W)
    magnitude_spectrum = np.log1p(np.abs(fft_latent_shifted))  # Use log to enhance visibility of smaller values
    return magnitude_spectrum, fft_latent_shifted


def low_pass_filter(latent, cutoff_frequency):
    """Apply a low-pass filter to the latent space using FFT."""
    latent = latent.squeeze(0).detach().cpu().numpy()
    rows, cols = latent.shape[1:3]
    crow, ccol = rows // 2, cols // 2  # center
    mask = np.zeros((latent.shape[0], rows, cols), np.uint8)
    mask[..., crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 1

    # Apply the mask in the Fourier domain for all channels at once
    fft_latent = np.fft.fftshift(np.fft.fftn(latent, axes=(-2, -1)))
    filtered_fft = fft_latent * mask[None, ...]
    filtered_latent = np.abs(np.fft.ifftn(np.fft.ifftshift(filtered_fft), axes=(-2, -1)))

    return torch.tensor(filtered_latent).unsqueeze(0).to(device)


def high_pass_filter(latent, cutoff_frequency):
    """Apply a high-pass filter to the latent space using FFT."""
    latent = latent.squeeze(0).detach().cpu().numpy()
    rows, cols = latent.shape[1:3]
    crow, ccol = rows // 2, cols // 2  # center
    mask = np.ones((latent.shape[0], rows, cols), np.uint8)
    mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0

    # Apply the mask in the Fourier domain for all channels at once
    fft_latent = np.fft.fftshift(np.fft.fftn(latent, axes=(-2, -1)))
    filtered_fft = fft_latent * mask[None, ...]  # Apply the mask along the last axis (channels)
    filtered_latent = np.abs(np.fft.ifftn(np.fft.ifftshift(filtered_fft), axes=(-2, -1)))

    return torch.tensor(filtered_latent).unsqueeze(0).to(device)


def generate_random_noise_image(shape):
    """Generate a random RGB noise image with N(0,1) distribution and scale it to [0, 255]."""
    random_noise_image = np.random.normal(0, 1, shape)
    random_noise_image = cv2.normalize(random_noise_image, None, 0, 255, cv2.NORM_MINMAX)
    random_noise_image = random_noise_image.astype(np.uint8)
    return random_noise_image


def calculate_errors(gt_image, compared_image, error_type):
    """Calculate error map based on the selected error type for each RGB channel."""
    error_map = np.zeros_like(gt_image, dtype=np.float32)
    for c in range(3):  # Only for RGB channels
        if error_type == 'mse':
            error_map[:, :, c] = (gt_image[:, :, c] - compared_image[:, :, c]) ** 2
        elif error_type == 'ssim':
            error_map[:, :, c] = 1 - ssim(gt_image[:, :, c], compared_image[:, :, c],
                                          data_range=gt_image.max() - gt_image.min(), full=True)[1]
        elif error_type == 'cosine':
            gt_flat = gt_image[:, :, c].flatten().reshape(1, -1)
            compared_flat = compared_image[:, :, c].flatten().reshape(1, -1)
            cosine_sim = cosine_similarity(gt_flat, compared_flat)[0][0]
            error_map[:, :, c] = 1 - cosine_sim
        else:
            raise ValueError("Unsupported error type")

    return cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def latents_to_image(latents):
    """Convert latents back to images using the InstructPix2Pix VAE."""
    with torch.no_grad():
        if len(latents.shape) == 5:
            latents = latents.squeeze(0)
        decoded_img = ip2p.latents_to_img(latents.to(ip2p.auto_encoder.dtype))
    decoded_img = decoded_img.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    return decoded_img[0].cpu().numpy() * 255  # Remove batch dimension and scale to [0, 255]


def image_to_latents(image):
    """Convert images to latents using the InstructPix2Pix VAE."""
    image_tensor = torch.tensor(image).unsqueeze(0).to(device)  # Add batch dimension
    image_tensor = image_tensor.permute(0, 3, 1, 2).float() / 255.0  # (B, H, W, C) -> (B, C, H, W), normalize to [0, 1]
    image_tensor = image_tensor.to(ip2p.auto_encoder.dtype)
    return ip2p.imgs_to_latent(image_tensor)


def normalize_for_visualization(image):
    """Normalize images for visualization, clip negative values to zero."""
    image = np.real(image)  # Remove any imaginary components if present.
    image = np.clip(image, 0, None)  # Clip any negative values to zero
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def visualize_frequency_domain(gt_image, rendered_image, random_image, mode, output_dir):
    """Visualize and save frequency domain plots with different error metrics."""

    # Convert images to latents
    gt_latent = image_to_latents(gt_image)
    rendered_latent = image_to_latents(rendered_image)
    random_latent = image_to_latents(random_image)

    # Apply filters based on mode
    if mode == 'low_frequency':
        gt_latent = low_pass_filter(gt_latent, cutoff_frequency=30).squeeze(0)
        rendered_latent = low_pass_filter(rendered_latent, cutoff_frequency=30).squeeze(0)
        random_latent = low_pass_filter(random_latent, cutoff_frequency=30).squeeze(0)
    elif mode == 'high_frequency':
        gt_latent = high_pass_filter(gt_latent, cutoff_frequency=30).squeeze(0)
        rendered_latent = high_pass_filter(rendered_latent, cutoff_frequency=30).squeeze(0)
        random_latent = high_pass_filter(random_latent, cutoff_frequency=30).squeeze(0)

    # FFT and Magnitude Spectrum
    gt_magnitude, gt_fft = apply_fft(gt_latent)
    rendered_magnitude, rendered_fft = apply_fft(rendered_latent)
    random_magnitude, random_fft = apply_fft(random_latent)

    print(f"GT FFT min/max: {np.min(gt_fft)} / {np.max(gt_fft)}")
    print(f"Rendered FFT min/max: {np.min(rendered_fft)} / {np.max(rendered_fft)}")
    print(f"Random FFT min/max: {np.min(random_fft)} / {np.max(random_fft)}")

    # Initialize arrays to hold frequency errors
    mse_rendered_total = np.abs(gt_fft - rendered_fft) ** 2
    mse_random_total = np.abs(gt_fft - random_fft) ** 2

    # SSIM between the GT and rendered FFT for all channels at once
    ssim_rendered = 1 - ssim(np.abs(gt_fft), np.abs(rendered_fft), data_range=np.max(gt_fft) - np.min(gt_fft), full=False, channel_axis=0)
    ssim_random = 1 - ssim(np.abs(gt_fft), np.abs(random_fft), data_range=np.max(gt_fft) - np.min(gt_fft), full=False, channel_axis=0)

    # Cosine Similarity Error
    cosine_rendered = 1 - cosine_similarity(np.abs(gt_fft).reshape(1, -1), np.abs(rendered_fft).reshape(1, -1))[0][0]
    cosine_random = 1 - cosine_similarity(np.abs(gt_fft).reshape(1, -1), np.abs(random_fft).reshape(1, -1))[0][0]

    # Convert frequency errors back to image space
    mse_rendered_image = np.real(np.fft.ifftn(mse_rendered_total, axes=(-2, -1)))
    mse_random_image = np.real(np.fft.ifftn(mse_random_total, axes=(-2, -1)))
    ssim_rendered_image = np.real(np.fft.ifftn(ssim_rendered * np.abs(gt_fft), axes=(-2, -1)))
    ssim_random_image = np.real(np.fft.ifftn(ssim_random * np.abs(gt_fft), axes=(-2, -1)))
    cosine_rendered_image = np.full_like(gt_image[..., 0], cosine_rendered * 255)
    cosine_random_image = np.full_like(gt_image[..., 0], cosine_random * 255)

    # Combine channels (e.g., sum or average) to convert to 2D image
    mse_rendered_image = mse_rendered_image.sum(axis=0)  # Summing over channels
    mse_random_image = mse_random_image.sum(axis=0)
    ssim_rendered_image = ssim_rendered_image.sum(axis=0)
    ssim_random_image = ssim_random_image.sum(axis=0)

    # Normalize images for visualization
    mse_rendered_image = normalize_for_visualization(mse_rendered_image)
    mse_random_image = normalize_for_visualization(mse_random_image)
    ssim_rendered_image = normalize_for_visualization(ssim_rendered_image)
    ssim_random_image = normalize_for_visualization(ssim_random_image)

    # Visualize the errors in image space
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # MSE Image Space Visualization
    axes[0, 0].imshow(mse_rendered_image, cmap='gray')
    axes[0, 0].set_title("MSE Error (Image Space) GT vs Rendered Noise")
    axes[0, 1].imshow(mse_random_image, cmap='gray')
    axes[0, 1].set_title("MSE Error (Image Space) GT vs Random Noise")

    # SSIM Image Space Visualization
    axes[1, 0].imshow(ssim_rendered_image, cmap='gray')
    axes[1, 0].set_title("SSIM Error (Image Space) GT vs Rendered Noise")
    axes[1, 1].imshow(ssim_random_image, cmap='gray')
    axes[1, 1].set_title("SSIM Error (Image Space) GT vs Random Noise")

    # Cosine Similarity Image Space Visualization
    axes[2, 0].imshow(cosine_rendered_image, cmap='gray')
    axes[2, 0].set_title("Cosine Similarity Error (Image Space) GT vs Rendered Noise")
    axes[2, 1].imshow(cosine_random_image, cmap='gray')
    axes[2, 1].set_title("Cosine Similarity Error (Image Space) GT vs Random Noise")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'image_space_errors_total_{mode}.png'))
    plt.show()




def visualize_image_space_errors(gt_image, rendered_image, random_image, error_types, output_dir, mode):
    """Visualize and save error maps in image space for different error types using latents."""

    # Convert images to latents
    gt_latent = image_to_latents(gt_image)
    rendered_latent = image_to_latents(rendered_image)
    random_latent = image_to_latents(random_image)

    # Apply the correct filters to the latents for low/high frequency
    if mode == 'low_frequency':
        gt_latent = low_pass_filter(gt_latent, cutoff_frequency=30)
        rendered_latent = low_pass_filter(rendered_latent, cutoff_frequency=30)
        random_latent = low_pass_filter(random_latent, cutoff_frequency=30)
    elif mode == 'high_frequency':
        gt_latent = high_pass_filter(gt_latent, cutoff_frequency=30)
        rendered_latent = high_pass_filter(rendered_latent, cutoff_frequency=30)
        random_latent = high_pass_filter(random_latent, cutoff_frequency=30)

    fig, axes = plt.subplots(len(error_types), 5, figsize=(20, 5 * len(error_types)))

    # Decode the latents back to image space
    decoded_gt_image = latents_to_image(gt_latent)
    decoded_rendered_image = latents_to_image(rendered_latent)
    decoded_random_image = latents_to_image(random_latent)

    # Convert to float32 for compatibility with imshow
    decoded_gt_image = decoded_gt_image.astype(np.float32)
    decoded_rendered_image = decoded_rendered_image.astype(np.float32)
    decoded_random_image = decoded_random_image.astype(np.float32)

    for i, error_type in enumerate(error_types):
        # Calculate errors in latent space
        error_rendered_map = calculate_errors(decoded_gt_image, decoded_rendered_image, error_type)
        error_random_map = calculate_errors(decoded_gt_image, decoded_random_image, error_type)

        # Convert errors to a supported type (float32 or uint8)
        error_rendered_map = error_rendered_map.astype(np.float32)
        error_random_map = error_random_map.astype(np.float32)

        # Visualize the decoded images and their error maps
        axes[i, 0].imshow(decoded_gt_image, cmap='gray')
        axes[i, 0].set_title(f"Decoded GT Image ({mode})")
        axes[i, 1].imshow(decoded_rendered_image, cmap='gray')
        axes[i, 1].set_title(f"Decoded Rendered Noise ({mode})")
        axes[i, 2].imshow(decoded_random_image, cmap='gray')
        axes[i, 2].set_title(f"Decoded Random Noise ({mode})")
        axes[i, 3].imshow(error_rendered_map, cmap='hot')
        axes[i, 3].set_title(f"Error GT vs Rendered ({error_type.upper()})")
        axes[i, 4].imshow(error_random_map, cmap='hot')
        axes[i, 4].set_title(f"Error GT vs Random ({error_type.upper()})")

    plt.tight_layout()

    # Ensure the figure is saved with correct dtype
    plt.savefig(os.path.join(output_dir, f'image_space_errors_{mode}.png'), dpi=300)
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
