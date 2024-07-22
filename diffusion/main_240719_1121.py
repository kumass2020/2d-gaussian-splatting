import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms
from tqdm import tqdm
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from torch_fidelity import calculate_metrics

# Set model ID
model_id = "google/ddpm-ema-church-256"

# Initialize scheduler and model
scheduler = DDPMScheduler.from_pretrained(model_id)
model = UNet2DModel.from_pretrained(model_id).to("cuda")

scheduler.set_timesteps(50)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")

def verify_and_clean_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(directory, filename)
            try:
                with Image.open(filepath) as img:
                    img.verify()  # Verify if image is intact
            except (IOError, SyntaxError) as e:
                print(f"Removing corrupted image: {filename}")
                os.remove(filepath)

# Custom dataset class to handle flat directory structure
class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # The second item is a dummy label

# Function to preprocess and save images
def preprocess_and_save_images(dataset_dir, output_dir, sample_size):
    transform = transforms.Compose([
        transforms.Resize((sample_size, sample_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = FlatImageDataset(root_dir=dataset_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)  # Use num_workers=0 to avoid multiprocessing issues

    os.makedirs(output_dir, exist_ok=True)

    for i, (img, _) in enumerate(loader):
        for j in range(img.size(0)):
            img_pil = transforms.ToPILImage()(img[j])
            img_pil.save(os.path.join(output_dir, f'image_{i * img.size(0) + j}.png'))

# Verify and clean images in the real_images directory
real_images_dir = 'real_images'
verify_and_clean_images(real_images_dir)

# Preprocess and save images
preprocessed_real_images_dir = 'preprocessed_real_images'
preprocess_and_save_images(real_images_dir, preprocessed_real_images_dir, sample_size)

# Function to normalize noise in 16x16 tiles
def normalize_tiles(image_tensor, tile_size=16):
    c, h, w = image_tensor.shape
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = image_tensor[:, i:i+tile_size, j:j+tile_size]
            mean = tile.mean(dim=[1, 2], keepdim=True)
            std = tile.std(dim=[1, 2], keepdim=True)
            image_tensor[:, i:i+tile_size, j:j+tile_size] = (tile - mean) / (std + 1e-8)
    return image_tensor

# Function to generate and save images
def generate_and_save_images(model, scheduler, noise=None, noise_dir=None, num_images=100, sample_size=256,
                             output_dir='output_dm', use_normalization=True, use_normalize_tiles=False):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    use_predefined_noise = noise is not None

    for i in tqdm(range(num_images), desc="Generating images"):
        if use_predefined_noise:
            input = noise.to(device)
        else:
            noise_path = os.path.join(noise_dir, f'noise_{i + 1:05d}.png')
            noise_image = Image.open(noise_path)
            if noise_image.mode != 'RGB':
                noise_image = noise_image.convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((sample_size, sample_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            input = transform(noise_image).unsqueeze(0).to(device)

            if use_normalize_tiles:
                input = normalize_tiles(input.squeeze(0)).unsqueeze(0)

            if use_normalization:
                # Calculate mean and std of the input tensor
                mean = input.mean([0, 2, 3])
                std = input.std([0, 2, 3])
                normalize = transforms.Normalize(mean, std)
                input = normalize(input[0]).unsqueeze(0).to(device)

        for t in scheduler.timesteps:
            with torch.no_grad():
                noisy_residual = model(input, t).sample
            previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
            input = previous_noisy_sample

        image = (input / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype("uint8"))
        image.save(f"{output_dir}/image_{i}.png")

    torch.cuda.empty_cache()  # Clean up CUDA memory

# Function to plot 25 images in a 5x5 grid
def plot_images_grid(output_dir='output_dm', num_images=25):
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(num_images):
        img = mpimg.imread(f'{output_dir}/image_{i}.png')
        ax = axs[i // 5, i % 5]
        ax.imshow(img)
        ax.axis('off')

    # Adjust layout to remove any excess margins
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    plt.show()

# Function to calculate FID and IS
def calculate_fid_is(generated_dir, real_dir):
    metrics = calculate_metrics(input1=generated_dir, input2=real_dir, cuda=True, fid=True, isc=True, prc=True, verbose=True)
    fid_value = metrics['frechet_inception_distance']
    is_value = metrics['inception_score_mean']
    return fid_value, is_value

# Generate and save images with totally random noise
random_noise_output_dir = 'output_dm/random_noise'
generate_and_save_images(model, scheduler, noise=torch.randn(1, 3, sample_size, sample_size).to("cuda"),
                         num_images=100, sample_size=256, output_dir=random_noise_output_dir, use_normalization=False)

# Generate and save images with predefined noise
predefined_noise_output_dir = 'output_dm/predefined_noise'
generate_and_save_images(model, scheduler, noise=None, noise_dir='output/240718-2337/train/ours_30000/vis',
                         num_images=100, sample_size=256, output_dir=predefined_noise_output_dir, use_normalization=True)

# Generate and save images with predefined noise without normalization
predefined_noise_nn_output_dir = 'output_dm/predefined_noise_nn'
generate_and_save_images(model, scheduler, noise=None, noise_dir='output/240718-2337/train/ours_30000/vis',
                         num_images=100, sample_size=256, output_dir=predefined_noise_nn_output_dir, use_normalization=False)

# # Generate and save images with tile-normalized noise
# tile_normalized_noise_output_dir = 'output_dm/tile_normalized_noise'
# generate_and_save_images(model, scheduler, noise=None, noise_dir='output/240718-2337/train/ours_30000/vis',
#                          num_images=100, sample_size=256, output_dir=tile_normalized_noise_output_dir, use_normalize_tiles=True)

# Calculate FID and IS for random noise images
fid_random, is_random = calculate_fid_is(random_noise_output_dir, preprocessed_real_images_dir)
print(f"Random Noise - FID: {fid_random}, IS: {is_random}")

# Calculate FID and IS for predefined noise images
fid_predefined, is_predefined = calculate_fid_is(predefined_noise_output_dir, preprocessed_real_images_dir)
print(f"Predefined Noise - FID: {fid_predefined}, IS: {is_predefined}")

# Calculate FID and IS for predefined noise images without normalization
fid_predefined_nn, is_predefined_nn = calculate_fid_is(predefined_noise_nn_output_dir, preprocessed_real_images_dir)
print(f"Predefined Noise without normalization - FID: {fid_predefined_nn}, IS: {is_predefined_nn}")

# # Calculate FID and IS for tile-normalized noise images
# fid_tile_normalized, is_tile_normalized = calculate_fid_is(tile_normalized_noise_output_dir, preprocessed_real_images_dir)
# print(f"Tile Normalized Noise - FID: {fid_tile_normalized}, IS: {is_tile_normalized}")

# Plot images
plot_images_grid(random_noise_output_dir, num_images=25)
plot_images_grid(predefined_noise_output_dir, num_images=25)
plot_images_grid(predefined_noise_nn_output_dir, num_images=25)
# plot_images_grid(tile_normalized_noise_output_dir, num_images=25)
