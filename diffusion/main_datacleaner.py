import os
from PIL import Image

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

# Verify and clean images in the real_images directory
real_images_dir = '../real_images'  # Ensure this is the correct path
verify_and_clean_images(real_images_dir)
