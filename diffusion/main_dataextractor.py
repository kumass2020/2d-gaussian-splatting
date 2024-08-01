import os
import lmdb
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image


def extract_lmdb_images(lmdb_path, output_dir, num_images=1000):
    os.makedirs(output_dir, exist_ok=True)

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(tqdm(cursor, total=num_images, desc="Extracting images")):
            if i >= num_images:
                break
            img = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.save(os.path.join(output_dir, f'image_{i}.png'))


lmdb_path = '../data/church_outdoor_train_lmdb'
output_dir = '../real_images'
extract_lmdb_images(lmdb_path, output_dir, num_images=1000)
