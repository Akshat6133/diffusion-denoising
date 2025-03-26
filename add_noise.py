import os
import cv2
import numpy as np


root_dir = "./mf_dataset"
mf_file_names = os.listdir("./mf_dataset")

# Create a new directory for the noisy dataset
noisy_dir = "./dataset_noisy"
os.makedirs(noisy_dir, exist_ok=True)

# Function to add Gaussian noise to an image
def add_gaussian_noise(image, mean=0, stddev=25):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

    # Function to add salt-and-pepper noise to an image
    def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
        noisy_image = image.copy()
        total_pixels = image.size
        num_salt = int(total_pixels * salt_prob)
        num_pepper = int(total_pixels * pepper_prob)

        # Add salt noise
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1]] = 255

        # Add pepper noise
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1]] = 0

        return noisy_image

    # Function to add Poisson noise to an image
    def add_poisson_noise(image):
        noisy_image = np.random.poisson(image).astype(np.float32)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    # Function to add speckle noise to an image
    def add_speckle_noise(image, mean=0, stddev=0.1):
        noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + image.astype(np.float32) * noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Process each file in the dataset
for file_name in mf_file_names:
    file_path = os.path.join(root_dir, file_name)
    if os.path.isfile(file_path):
        # Read the image
        image = cv2.imread(file_path)
        if image is not None:
            # Add Gaussian noise
            noisy_image = add_gaussian_noise(image)
            # Save the noisy image to the new directory
            noisy_file_path = os.path.join(noisy_dir, file_name)
            cv2.imwrite(noisy_file_path, noisy_image)