import os
import torch

epochs = 1
bs = 4

grey_dir = "./grayscale_dataset"
mf_dir = "./mf_dataset"
noisy_dir = "./noisy_dataset"
mf_file_names = os.listdir("./grayscale_dataset")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")