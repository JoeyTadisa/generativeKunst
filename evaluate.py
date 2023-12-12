# import os
# import random
# import torch
# import scipy
# import torchvision
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from PIL import Image
# from pytorch_msssim import ms_ssim
# import torch.nn.functional as F
# from scipy.stats import chisquare, wasserstein_distance

# class ImageComparator:
#     def __init__(self, original_directories, generated_directories):
#         self.original_images = self.load_images(original_directories)
#         self.generated_images = self.load_images(generated_directories)

#         self.original_histogram = self.compute_histograms(self.original_images)
#         self.generated_histogram = self.compute_histograms(self.generated_images)

#     def load_images(self, directories):
#         images = []
#         num_images = None

#         # Find the minimum number of images across all directories
#         for directory in directories:
#             images_in_directory = [filename for filename in os.listdir(directory) if
#                                    filename.endswith(('.jpg', '.jpeg', '.png'))]
#             num_images_in_directory = len(images_in_directory)

#             if num_images is None or num_images_in_directory < num_images:
#                 num_images = num_images_in_directory

#         # Load images from each directory, limiting larger directories to the number of images in the smallest directory
#         for directory in directories:
#             images_in_directory = [filename for filename in os.listdir(directory) if
#                                    filename.endswith(('.jpg', '.jpeg', '.png'))]

#             # Randomly select num_images from the current directory
#             selected_images = np.random.choice(images_in_directory, size=num_images, replace=False)

#             for filename in selected_images:
#                 image_path = os.path.join(directory, filename)
#                 img = Image.open(image_path).convert('RGB').resize((256, 256))
#                 img = np.array(img)

#                 # Ensure all images have the same valid shape and color space
#                 if img.shape[0] > 0 and img.shape[1] > 0 and img.shape[2] == 3:
#                     images.append(img)

#         return np.array(images)
    
#     def sample_image_pairs(self, num_pairs):
#         if num_pairs is None:
#             return self.original_images, self.generated_images

#         sampled_original_images = random.sample(list(self.original_images), num_pairs)
#         sampled_generated_images = random.sample(list(self.generated_images), num_pairs)

#         return np.array(sampled_original_images), np.array(sampled_generated_images)

#     def compute_histograms(self, images):
#         # Assuming images is a 4D numpy array (num_images, height, width, channels)
#         histograms = []
#         for image in images:
#             histograms.append([np.histogram(channel.flatten(), bins=256, range=(0, 256))[0] for channel in image])

#         return histograms

#     def compare_histograms(self, metric='chi-square'):
#         if self.original_images.shape[-1] != self.generated_images.shape[-1]:
#             raise ValueError("Original and generated images must have the same number of channels.")
#         if self.original_images.shape[1:-1] != self.generated_images.shape[1:-1]:
#             raise ValueError("Original and generated images must have the same dimensions.")

#         if metric == 'chi-square':
#             # Compute histograms for each channel
#             original_histograms = [np.histogram(channel.flatten(), bins=256, range=(0, 256))[0] for channel in np.moveaxis(self.original_images, -1, 0)]
#             generated_histograms = [np.histogram(channel.flatten(), bins=256, range=(0, 256))[0] for channel in np.moveaxis(self.generated_images, -1, 0)]

#             # Concatenate histograms for each channel
#             original_histogram_flat = np.concatenate(original_histograms, axis=None)
#             generated_histogram_flat = np.concatenate(generated_histograms, axis=None)

#             # Ensure histograms have the same shape
#             min_len = min(len(original_histogram_flat), len(generated_histogram_flat))
#             original_histogram_flat = original_histogram_flat / np.sum(original_histogram_flat)
#             generated_histogram_flat = generated_histogram_flat / np.sum(generated_histogram_flat)
#             distance, _ = chisquare(original_histogram_flat, f_exp=generated_histogram_flat)
#         elif metric == 'earth-mover':
#             # Earth Mover's Distance (EMD)
#             original_histogram_flat = np.concatenate([channel.flatten() for channel in np.moveaxis(self.original_images, -1, 0)])
#             generated_histogram_flat = np.concatenate([channel.flatten() for channel in np.moveaxis(self.generated_images, -1, 0)])
#             distance = wasserstein_distance(original_histogram_flat, generated_histogram_flat)
#         elif metric == 'histogram-intersection':
#             # Histogram Intersection
#             original_histogram_flat = np.concatenate(self.original_histogram).flatten()
#             generated_histogram_flat = np.concatenate(self.generated_histogram).flatten()
#             min_len = min(len(original_histogram_flat), len(generated_histogram_flat))
#             original_histogram_flat = original_histogram_flat[:min_len]
#             generated_histogram_flat = generated_histogram_flat[:min_len]
#             minima = np.minimum(original_histogram_flat, generated_histogram_flat)
#             intersection = np.true_divide(np.sum(minima), np.sum(generated_histogram_flat))
#             distance = 1 - intersection
#         else:
#             raise ValueError("Invalid metric. Supported metrics: 'chi-square', 'earth-mover', 'histogram-intersection'.")

#         return distance

#     def calculate_msssim_torch(self):
#         msssim_values = []

#         # Create data loaders for the original and generated images
#         original_loader = DataLoader(self.original_images, batch_size=1)
#         generated_loader = DataLoader(self.generated_images, batch_size=1)

#         # Iterate over all combinations of original and generated images
#         for original_imgs in tqdm(original_loader, desc="Calculating MS-SSIM"):
#             for generated_imgs in generated_loader:
#                 # The images are already PyTorch tensors, so no need to convert
#                 original_imgs_torch = original_imgs.float()
#                 generated_imgs_torch = generated_imgs.float()

#                 # Resize the images if necessary
#                 if original_imgs_torch.shape[-1] < 161 or original_imgs_torch.shape[-2] < 161:
#                     original_imgs_torch = F.interpolate(original_imgs_torch, size=(161, 161))
#                     generated_imgs_torch = F.interpolate(generated_imgs_torch, size=(161, 161))

#                 # Calculate the MS-SSIM
#                 msssim = ms_ssim(original_imgs_torch, generated_imgs_torch, data_range=255.0)
#                 msssim_values.append(msssim.item())

#         return np.array(msssim_values)
    
#     def calculate_fid_torch(self):
#         fid_values = []

#         # Create data loaders for the original and generated images
#         original_loader = DataLoader(self.original_images, batch_size=1)
#         generated_loader = DataLoader(self.generated_images, batch_size=1)

#         # Initialize the Inception model
#         inception_model = torchvision.models.inception_v3(pretrained=True).eval()

#         # Iterate over all combinations of original and generated images
#         for original_imgs in tqdm(original_loader, desc="Calculating FID"):
#             for generated_imgs in generated_loader:
#                 # The images are already PyTorch tensors, so no need to convert
#                 original_imgs_torch = original_imgs.float()
#                 generated_imgs_torch = generated_imgs.float()

#                 # Resize the images if necessary
#                 if original_imgs_torch.shape[-1] < 299 or original_imgs_torch.shape[-2] < 299:
#                     original_imgs_torch = F.interpolate(original_imgs_torch, size=(299, 299))
#                     generated_imgs_torch = F.interpolate(generated_imgs_torch, size=(299, 299))

#                 # Calculate the features from the Inception model
#                 original_features = inception_model(original_imgs_torch)
#                 generated_features = inception_model(generated_imgs_torch)

#                 # Calculate the mean and covariance of the features
#                 original_mean, original_cov = np.mean(original_features, axis=0), np.cov(original_features, rowvar=False)
#                 generated_mean, generated_cov = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

#                 # Calculate the FID
#                 mean_diff = np.sum((original_mean - generated_mean) ** 2.0)
#                 cov_sqrt, _ = scipy.linalg.sqrtm(original_cov.dot(generated_cov), disp=False)

#                 if not np.iscomplexobj(cov_sqrt):
#                     trace_cov_sqrt = np.trace(cov_sqrt)
#                 else:
#                     trace_cov_sqrt = np.trace(cov_sqrt.real)

#                 fid = mean_diff + np.trace(original_cov) + np.trace(generated_cov) - 2 * trace_cov_sqrt
#                 fid_values.append(fid)

#         return np.array(fid_values)

#     def plot_histograms(self, save_directory):
#         # Ensure save_directory exists
#         os.makedirs(save_directory, exist_ok=True)

#         for i, (original_histogram, generated_histogram) in enumerate(zip(self.original_histogram, self.generated_histogram)):
#             fig, ax = plt.subplots(3, 1, figsize=(10, 6))

#             for j in range(3):  # For each color channel
#                 ax[j].bar(range(256), original_histogram[j], alpha=0.7, label='Original')
#                 ax[j].bar(range(256), generated_histogram[j], alpha=0.7, label='Generated')
#                 ax[j].set_xlim([0, 256])
#                 ax[j].legend(loc='upper right')

#             plt.tight_layout()
#             plt.savefig(os.path.join(save_directory, f'histogram_{i}.png'))
#             plt.close(fig)


# # Example usage:
# original_directories = ['./data/full-finetune/all_kunst/']
# generated_directories = ['./outputs/lora/test_batch_21_11_23/']

# image_comparator = ImageComparator(original_directories=original_directories, generated_directories=generated_directories)

# chi_square_distance = image_comparator.compare_histograms(metric='chi-square')
# emd_distance = image_comparator.compare_histograms(metric='earth-mover')
# histogram_intersection_distance = image_comparator.compare_histograms(metric='histogram-intersection')
# msssim_values = image_comparator.calculate_msssim_torch()

# print(f"Chi-Square Distance: {chi_square_distance}")
# print(f"Earth Mover's Distance: {emd_distance}")
# print(f"Histogram Intersection Distance: {histogram_intersection_distance}")
# print(f"MSSIM: {np.average(msssim_values)}")

# image_comparator.plot_histograms('./outputs/lora/test_batch_21_11_23/graphs/')


import os
import random
import torch
import scipy
import torchvision
import numpy as np
from torch import Tensor
from scipy.linalg import sqrtm
from torchvision import transforms as trans
from torchvision.models import inception_v3
from torchvision.datasets import DatasetFolder
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
from torchmetrics.image.inception import InceptionScore
from pytorch_msssim import ms_ssim
import torch.nn.functional as F
from scipy.stats import chisquare, wasserstein_distance
from torchvision.io import read_image
from torchvision.transforms import ToPILImage, ToTensor, Resize

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()

        # Filter out directories and only include .png files
        self.img_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and f.endswith('.png')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = read_image(img_path)
        image = self.to_pil(image)  # Convert tensor to PIL Image
        image = image.convert("RGB")  # Convert image to RGB
        if self.transform:
            image = self.transform(image)
        image = self.to_tensor(image)  # Convert PIL Image back to tensor
        image = image * 255  # Scale the images to [0, 255]
        image = image.type(torch.uint8)  # Convert the images to torch.uint8
        return image

class ImageComparator:
    def __init__(self, original_directories, generated_directories):
        # self.original_images = self.load_images(original_directories)
        # self.generated_images = self.load_images(generated_directories)
        self.original_directories = original_directories
        self.generated_directories = generated_directories

        # self.original_histogram = self.compute_histograms(self.original_images)
        # self.generated_histogram = self.compute_histograms(self.generated_images)
        
    def loader(path: str) -> torch.Tensor:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return trans.ToTensor()(img.convert('RGB'))

        
    def calculate_fid_score(self, og_dir, gen_dir):

        # Define your transform
        transform = Resize((256, 256))
        og_dataset = CustomImageDataset(img_dir=og_dir[0], transform=transform)
        gen_dataset = CustomImageDataset(img_dir=gen_dir[0], transform=transform)


        dataloader1 = DataLoader(og_dataset, batch_size=32, shuffle=True)
        dataloader2 = DataLoader(gen_dataset, batch_size=32, shuffle=True)

        fid = FrechetInceptionDistance(feature=2048, reset_real_features=True, normalize=False)

        for original_batch, generated_batch in zip(dataloader1, dataloader2):
            fid.update(original_batch, True)
            fid.update(generated_batch, False)
        fid_score = fid.compute()
        return fid_score
    
    def calculate_inception_score(self, og_dir, gen_dir):
        # Define your transform
        transform = Resize((256, 256))
        og_dataset = CustomImageDataset(img_dir=og_dir[0], transform=transform)
        gen_dataset = CustomImageDataset(img_dir=gen_dir[0], transform=transform)

        dataloader1 = DataLoader(og_dataset, batch_size=32, shuffle=True)
        dataloader2 = DataLoader(gen_dataset, batch_size=32, shuffle=True)

        inception_score = InceptionScore(feature=2048)

        for original_batch, generated_batch in zip(dataloader1, dataloader2):
            inception_score.update(original_batch)
            inception_score.update(generated_batch)
        mean_score, std_score = inception_score.compute()
        return mean_score, std_score
    
    def calculate_ms_ssim(self, og_dir, gen_dir):
        # Define your transform
        transform = Resize((512, 512))
        og_dataset = CustomImageDataset(img_dir=og_dir[0], transform=transform)
        gen_dataset = CustomImageDataset(img_dir=gen_dir[0], transform=transform)

        dataloader1 = DataLoader(og_dataset, batch_size=32, shuffle=True)
        dataloader2 = DataLoader(gen_dataset, batch_size=32, shuffle=True)

        ms_ssim_scores = []

        for original_batch, generated_batch in zip(dataloader1, dataloader2):
            original_batch = original_batch.unsqueeze(0)  # Add channel dimension
            generated_batch = generated_batch.unsqueeze(0)  # Add channel dimension
            ms_ssim_score = multiscale_structural_similarity_index_measure(original_batch, generated_batch, data_range=255)
            ms_ssim_scores.append(ms_ssim_score.item())

        average_ms_ssim_score = sum(ms_ssim_scores) / len(ms_ssim_scores)
        return average_ms_ssim_score


    # def load_images(self, directories):
    #     images = []
    #     num_images = None

    #     # Find the minimum number of images across all directories
    #     for directory in directories:
    #         images_in_directory = [filename for filename in os.listdir(directory) if
    #                                filename.endswith(('.jpg', '.jpeg', '.png'))]
    #         num_images_in_directory = len(images_in_directory)

    #         if num_images is None or num_images_in_directory < num_images:
    #             num_images = num_images_in_directory

    #     # Load images from each directory, limiting larger directories to the number of images in the smallest directory
    #     for directory in directories:
    #         images_in_directory = [filename for filename in os.listdir(directory) if
    #                                filename.endswith(('.jpg', '.jpeg', '.png'))]

    #         # Randomly select num_images from the current directory
    #         selected_images = np.random.choice(images_in_directory, size=num_images, replace=False)

    #         for filename in selected_images:
    #             image_path = os.path.join(directory, filename)
    #             img = Image.open(image_path).convert('RGB').resize((256, 256))
    #             img = np.array(img)

    #             # Ensure all images have the same valid shape and color space
    #             if img.shape[0] > 0 and img.shape[1] > 0 and img.shape[2] == 3:
    #                 images.append(img)

    #     return np.array(images)
    
    # def sample_image_pairs(self, num_pairs):
    #     if num_pairs is None:
    #         return self.original_images, self.generated_images

    #     sampled_original_images = random.sample(list(self.original_images), num_pairs)
    #     sampled_generated_images = random.sample(list(self.generated_images), num_pairs)

    #     return np.array(sampled_original_images), np.array(sampled_generated_images)

    def compute_histograms(self, images):
        # Assuming images is a 4D numpy array (num_images, height, width, channels)
        histograms = []
        for image in images:
            histograms.append([np.histogram(channel.flatten(), bins=256, range=(0, 256))[0] for channel in image])

        return histograms

    def compare_histograms(self, metric='chi-square'):
        if self.original_images.shape[-1] != self.generated_images.shape[-1]:
            raise ValueError("Original and generated images must have the same number of channels.")
        if self.original_images.shape[1:-1] != self.generated_images.shape[1:-1]:
            raise ValueError("Original and generated images must have the same dimensions.")

        if metric == 'chi-square':
            # Compute histograms for each channel
            original_histograms = [np.histogram(channel.flatten(), bins=256, range=(0, 256))[0] for channel in np.moveaxis(self.original_images, -1, 0)]
            generated_histograms = [np.histogram(channel.flatten(), bins=256, range=(0, 256))[0] for channel in np.moveaxis(self.generated_images, -1, 0)]

            # Concatenate histograms for each channel
            original_histogram_flat = np.concatenate(original_histograms, axis=None)
            generated_histogram_flat = np.concatenate(generated_histograms, axis=None)

            # Ensure histograms have the same shape
            min_len = min(len(original_histogram_flat), len(generated_histogram_flat))
            original_histogram_flat = original_histogram_flat / np.sum(original_histogram_flat)
            generated_histogram_flat = generated_histogram_flat / np.sum(generated_histogram_flat)
            distance, _ = chisquare(original_histogram_flat, f_exp=generated_histogram_flat)
        elif metric == 'earth-mover':
            # Earth Mover's Distance (EMD)
            original_histogram_flat = np.concatenate([channel.flatten() for channel in np.moveaxis(self.original_images, -1, 0)])
            generated_histogram_flat = np.concatenate([channel.flatten() for channel in np.moveaxis(self.generated_images, -1, 0)])
            distance = wasserstein_distance(original_histogram_flat, generated_histogram_flat)
        elif metric == 'histogram-intersection':
            # Histogram Intersection
            original_histogram_flat = np.concatenate(self.original_histogram).flatten()
            generated_histogram_flat = np.concatenate(self.generated_histogram).flatten()
            min_len = min(len(original_histogram_flat), len(generated_histogram_flat))
            original_histogram_flat = original_histogram_flat[:min_len]
            generated_histogram_flat = generated_histogram_flat[:min_len]
            minima = np.minimum(original_histogram_flat, generated_histogram_flat)
            intersection = np.true_divide(np.sum(minima), np.sum(generated_histogram_flat))
            distance = 1 - intersection
        else:
            raise ValueError("Invalid metric. Supported metrics: 'chi-square', 'earth-mover', 'histogram-intersection'.")

        return distance
    
    def calculate_msssim_torch(self):
        msssim_values = []

        # Create data loaders for the original and generated images
        original_loader = DataLoader(self.original_images, batch_size=32)
        generated_loader = DataLoader(self.generated_images, batch_size=32)

        # Iterate over pairs of original and generated images
        for (original_imgs, generated_imgs) in tqdm(zip(original_loader, generated_loader), desc="Calculating MS-SSIM"):
            # The images are already PyTorch tensors, so no need to convert
            original_imgs_torch = original_imgs.float()
            generated_imgs_torch = generated_imgs.float()

            # Resize the images if necessary
            if original_imgs_torch.shape[-1] < 161 or original_imgs_torch.shape[-2] < 161:
                original_imgs_torch = F.interpolate(original_imgs_torch, size=(161, 161))
                generated_imgs_torch = F.interpolate(generated_imgs_torch, size=(161, 161))

            # Calculate the MS-SSIM
            msssim = ms_ssim(original_imgs_torch, generated_imgs_torch, data_range=1.0)
            msssim_values.append(msssim.item())

        return np.array(msssim_values)

    """
    def calculate_msssim_torch(self):
        msssim_values = []

        # Create data loaders for the original and generated images
        original_loader = DataLoader(self.original_images, batch_size=1)
        generated_loader = DataLoader(self.generated_images, batch_size=1)

        # Iterate over all combinations of original and generated images
        for original_imgs in tqdm(original_loader, desc="Calculating MS-SSIM"):
            for generated_imgs in generated_loader:
                # The images are already PyTorch tensors, so no need to convert
                original_imgs_torch = original_imgs.float()
                generated_imgs_torch = generated_imgs.float()

                # Resize the images if necessary
                if original_imgs_torch.shape[-1] < 161 or original_imgs_torch.shape[-2] < 161:
                    original_imgs_torch = F.interpolate(original_imgs_torch, size=(161, 161))
                    generated_imgs_torch = F.interpolate(generated_imgs_torch, size=(161, 161))

                # Calculate the MS-SSIM
                msssim = ms_ssim(original_imgs_torch, generated_imgs_torch, data_range=255.0)
                msssim_values.append(msssim.item())

        return np.array(msssim_values)
        """
    def torch_cov(self, m, rowvar=False):
        '''Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            m: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the relationship
                is transposed: each column represents a variable, while the rows
                contain observations.

        Returns:
            The covariance matrix of the variables.
        '''
        if m.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if m.dim() < 2:
            m = m.view(1, -1)
        if not rowvar and m.size(0) != 1:
            m = m.t()
        # m = m.type(torch.double)  # uncomment this line if necessary
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)
        mt = m.t()  # if complex: mt = m.t().conj()
        return fact * m.matmul(mt).squeeze()


    def calculate_activations(self, model, images):
        # Convert images to PyTorch tensors and normalize
        images = torch.tensor(images).permute(0, 3, 1, 2).float()
        images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images = normalize(images)

        # Pass images through model and return activations
        with torch.no_grad():
            activations = model(images)[0]
        print(f"Activations shape: {activations.shape}")
        return activations.unsqueeze(0)

    def calculate_fid(self):
        # Load InceptionV3 model
        model = inception_v3(pretrained=True)
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.eval()

        # Calculate activations for original and generated images
        act1 = self.calculate_activations(model, self.original_images)
        act2 = self.calculate_activations(model, self.generated_images)

        # Calculate mean and covariance
        mu1, sigma1 = act1.mean(dim=0), self.torch_cov(act1)
        mu2, sigma2 = act2.mean(dim=0), self.torch_cov(act2)

        print(f"Sigma1 shape: {sigma1.shape}")
        print(f"Sigma2 shape: {sigma2.shape}")

        # Calculate Frechet distance
        fid = self.frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid
    
    def frechet_distance(self, mu_x: Tensor, sigma_x: Tensor, mu_y: Tensor, sigma_y: Tensor) -> Tensor:
        a = (mu_x - mu_y).square().sum(dim=-1)
        b = sigma_x.trace() + sigma_y.trace()
        c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum(dim=-1)
        return a + b - 2 * c

    def calculate_fid_torch(self):
        fid_values = []

        # Create data loaders for the original and generated images
        original_loader = DataLoader(self.original_images, batch_size=1)
        generated_loader = DataLoader(self.generated_images, batch_size=1)

        # Initialize the Inception model
        inception_model = torchvision.models.inception_v3(pretrained=True).eval()

        # Iterate over all combinations of original and generated images
        for original_imgs in tqdm(original_loader, desc="Calculating FID"):
            for generated_imgs in generated_loader:
                # The images are already PyTorch tensors, so no need to convert
                original_imgs_torch = original_imgs.float()
                generated_imgs_torch = generated_imgs.float()

                # Resize the images if necessary
                if original_imgs_torch.shape[-1] < 299 or original_imgs_torch.shape[-2] < 299:
                    original_imgs_torch = F.interpolate(original_imgs_torch, size=(299, 299))
                    generated_imgs_torch = F.interpolate(generated_imgs_torch, size=(299, 299))

                # Calculate the features from the Inception model
                original_features = inception_model(original_imgs_torch)
                generated_features = inception_model(generated_imgs_torch)

                # Calculate the mean and covariance of the features
                original_mean = torch.mean(original_features, dim=0)
                original_cov = self.torch_cov(original_features)

                generated_mean = torch.mean(generated_features, dim=0)
                generated_cov = self.torch_cov(generated_features)

                # Calculate the FID
                mean_diff = torch.sum((original_mean - generated_mean) ** 2.0)
                cov_sqrt = self.sqrtm_torch(original_cov.mm(generated_cov))

                if not torch.is_complex(cov_sqrt):
                    trace_cov_sqrt = torch.trace(cov_sqrt)
                else:
                    trace_cov_sqrt = torch.trace(cov_sqrt.real)

                fid = mean_diff + torch.trace(original_cov) + torch.trace(generated_cov) - 2 * trace_cov_sqrt
                fid_values.append(fid.item())
        return np.array(fid_values)

    def plot_histograms(self, save_directory):
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        for i, (original_histogram, generated_histogram) in enumerate(zip(self.original_histogram, self.generated_histogram)):
            fig, ax = plt.subplots(3, 1, figsize=(10, 6))

            for j in range(3):  # For each color channel
                ax[j].bar(range(256), original_histogram[j], alpha=0.7, label='Original')
                ax[j].bar(range(256), generated_histogram[j], alpha=0.7, label='Generated')
                ax[j].set_xlim([0, 256])
                ax[j].legend(loc='upper right')

            plt.tight_layout()
            plt.savefig(os.path.join(save_directory, f'histogram_{i}.png'))
            plt.close(fig)

# Example usage:
original_directories = ["./data/full-finetune/all_kunst_v2/"]
generated_directories = ['./outputs/lora/test_batch_21_11_23/']

image_comparator = ImageComparator(original_directories=original_directories, generated_directories=generated_directories)

# chi_square_distance = image_comparator.compare_histograms(metric='chi-square')
# print(f"Chi-Square Distance: {chi_square_distance}")
# emd_distance = image_comparator.compare_histograms(metric='earth-mover')
# print(f"Earth Mover's Distance: {emd_distance}")
# histogram_intersection_distance = image_comparator.compare_histograms(metric='histogram-intersection')
# print(f"Histogram Intersection Distance: {histogram_intersection_distance}")
fid = image_comparator.calculate_fid_score(original_directories, generated_directories)
print(f"FID: {fid}")
insep_score_mean, insep_score_std = image_comparator.calculate_inception_score(original_directories, generated_directories)
print(f"Mean Inception Score: {insep_score_mean} +/- {insep_score_std}")
msssim = image_comparator.calculate_ms_ssim(original_directories, generated_directories)
print(f"MSSSIM: {msssim}")

# msssim_values = image_comparator.calculate_msssim_torch()
# print(f"MSSIM: {np.average(msssim_values)}")

# image_comparator.plot_histograms('./outputs/lora/test_batch_21_11_23/graphs/')
