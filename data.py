from os import listdir
from os.path import join
from typing import List, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (ColorJitter, Compose, GaussianBlur, 
                                    Normalize, RandomApply, 
                                    RandomGrayscale, RandomHorizontalFlip, 
                                    RandomResizedCrop, ToTensor)


def create_simsiam_transforms(size: int = 224, 
                              normalize: bool = True) -> Compose:
    """
    Returns a Compose object consisting of SimSiam's augmentations
    
    Args:
      size (int): Desired image size
      normalize (bool): Whether to normalize with ImageNet statistics
    """
    color_jitter = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, 
                               hue=0.1)

    kernel_size = int(0.1*size)
    kernel_size += (kernel_size-1)%2
    gaussian_blur = GaussianBlur(kernel_size, sigma=(0.1, 2.0))

    if normalize:
        normalize_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize_stats = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    transforms = Compose([RandomResizedCrop(size, scale=[0.2, 1.0]),
                          RandomHorizontalFlip(p=0.5),
                          RandomApply([color_jitter], p=0.8),
                          RandomGrayscale(p=0.2),
                          RandomApply([gaussian_blur], p=0.5),
                          ToTensor(),
                          Normalize(*normalize_stats)])
    return transforms


class SimSiamDataset(Dataset):
    """
    Dataset for SimSiam with its augmentaions

    Args:
        path (str): Path to images
        valid_exts (List[str]): List of valid image extensions
        size (int): Desired image size
        normalize (bool): Whether to normalize with ImageNet statistics
    """
    def __init__(self, path: str = 'images/', 
                 valid_exts: List[str] = ['jpeg', 'jpg'],
                 size: int = 224, normalize: bool = True):
        self.files = []

        for file in listdir(path):
            ext = file.split('.')[-1]
            if ext in valid_exts:
                file = join(path, file)
                self.files.append(file)

        self.transforms = create_simsiam_transforms(size=size, 
                                                    normalize=normalize)

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        file = self.files[i]
        img = Image.open(file)

        x1 = self.transforms(img)
        x2 = self.transforms(img)
        return x1, x2


def create_simsiam_dataloader(path: str = 'images/', 
                              valid_exts: List[str] = ['jpeg', 'jpg'],
                              size: int = 224, normalize: bool = True,
                              batch_size: int = 32, 
                              num_workers: int = 8) -> DataLoader:
    """
    Returns DataLoader from SimSiamDataset

    Args:
        path (str): Path to images
        valid_exts (List[str]): List of valid image extensions
        size (int): Desired size
        normalize (bool): Whether to normalize with ImageNet statistics
        batch_size (int): Batch size
        num_workers (int): Number of workers
    """
    dataset = SimSiamDataset(path=path, valid_exts=valid_exts, size=size, 
                             normalize=normalize)
                             
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    return dataloader
