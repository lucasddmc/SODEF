"""
Data loading utilities for CIFAR-10
Handles dataset preparation, normalization, and data loaders
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


def get_cifar10_transforms(train=True, normalize=True):
    """Get CIFAR-10 transforms"""
    transform_list = []
    
    if train:
        # Training augmentations
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalization
    if normalize:
        transform_list.append(
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        )
    
    return transforms.Compose(transform_list)


def get_cifar10_loaders(data_dir='./data', batch_size=128, num_workers=4, 
                       download=True, normalize=True):
    """Get CIFAR-10 data loaders"""
    
    # Training transforms with augmentation
    train_transform = get_cifar10_transforms(train=True, normalize=normalize)
    
    # Test transforms without augmentation
    test_transform = get_cifar10_transforms(train=False, normalize=normalize)
    
    # Datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=test_transform
    )
    
    # For evaluation (same as test but might need different sampling)
    eval_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=test_transform
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader, eval_loader


def normalize_tensor(x, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Normalize tensor with CIFAR-10 statistics"""
    mean = torch.tensor(mean).view(3, 1, 1).to(x.device)
    std = torch.tensor(std).view(3, 1, 1).to(x.device)
    return (x - mean) / std


def denormalize_tensor(x, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Denormalize tensor"""
    mean = torch.tensor(mean).view(3, 1, 1).to(x.device)
    std = torch.tensor(std).view(3, 1, 1).to(x.device)
    return x * std + mean


def clamp_tensor(x, lower_limit=0, upper_limit=1):
    """Clamp tensor values"""
    return torch.clamp(x, lower_limit, upper_limit)


class CIFAR10DataModule:
    """Data module for easy management of CIFAR-10 data loaders"""
    
    def __init__(self, data_dir='./data', batch_size=128, num_workers=4,
                 download=True, normalize=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.normalize = normalize
        
        self._train_loader = None
        self._test_loader = None
        self._eval_loader = None
        
    def setup(self):
        """Setup data loaders"""
        loaders = get_cifar10_loaders(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            download=self.download,
            normalize=self.normalize
        )
        self._train_loader, self._test_loader, self._eval_loader = loaders
        
    @property
    def train_loader(self):
        if self._train_loader is None:
            self.setup()
        return self._train_loader
    
    @property
    def test_loader(self):
        if self._test_loader is None:
            self.setup()
        return self._test_loader
    
    @property
    def eval_loader(self):
        if self._eval_loader is None:
            self.setup()
        return self._eval_loader
    
    def get_sample_batch(self, loader_type='test'):
        """Get a sample batch for testing"""
        if loader_type == 'train':
            loader = self.train_loader
        elif loader_type == 'test':
            loader = self.test_loader
        else:
            loader = self.eval_loader
            
        return next(iter(loader))