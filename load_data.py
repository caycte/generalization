from torchvision.transforms.functional import pad
from torchvision import transforms
from torchvision import transforms, datasets

import numpy as np
import numbers

def add_padding(img):
    padding = transforms.Pad((512-32)//2, fill=0)  # You can adjust the fill value as needed

    return padding(img)
    

    # Load and Preprocess data.
# Loading: If the dataset is not in the given directory, it will be downloaded.
# Preprocessing: This includes normalizing each channel and data augmentation by random cropping and horizontal flipping
def load_data(split, dataset_name, datadir, nchannels):

    if dataset_name == 'MNIST':
        normalize = transforms.Normalize(mean=[0.131], std=[0.289])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([add_padding, transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([add_padding, transforms.ToTensor(), normalize])

    get_dataset = getattr(datasets, dataset_name)
    if dataset_name == 'SVHN':
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
    else:
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)

    return dataset