import torch
import torchvision
from torchvision import transforms,datasets
import random
from torch.utils.data import random_split,Subset

train_transform_cifar = transforms.Compose([
    transforms.Resize(32),
    transforms.ColorJitter(brightness=random.uniform(0,0.2),contrast=random.uniform(0,0.2),saturation=random.uniform(0,0.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0, 15)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

test_transform_cifar = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
     ])

train_transform_tiny = transforms.Compose([
    transforms.Resize(32),
    transforms.ColorJitter(brightness=random.uniform(0,0.2),contrast=random.uniform(0,0.2),saturation=random.uniform(0,0.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0, 15)),
    transforms.ToTensor(),
    transforms.Normalize((0.4594, 0.4308, 0.3452), (0.2570, 0.2453, 0.2512))
    ])

test_transform_tiny = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4594, 0.4308, 0.3452), (0.2570, 0.2453, 0.2512))
     ])



def create_datasets(experiment):
    """
    Function to build the training, validation, and testing dataset.
    """
    if experiment == 'a':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform_cifar)

        test_val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform_cifar)

        validation_set, test_set = random_split(test_val_set, [5000, 5000])
    elif experiment == 'b':
        train_dir='./tiny-image-subset/train'
        val_dir='./tiny-image-subset/val'
        train_data = datasets.ImageFolder(train_dir,       
                            transform=train_transform_tiny)
        test_set = datasets.ImageFolder(val_dir,
                            transform=test_transform_tiny)
        train_dataset_size = len(train_data)
        valid_size = int(0.1*train_dataset_size)
        indices = torch.randperm(len(train_data)).tolist()
        train_set = Subset(train_data, indices[:-valid_size])
        validation_set = Subset(train_data, indices[-valid_size:])

    print(f"Total training images: {len(train_set)}")
    print(f"Total validation images: {len(validation_set)}")
    print(f"Total test images: {len(test_set)}")
    return train_set, validation_set, test_set



def create_data_loaders(dataset_train, dataset_valid, dataset_test,BATCH_SIZE,NUM_WORKERS=1):
    """
    Function to build the data loaders.
    """
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    return train_loader, valid_loader, test_loader



