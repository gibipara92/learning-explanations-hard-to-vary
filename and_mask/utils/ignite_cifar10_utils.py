import os
import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from and_mask.models.fastresnet import FastResnet


def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_train_test_loaders(path, batch_size, num_workers,
                           random_seed,
                           random_labels_fraction,
                           pin_memory=True,
                           test_batch_size=1000):
    train_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if not os.path.exists(path):
        os.makedirs(path)
        download = True
    else:
        download = True if len(os.listdir(path)) < 1 else False

    train_ds = datasets.CIFAR10(root=path, train=True, download=download,
                                transform=train_transform)

    test_ds = datasets.CIFAR10(root=path, train=False, download=download,
                               transform=test_transform)

    assert random_labels_fraction >= 0.0

    label_rng = np.random.RandomState(seed=random_seed)
    print(f'RANDOM LABELS FRACTION: {random_labels_fraction}')
    n_random = int(round(random_labels_fraction * len(train_ds.targets)))
    rnd_idxs = label_rng.choice(np.arange(len(train_ds.targets)),
                                size=n_random,
                                replace=False)

    original_targets = train_ds.targets.copy()

    for rnd_idx in rnd_idxs:
        train_ds.targets[rnd_idx] = label_rng.randint(10)

    mislabeled_idxs, = np.where(np.array(original_targets) != np.array(train_ds.targets))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                       num_workers=num_workers,
                       pin_memory=pin_memory,
                       shuffle=True,
                       drop_last=True)
    mislabeled_train_loader = DataLoader(train_ds, batch_size=test_batch_size,
                       num_workers=num_workers,
                       pin_memory=pin_memory,
                       sampler=SubsetRandomSampler(indices=mislabeled_idxs),
                       drop_last=False)

    test_loader = DataLoader(test_ds, batch_size=test_batch_size,
                       num_workers=num_workers,
                       pin_memory=pin_memory)
    return train_loader, test_loader, mislabeled_train_loader


def get_model(num_classes):
    return FastResnet(num_classes=num_classes)
