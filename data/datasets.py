import torch
from torch.utils.data import Subset, DataLoader
from data.preprocess import WiderFaceDetection, detection_collate
import os
import torchvision.transforms as transforms






def create_ms1mv2_datasets(s, path):
    dataset = WiderFaceDetection(folders_path=path, preproc=None, imgsz=s)
    num_train_dataset = int(len(dataset) * 0.9)

    indices = torch.randperm(len(dataset)).tolist()

    train_dataset = Subset(dataset, indices[:num_train_dataset])
    valid_dataset = Subset(dataset, indices[num_train_dataset:])

    return train_dataset, valid_dataset


def divide_dataset(evaluation):
    f = open(evaluation, 'r')
    lines = f.readlines()
    boundary = []
    prev = 0
    for i in range(len(lines)):
        line = lines[i]
        line = line.split(' ')
        line = [i for i in line if i.strip()]
        ty = line[-1]
        ty = ty.replace('\n', '')
        if prev != ty:
            boundary.append(i)
            prev = ty

    return boundary


def create_data_loaders(dataset_train, dataset_valid, BATCH_SIZE):
    """
    Function to build the data loaders.
    Parameters:
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    :param dataset_test: The test dataset.
    """
    nd = torch.cuda.device_count()

    # train_sampler = None if rank == -1 else distributed.DistributedSampler(dataset_train, shuffle=train_shuffle)
    # val_sampler = None if rank == -1 else distributed.DistributedSampler(dataset_valid, shuffle=val_shuffle)
    nw = os.cpu_count() // max(nd, 1)  # number of workers
    nw = min(8, nw)


    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=nw
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw
    )

    return train_loader, valid_loader
