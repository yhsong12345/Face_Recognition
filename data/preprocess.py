import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class WiderFaceDetection(data.Dataset):
    def __init__(self, folders_path, preproc=None, imgsz=640):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        self.imgsz = (imgsz, imgsz)

        self.num_classes = 0

        self.img_transform = transforms.Compose(
            [
                transforms.Resize(self.imgsz),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ]
        )

        folders = os.listdir(folders_path)
        for i, folder in enumerate(folders):
            if isinstance(folder, str):
                identity = int(i)
            imgs_path = os.path.join(folders_path, folder)
            imgs = os.listdir(imgs_path)
            for img in imgs:
                img_path = os.path.join(imgs_path, img)
                self.imgs_path.append(img_path)
                self.words.append(identity)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        labels = self.words[index]
        # annotations = np.zeros((1))
        # annotations[0] = labels
        annotations = torch.tensor(labels, dtype=torch.long)
        # annotations = np.zeros(1)
        # if len(labels) == 0:
        #     return annotations
        # for label in labels:
        #     annotation = np.zeros(1)
        #     annotation[0] = label[0]  # identity
        #
        #     annotations = np.append(annotations, annotation, axis=0)
        # target = np.array(annotations)
        #
        # if self.preproc is not None:
        #     img, target = self.preproc(img, target)

        return self.img_transform(img), annotations


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).type(torch.LongTensor)
                targets.append(annos)

    return (torch.stack(imgs, 0), torch.stack(targets))
