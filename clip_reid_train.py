import argparse
import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import CLIPFeatureExtractor, CLIPModel
from wandb.integration.lightning.fabric import WandbLogger


class ReIDDataset(Dataset):
    """
    Person ReID Dataset.
    Accepts a list of (img_path, pid) samples.
    Applies the provided `preprocess` (open_clip transforms) to images.
    """

    def __init__(self, samples, preprocess):
        """
        Args:
            samples: list of (img_path, pid) tuples
            preprocess: open_clip preprocessing transform
        """
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pid = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        inputs = self.preprocess(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (3, H, W)
        return pixel_values, torch.tensor(pid)


# ==============================================================
# 2. Identity Sampler (P×K sampling)
# ==============================================================


class RandomIdentitySampler(Sampler):
    """Samples P identities, each with K instances per batch."""

    def __init__(self, dataset, num_instances=4):
        self.dataset = dataset
        self.num_instances = num_instances
        self.index_dic = {}
        for index, (_, pid) in enumerate(dataset.samples):
            self.index_dic.setdefault(pid, []).append(index)
        self.pids = list(self.index_dic.keys())

    def __iter__(self):
        batch_idxs = []
        while len(batch_idxs) < len(self.dataset):
            selected_pids = random.sample(self.pids, min(len(self.pids), 16))  # P=16
            for pid in selected_pids:
                idxs = self.index_dic[pid]
                if len(idxs) < self.num_instances:
                    idxs = random.choices(idxs, k=self.num_instances)
                else:
                    idxs = random.sample(idxs, self.num_instances)
                batch_idxs.extend(idxs)
        return iter(batch_idxs)

    def __len__(self):
        return len(self.dataset)


# ==============================================================
# 3. ReID Head (BNNeck + Classifier)
# ==============================================================


class ReIDHead(nn.Module):
    def __init__(self, in_dim, num_classes, feat_dim=512):
        super().__init__()
        self.feat_fc = nn.Linear(in_dim, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)
        self.cls = nn.Linear(feat_dim, num_classes, bias=False)

    def forward(self, x):
        feat = self.feat_fc(x)
        bn_feat = self.bn(feat)
        cls_score = self.cls(bn_feat)
        return bn_feat, cls_score


# ==============================================================
# 4. CLIP + ReID Head model
# ==============================================================


class CLIPReIDModel(L.LightningModule):
    def __init__(self, num_classes=751, freeze_clip=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.image_encoder = self.clip.vision_model
        self.proj = self.clip.visual_projection
        self.reid_head = ReIDHead(in_dim=512, num_classes=num_classes)

        if freeze_clip:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)
            for p in self.proj.parameters():
                p.requires_grad_(False)

    def forward(self, images):
        outputs = self.image_encoder(pixel_values=images)
        pooled = outputs.pooler_output
        feat = self.proj(pooled)
        feat = F.normalize(feat, dim=-1)
        bn_feat, cls_score = self.reid_head(feat)
        return feat, bn_feat, cls_score

    def _setup_transform(self):
        self.preprocess = CLIPFeatureExtractor.from_pretrained(self.model_name)

    def extract_embedding(
        self, image: Union[Path, str, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        단일 이미지에서 임베딩 추출

        Args:
            image: 이미지 (파일 경로, PIL Image, 또는 numpy 배열)

        Returns:
            임베딩 벡터 (numpy 배열)
        """
        # 이미지 로드 및 전처리
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            image = np.asarray(image)[:, :, ::-1]  # RGB -> BGR
            image = self.preprocess(image, return_tensors="pt")
            images = image["pixel_values"].squeeze(0)

        elif isinstance(image, np.ndarray):
            image = self.preprocess(image, return_tensors="pt")
            images = image["pixel_values"].squeeze(0)

        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                images = image.unsqueeze(0)
            images = images.to(self.device)

        # 임베딩 추출
        with torch.no_grad():
            embeddings, _, _ = self.forward(images)

        return embeddings

    def extract_embeddings_batch(
        self, images: List[Union[Path, str, Image.Image, np.ndarray]]
    ) -> np.ndarray:
        """
        배치 이미지에서 임베딩 추출

        Args:
            images: 이미지 리스트

        Returns:
            임베딩 행렬 (numpy 배열)
        """
        # imgs = []
        # for image in images:
        #     if isinstance(image, str) or isinstance(image, Path):
        #         image = Image.open(image).convert('RGB')
        #         image = np.asarray(image)[:, :, ::-1]
        #     elif isinstance(image, np.ndarray):
        #         image = self.to_pil(image)
        #     else:
        #         raise TypeError('Elements must be str or numpy.ndarray')
        #     image = self.preprocess(image)
        #     image = image.unsqueeze(0)
        #     imgs.append(image)
        # imgs = torch.stack(imgs, dim=0).to(self.device)
        images = [
            self.preprocess(Image.open(p).convert("RGB"), return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)
            for p in images
        ]
        images = torch.stack(images).cuda()  # shape: (B, 3, 256, 128)

        with torch.no_grad():
            embeddings, _, _ = self.forward(images)

        return embeddings


# class CLIPReIDModel(L.LightningModule):
#     def __init__(
#         self,
#         num_classes=751,
#         model_name="ViT-B-16",
#         pretrained="openai",
#         feat_dim=512,
#         freeze_clip=True,
#     ):
#         super().__init__()

#         self.clip_model, _, _ = open_clip.create_model_and_transforms(
#             model_name, pretrained=pretrained
#         )

#         # freeze encoder
#         if freeze_clip:
#             for p in self.clip_model.parameters():
#                 p.requires_grad = False

#         in_dim = self.clip_model.visual.output_dim
#         self.reid_head = ReIDHead(in_dim, num_classes, feat_dim)

#     def forward(self, x):
#         feats = self.clip_model.encode_image(x)  # forward pass ok
#         raw_feat, bn_feat, cls_scores = self.reid_head(feats)
#         return raw_feat, bn_feat, cls_scores


def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1 + torch.cos(torch.pi * progress))
        return max(min_lr, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class L_CLIPReIDModel(L.LightningModule):
    def __init__(self, lr, num_classes=10, freeze_clip=True):
        super().__init__()
        self.lr = lr

        self.model = CLIPReIDModel(num_classes=num_classes, freeze_clip=freeze_clip)

        self.loss = ReIDLoss()

    def forward(self, x, y=None):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        img, labels = batch
        _, bn_feat, cls_score = self.model(img)
        loss = self.loss(cls_score, bn_feat, labels)
        values = {"Training loss": loss}
        self.log_dict(values, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        _, bn_feat, cls_score = self.model(img)
        loss = self.loss(cls_score, bn_feat, labels)
        values = {"Validation loss": loss}
        self.log_dict(values, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=5e-4,
        )

        lr_scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        # lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        # lr_scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=3e-4,
        #     steps_per_epoch=len(dataloader),
        #     epochs=50,
        #     pct_start=0.1,
        #     anneal_strategy="cos",
        # )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


# ==============================================================
# 5. Hard Triplet Loss + Combined Loss
# ==============================================================


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        N = dist_mat.size(0)
        mask_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        mask_neg = ~mask_pos

        dist_ap, dist_an = [], []

        for i in range(N):
            pos_mask = mask_pos[i].clone()
            pos_mask[i] = False  # ignore self
            if pos_mask.sum() == 0:
                continue
            neg_mask = mask_neg[i]
            if neg_mask.sum() == 0:
                continue
            dist_ap.append(dist_mat[i][pos_mask].max().unsqueeze(0))
            dist_an.append(dist_mat[i][neg_mask].min().unsqueeze(0))

        if len(dist_ap) == 0 or len(dist_an) == 0:
            # fallback to 0
            return torch.tensor(0.0, device=embeddings.device)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        loss = F.relu(dist_ap - dist_an + self.margin)
        return loss.mean()


class ReIDLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.triplet = BatchHardTripletLoss(margin=margin)
        # self.triplet = nn.TripletMarginLoss(margin=margin)

    def forward(self, cls_score, bn_feat, labels):
        loss_id = self.ce(cls_score, labels)
        ## for batchhardtripletloss
        loss_triplet = self.triplet(bn_feat, labels)
        ## for tiplet loss
        # loss_triplet = self.triplet(bn_feat, bn_feat, bn_feat)
        return loss_id + loss_triplet


def split_reid_dataset(root_dir, train_ratio=0.9, seed=42):
    """
    Splits a ReID dataset (folder per person) into train and val sets.
    Returns:
        train_samples: list of (img_path, pid)
        val_samples: list of (img_path, pid)
    """
    random.seed(seed)
    train_samples, val_samples = [], []

    pid_dirs = sorted(os.listdir(root_dir))
    for i, pid in enumerate(pid_dirs):
        pid_path = os.path.join(root_dir, pid)
        if not os.path.isdir(pid_path):
            continue

        imgs = [
            os.path.join(pid_path, img)
            for img in os.listdir(pid_path)
            if img.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        random.shuffle(imgs)

        n_train = max(1, int(len(imgs) * train_ratio))  # at least 1 img in train
        train_samples.extend([(img, int(i)) for img in imgs[:n_train]])
        val_samples.extend([(img, int(i)) for img in imgs[n_train:]])

    print(
        f"Total IDs: {len(pid_dirs)}, Train samples: {len(train_samples)}, Val samples: {len(val_samples)}"
    )
    return train_samples, val_samples


def main(args):
    lr = args["learning_rate"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    imgsz = (384, 128)
    model = args["model"]
    d = args["save_dir"]
    f = args["data"]
    name = args["name"]
    project = args["project"]

    save_dir = os.getcwd() + "/" + d + f"/{model}_{name}"

    # Check the save_dir exists or not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # else:
    #     while os.path.exists(save_dir):
    #         for i in range(100):
    #             save_dir = save_dir + f"_{i}"
    #             os.makedirs(save_dir)
    train_samples, val_samples = split_reid_dataset(f, train_ratio=0.9)
    processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

    train_dataset = ReIDDataset(train_samples, processor)
    val_dataset = ReIDDataset(val_samples, processor)
    # sampler = RandomIdentitySampler(train_dataset, num_instances=4)

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Traning Dataset: {len(train_dataset)} found")
    logging.info(f"Validation Dataset: {len(val_dataset)} found")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    wandb.login(key="2892bd003dbade283d8f143bd55ceebe4eaf5690")

    wandb_logger = WandbLogger(project=project, name=f"{model}_{name}", log_model=False)

    model = L_CLIPReIDModel(lr, num_classes=70, freeze_clip=True)
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=epochs,
        profiler="simple",
        devices=4,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        # strategy="ddp",
        # strategy="ddp_find_unused_parameters_true",
        strategy=DDPStrategy(find_unused_parameters=True),
        use_distributed_sampler=True,
        callbacks=[
            ModelCheckpoint(
                dirpath=save_dir,
                save_top_k=1,
                monitor="Validation loss",
                mode="min",
                filename="best",
            ),
            ModelSummary(max_depth=3),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=wandb_logger,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train our network for",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument(
        "--model", type=str, default="clip_reid", help="Model Selection"
    )
    parser.add_argument(
        "-sav_dir",
        type=str,
        dest="save_dir",
        help="directory",
        default="outputs",
    )
    parser.add_argument("--name", type=str, default="batchhardtriplet")
    parser.add_argument("--project", type=str, default="face recognition")
    parser.add_argument(
        "--data",
        type=str,
        default="/purestorage/AILAB/AI_4/datasets/cctv/image/stage1_data/2025-10-15",
    )
    args = vars(parser.parse_args())
    main(args)
