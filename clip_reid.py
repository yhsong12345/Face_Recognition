import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import CLIPModel, CLIPProcessor

# ==============================================================
# 1. Dataset (simple folder-based version)
# ==============================================================


class ReIDDataset(Dataset):
    """
    Expected folder structure:
      root/
        0001/
            img1.jpg
            img2.jpg
        0002/
            img3.jpg
            img4.jpg
    """

    def __init__(self, root_dir, processor):
        self.samples = []
        self.processor = processor
        pid_dirs = sorted(os.listdir(root_dir))
        for i, pid in enumerate(pid_dirs):
            pid_path = os.path.join(root_dir, pid)
            if not os.path.isdir(pid_path):
                continue
            for img_name in os.listdir(pid_path):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(pid_path, img_name), int(i)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pid = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
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
    def __init__(self, in_dim=512, num_classes=751):
        super().__init__()
        self.bnneck = nn.BatchNorm1d(in_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(in_dim, num_classes, bias=False)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_out")

    def forward(self, feat):
        bn_feat = self.bnneck(feat)
        cls_score = self.classifier(bn_feat)
        return bn_feat, cls_score


# ==============================================================
# 4. CLIP + ReID Head model
# ==============================================================


class CLIPReIDModel(nn.Module):
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
            dist_ap.append(dist_mat[i][mask_pos[i]].max().unsqueeze(0))
            dist_an.append(dist_mat[i][mask_neg[i]].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        loss = F.relu(dist_ap - dist_an + self.margin)
        return loss.mean()


class ReIDLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.triplet = BatchHardTripletLoss(margin=margin)

    def forward(self, cls_score, bn_feat, labels):
        loss_id = self.ce(cls_score, labels)
        loss_triplet = self.triplet(bn_feat, labels)
        return loss_id + loss_triplet


# ==============================================================
# 6. Training Loop
# ==============================================================


def train_clip_reid(root_dir, num_classes=751, epochs=10, lr=3e-4, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    dataset = ReIDDataset(root_dir, processor)
    sampler = RandomIdentitySampler(dataset, num_instances=4)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=4
    )

    model = CLIPReIDModel(num_classes=num_classes, freeze_clip=True).to(device)
    criterion = ReIDLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=5e-4
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, bn_feat, cls_score = model(imgs)
            loss = criterion(cls_score, bn_feat, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "clip_reid.pth")
    print("✅ Training finished and model saved.")


# ==============================================================
# 7. Inference / Feature Extraction
# ==============================================================


@torch.no_grad()
def extract_features(model, dataloader):
    model.eval()
    all_feats, all_labels = [], []
    for imgs, labels in dataloader:
        imgs = imgs.cuda()
        _, bn_feat, _ = model(imgs)
        all_feats.append(F.normalize(bn_feat, dim=1))
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)
