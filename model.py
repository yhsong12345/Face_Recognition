import lightning as L
import torch
import torch.nn as nn
from models.common import ISEBottleneck, SEBasicBlock, SEBottleneck
from models.Resnet import LResNet
from torchvision import models
from utils.loss import ArcFace, ElasticArcFace, SubcenterArcFace  # ElasticCosFace


class Resnet(L.LightningModule):
    def __init__(self, model_name, pretrained=False, num_classes=10, loss=None):
        super().__init__()

        if model_name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            model = models.resnet101(pretrained=pretrained)
        elif model_name == "SE_LResnet50":
            model = LResNet(SEBottleneck, [3, 4, 6, 3], True, num_classes=num_classes)
        elif model_name == "SE_LResnet101":
            model = LResNet(SEBottleneck, [3, 4, 23, 3], True, num_classes=num_classes)
        elif model_name == "SE_LResnet50_IR":
            model = LResNet(
                ISEBottleneck, [3, 4, 6, 3], nn.PReLU(), num_classes=num_classes
            )
        elif model_name == "SE_LResnet101_IR":
            model = LResNet(
                ISEBottleneck, [3, 4, 23, 3], nn.PReLU(), num_classes=num_classes
            )

        if loss == "arcface":
            self.head = ArcFace(embed_size=512, num_classes=num_classes)
        elif loss == "subcenterarcface":
            self.head = SubcenterArcFace(embed_size=512, num_classes=num_classes)
        elif loss == "elasticarcface":
            self.head = ElasticArcFace(
                embed_size=512, num_classes=num_classes, s=64.0, m=0.5, std=0.0125
            )
        else:
            pass

        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(2048 * 7 * 7),
            nn.Dropout1d(0.4),
            nn.Linear(2048 * 7 * 7, 512),
            nn.BatchNorm1d(512),
        )

    def forward(self, x, y=None):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if y is not None:
            x = self.train_forward(x, y)
        return x

    def train_forward(self, x, y):
        return self.head(x, y)


class Face_Recognition(L.LightningModule):
    def __init__(
        self, model_name, lr, pretrained=False, num_classes=10, loss="arcface"
    ):
        super().__init__()
        self.lr = lr

        self.model = Resnet(
            model_name, pretrained=pretrained, num_classes=num_classes, loss=loss
        )

        self.loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x, y=None):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        img, targets = batch
        feature = self.model(img, targets)
        loss = self.loss(feature, targets)
        values = {"Training loss": loss}
        self.log_dict(values, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, targets = batch
        feature = self.model(img, targets)
        loss = self.loss(feature, targets)
        values = {"Validation loss": loss}
        self.log_dict(values, prog_bar=True)

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(
            params=[{"params": self.model.parameters()}],
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[40, 56, 64], gamma=0.1
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
