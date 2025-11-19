import argparse
import logging
import os

import lightning as L
import torch
import wandb
from data.datasets import create_data_loaders, create_ms1mv2_datasets
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from model import Face_Recognition
from wandb.integration.lightning.fabric import WandbLogger


def main(args):
    lr = args["learning_rate"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    imgsz = args["image_size"]
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

    train_dataset, valid_dataset = create_ms1mv2_datasets(imgsz, f)

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Traning Dataset: {len(train_dataset)} found")
    logging.info(f"Validation Dataset: {len(valid_dataset)} found")

    train_loader, valid_loader = create_data_loaders(
        train_dataset, valid_dataset, batch_size
    )

    wandb.login(key="2892bd003dbade283d8f143bd55ceebe4eaf5690")

    wandb_logger = WandbLogger(project=project, name=name, log_model=False)

    model = Face_Recognition(
        model_name=model, lr=lr, pretrained=False, num_classes=70, loss=name
    )
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=epochs,
        profiler="simple",
        devices=4,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        strategy="ddp",
        use_distributed_sampler=True,
        callbacks=[
            ModelCheckpoint(
                dirpath=save_dir,
                save_top_k=3,
                monitor="Validation loss",
                mode="min",
                filename="{epoch}_{step}_{Validation loss:.2f}",
            ),
            ModelSummary(max_depth=3),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=wandb_logger,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


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
        "--learning_rate", type=float, default=0.1, help="learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--model", type=str, default="resnet50", help="Model Selection")
    parser.add_argument(
        "-sav_dir",
        type=str,
        dest="save_dir",
        help="directory",
        default="outputs",
    )
    parser.add_argument("--name", type=str, default="elasticarcface")
    parser.add_argument("--project", type=str, default="face recognition")
    parser.add_argument(
        "--data",
        type=str,
        default="/purestorage/AILAB/AI_4/datasets/cctv/image/stage1_data/2025-10-15",
    )
    args = vars(parser.parse_args())
    main(args)
