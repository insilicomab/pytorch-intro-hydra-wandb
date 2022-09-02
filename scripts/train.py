import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim

import timm

import os
import hydra
from omegaconf import DictConfig

import wandb

from src import (
    set_device, seed_torch, DataTransform, CifarDataset,
    make_dataloader, EarlyStopping, train_model_wb,
)


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):

    # gpu or cpu
    device = set_device()
    
    # set up experiment
    seed_torch()

    # directory to save models
    SAVE_MODEL_PATH = cfg.save_model_path
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    # initialize wandb
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        config={
            'data': os.path.basename(cfg.wandb.data_dir),
            'model': cfg.model_name,
        }
    )


    # read label data
    train_master = pd.read_csv(cfg.train_master, sep='\t')
    train_num_classes = train_master['label_id'].nunique()
    assert (
        cfg.model.num_classes == train_num_classes
    ), f'num_classes should be {train_num_classes}'

    # image name list
    image_name_list = train_master['file_name'].values

    # label list
    label_list = train_master['label_id'].values

    # split train & val
    x_train, x_val, y_train, y_val = train_test_split(
        image_name_list,
        label_list,
        test_size=cfg.split.test_size,
        stratify=label_list,
        random_state=cfg.split.random_state
    )

    # dataset
    train_dataset = CifarDataset(x_train, y_train, cfg.dataset.tr_img_path, transform=DataTransform(), phase='train')
    val_dataset = CifarDataset(x_val, y_val, cfg.dataset.tr_img_path, transform=DataTransform(), phase='val')

    # dataloader
    dataloader = make_dataloader(
        tr_dataset=train_dataset,
        val_dataset=val_dataset, 
        tr_batch_size=cfg.dataloader.tr_batch_size,
        val_batch_size=cfg.dataloader.val_batch_size
    )

    # define model
    model = timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes
    )
    model.to(device)

    # loss function, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

    # earlystopping and reduce lr schedular
    ers = EarlyStopping(patience=cfg.earlystopping.patience, verbose=cfg.earlystopping.verbose)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.schedular.mode,
        factor=cfg.schedular.factor,
        patience=cfg.schedular.patience,
        min_lr=cfg.schedular.min_lr,
        verbose=cfg.schedular.verbose
    )

    wandb.watch(model, log="all")

    # training
    train_model_wb(
        model=model,
        epochs=cfg.train.epochs,
        dataloader=dataloader,
        device=device,
        loss_fn=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        earlystopping=ers,
        save_model_path=SAVE_MODEL_PATH,
        model_name=cfg.train.model_name
    )


if __name__ == '__main__':
    main()