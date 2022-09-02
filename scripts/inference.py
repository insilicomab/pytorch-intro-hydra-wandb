import numpy as np
import pandas as pd

import os

import torch
from torch.utils.data import DataLoader

import timm

import hydra
from omegaconf import DictConfig

import wandb

from src import set_device, seed_torch, DataTransform, CifarDataset


def predict_classes(model, test_dataloader, device):
    preds = []
    for images, _ in test_dataloader:
        images = images.to(device)
        
        model.eval()
        
        outputs = model(images)
        pred = torch.argmax(outputs, dim=1)
        pred = pred.to('cpu').numpy()

        preds.extend(pred)

        if len(preds) % 100 == 0:
            print(f'{len(preds)} predictions done!')

    return preds


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):

    # gpu or cpu
    device = set_device()
    
    # set up experiment
    seed_torch()

    # read data
    test = pd.read_csv(cfg.test_master, header=None, sep='\t')

    # image name list
    x_test = test[0].values

    # dummy label list
    dummy = test[0].values

    # dataset
    test_dataset = CifarDataset(x_test, dummy, cfg.dataset.test_img_path, transform=DataTransform(), phase='val')

    # dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # load model
    model = timm.create_model(
        cfg.model.name,
        pretrained=False,
        num_classes=cfg.model.num_classes
    )
    best_model = wandb.restore(cfg.load_model, run_path=cfg.wandb_run_path)
    model.load_state_dict(torch.load(best_model.name)) 
    model.to(device)

    # inference
    preds = predict_classes(model, test_dataloader, device)

    # submit
    sub = pd.read_csv(cfg.test_master, header=None, sep='\t')
    sub[1] = preds
    sub.to_csv(os.path.join(cfg.submit_path, cfg.submit_name), sep='\t', header=None, index=None)


if __name__ == '__main__':
    main()