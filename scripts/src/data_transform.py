from torchvision import transforms
from omegaconf import DictConfig


class DataTransform():

    def __init__(self, cfg: DictConfig):
        
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(cfg.transform.image_size),
                transforms.RandomHorizontalFlip(p=cfg.transform.randomhorizontalflip.p),
                transforms.RandomRotation(degrees=cfg.transform.randomrotation.degrees),
                transforms.RandomAffine(
                    degrees=cfg.transform.randomaffine.degrees,
                    translate=cfg.transform.randomaffine.translate,
                    scale=cfg.transform.randomaffine.scale
                    ),
                transforms.ColorJitter(
                    brightness=cfg.transform.colorjitter.brightness,
                    contrast=cfg.transform.colorjitter.contrast,
                    saturation=cfg.transform.colorjitter.saturation
                    ),
                transforms.ToTensor(),
                transforms.Normalize(
                    cfg.transform.normalize.mean,
                    cfg.transform.normalize.std
                    ),
                ]),
            'val': transforms.Compose([
                transforms.Resize(cfg.transform.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    cfg.transform.normalize.mean,
                    cfg.transform.normalize.std
                    ),
                ]),
        }
    
    def __call__(self, phase, img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
        """
        return self.data_transform[phase](img)