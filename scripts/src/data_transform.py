from torchvision import transforms


class DataTransform():

    def __init__(self):
        
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(96),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomRotation(degrees=20),
                transforms.RandomAffine(degrees=[-10, 10], translate=(0.1, 0.1), scale=(0.5, 1.5)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]),
            'val': transforms.Compose([
                transforms.Resize(96),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]),
        }
    
    def __call__(self, phase, img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
        """
        return self.data_transform[phase](img)