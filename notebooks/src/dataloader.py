from torch.utils.data import DataLoader


def make_dataloader(tr_dataset, val_dataset, tr_batch_size, val_batch_size):
    dataloader = {
        'train': DataLoader(tr_dataset, batch_size=tr_batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    }
    
    return dataloader