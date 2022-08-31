import os
import random

import numpy as np

import torch
from google.colab import drive


# set path to save model(Google Drive or Not)
def colab_save_model(colab_flg: bool, attach_path: str):

    if colab_flg:
        drive.mount('/content/drive')
        ATTACH_PATH = attach_path
    else:
        ATTACH_PATH = '.'
        
        
    SAVE_MODEL_PATH = f'{ATTACH_PATH}/model/'

    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    print(f'Save model path: {SAVE_MODEL_PATH}')
    return SAVE_MODEL_PATH


# set device (gpu or cpu)
def set_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    print(f'device：{device}')
    return device


# set random_seed
def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print('set random seed')


# early termination for training
class EarlyStopping:
    def __init__(self, patience=10, verbose=0):
        '''
        Parameters:
            patience(int): Number of epochs to monitor (default is 10)
            verbose(int): Whether to display early termination
                          True (1),False (0)        
        '''
        # Initialize instance variables
        # Initialize a counter for the number of epochs being monitored
        self.epoch = 0
        # Initialize loss for comparison with infinity 'inf'.
        self.pre_loss = float('inf')
        # Initialize the number of epochs to be monitored with parameters
        self.patience = patience
        # Initialize output flag of early termination message with parameter
        self.verbose = verbose
        
    def __call__(self, current_loss):
        '''
        Parameters:
            current_loss(float): Loss of validation data after 1 epoch
        Return:
            True: If the loss of the previous epoch is exceeded by the maximum number of monitoring times
            False: If the loss of the previous epoch is not exceeded by the maximum number of times monitored
        '''

        if self.pre_loss < current_loss:
            self.epoch += 1

            if self.epoch > self.patience:
                if self.verbose:
                    print('early stopping')
                return True

        else:
            self.epoch = 0
            self.pre_loss = current_loss
        
        return False