import numpy as np
import matplotlib.pyplot as plt

import os

import torch

from collections import defaultdict

import wandb


# training model
def train_model(model, epochs, dataloader, device, loss_fn, optimizer, scheduler, earlystopping, save_model_path, model_name):

    # dictionary object to save learning history
    history = defaultdict(list)
    
    # initialize best score
    best_loss = np.inf
    
    for epoch in range(epochs):
        
        print(f'Epoch: {epoch+1} / {epochs}')
        print('--------------------------')
        
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            # reset epock loss
            epoch_loss = 0.0
            # number of correct
            corrects = 0
            # reset prediction list
            pred_list = []
            # reset true list
            true_list = []
            
            for images, labels in dataloader[phase]:
                
                images = images.to(device)
                labels = labels.to(device)
                
                # initialize grad
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    
                    outputs = model(images)

                    loss = loss_fn(outputs, labels)

                    preds = torch.argmax(outputs, dim=1)
                    
                    if phase == 'train':
                        
                        loss.backward()

                        optimizer.step()
                    
                    epoch_loss += loss.item() * images.size(0)
                    
                    corrects += torch.sum(preds == labels.data)                    
                    
                    preds = preds.to('cpu').numpy()
                    pred_list.extend(preds)
                    
                    labels = labels.to('cpu').numpy()
                    true_list.extend(labels)
            
            epoch_loss = epoch_loss / len(dataloader[phase].dataset)
            
            accuracy = corrects.double() / len(dataloader[phase].dataset)
            accuracy = accuracy.to('cpu').detach().numpy().copy()
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_accuracy'].append(accuracy)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f}')
            
            if (phase == 'val') and (epoch_loss < best_loss):
                
                ## only if the validation score improves, do the following
                
                # best score update
                best_loss = epoch_loss

                param_name = f'{save_model_path}{model_name}_loss_{best_loss:.4f}.pth'

                torch.save(model.state_dict(), param_name)
        
        scheduler.step(epoch_loss)

        # determine early termination by passing loss of validation data to EarlyStopping object
        if (phase == 'val') and earlystopping(epoch_loss):
            # if the loss is not improved at the monitored epoch, the learning will be terminated
            break
    
    return history


# training model with wandb
def train_model_wb(model, epochs, dataloader, device, loss_fn, optimizer, scheduler, earlystopping, save_model_path, model_name):
    
    # initialize best score
    best_loss = np.inf
    
    for epoch in range(epochs):
        
        print(f'Epoch: {epoch+1} / {epochs}')
        print('--------------------------')
        
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            # reset epock loss
            epoch_loss = 0.0
            # number of correct
            corrects = 0
            # reset prediction list
            pred_list = []
            # reset true list
            true_list = []
            
            for images, labels in dataloader[phase]:
                
                images = images.to(device)
                labels = labels.to(device)
                
                # initialize grad
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    
                    outputs = model(images)

                    loss = loss_fn(outputs, labels)

                    preds = torch.argmax(outputs, dim=1)
                    
                    if phase == 'train':
                        
                        loss.backward()

                        optimizer.step()
                    
                    epoch_loss += loss.item() * images.size(0)
                    
                    corrects += torch.sum(preds == labels.data)                    
                    
                    preds = preds.to('cpu').numpy()
                    pred_list.extend(preds)
                    
                    labels = labels.to('cpu').numpy()
                    true_list.extend(labels)
            
            epoch_loss = epoch_loss / len(dataloader[phase].dataset)
            
            accuracy = corrects.double() / len(dataloader[phase].dataset)
            accuracy = accuracy.to('cpu').detach().numpy().copy()

            wandb.log({f'{phase}_loss': epoch_loss, f'{phase}_accuracy': accuracy})
            
            print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f}')
            
            if (phase == 'val') and (epoch_loss < best_loss):
                
                ## only if the validation score improves, do the following
                
                # best score update
                best_loss = epoch_loss

                param_name = f'{save_model_path}{model_name}_loss_{best_loss:.4f}.pth'

                torch.save(model.state_dict(), param_name)
                wandb.save(f'{model_name}.ckpt')
                wandb.save(f'{model_name}.pth')
                wandb.save(f'{model_name}.h5')
        
        scheduler.step(epoch_loss)

        # determine early termination by passing loss of validation data to EarlyStopping object
        if (phase == 'val') and earlystopping(epoch_loss):
            # if the loss is not improved at the monitored epoch, the learning will be terminated
            break


class TrainModel():

    def __init__(self, dataloader, epochs, device, loss_fn, optimizer, scheduler, earlystopping):
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.earlystopping = earlystopping

    
    def train(self, model, save_model_path, model_name):

        # dictionary object to save learning history
        history = defaultdict(list)
        
        # initialize best score
        best_loss = np.inf
        
        for epoch in range(self.epochs):
            
            print(f'Epoch: {epoch+1} / {self.epochs}')
            print('--------------------------')
            
            for phase in ['train', 'val']:
                
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                # reset epock loss
                epoch_loss = 0.0
                # number of correct
                corrects = 0
                # reset prediction list
                pred_list = []
                # reset true list
                true_list = []
                
                for images, labels in self.dataloader[phase]:
                    
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # initialize grad
                    self.optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase=='train'):
                        
                        outputs = model(images)

                        loss = self.loss_fn(outputs, labels)

                        preds = torch.argmax(outputs, dim=1)
                        
                        if phase == 'train':
                            
                            loss.backward()

                            self.optimizer.step()
                        
                        epoch_loss += loss.item() * images.size(0)
                        
                        corrects += torch.sum(preds == labels.data)                    
                        
                        preds = preds.to('cpu').numpy()
                        pred_list.extend(preds)
                        
                        labels = labels.to('cpu').numpy()
                        true_list.extend(labels)
                
                epoch_loss = epoch_loss / len(self.dataloader[phase].dataset)
                
                accuracy = corrects.double() / len(self.dataloader[phase].dataset)
                accuracy = accuracy.to('cpu').detach().numpy().copy()
                
                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_accuracy'].append(accuracy)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f}')
                
                if (phase == 'val') and (epoch_loss < best_loss):
                    
                    ## only if the validation score improves, do the following
                    
                    # best score update
                    best_loss = epoch_loss

                    param_name = f'{save_model_path}{model_name}_loss_{best_loss:.4f}.pth'

                    torch.save(model.state_dict(), param_name)
            
            self.scheduler.step(epoch_loss)

            # determine early termination by passing loss of validation data to EarlyStopping object
            if (phase == 'val') and self.earlystopping(epoch_loss):
                # if the loss is not improved at the monitored epoch, the learning will be terminated
                break
        
        return history


def plot_acc_loss(history):
    
    # plot the percentage of correct answers
    plt.plot(history['train_accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Accuracy')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # plot loss trends
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Loss')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
