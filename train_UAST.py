#!/usr/bin/env python3
import os
import sys
import glob
import pickle
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from natsort import natsorted

# Import your model configuration and UAST-Net model.
from models.UAST_Net import CONFIGS as CONFIGS_TM
import models.UAST_Net as UAST_Net

###########################################
# 1. Custom Loss Function
###########################################
def custom_loss(mean_pred, std_pred, mean_scores, var_scores, target, criterion,
                lambda_mean=1.0, lambda_var=1.0, lambda_stdvar=1.0, lambda_meanpred=1.0):
    """
    Computes a composite loss with four components:
      1. Mean Loss:       Ensure mean_scores ~ target.
      2. Variance Loss:   Drive var_scores to zero.
      3. Std/Variance Consistency: Enforce std_pred ~ var_scores.
      4. Mean Consistency:         Enforce mean_pred ~ mean_scores.
    """
    loss_mean = criterion(mean_scores, target)
    # loss_mean = F.mse_loss(mean_scores, target)
    loss_var = F.mse_loss(var_scores, torch.zeros_like(var_scores))
    loss_stdvar = F.mse_loss(std_pred, var_scores)
    loss_meanpred = F.mse_loss(mean_pred, mean_scores)
    total_loss = (lambda_mean * loss_mean +
                  lambda_var * loss_var +
                  lambda_stdvar * loss_stdvar +
                  lambda_meanpred * loss_meanpred)
    return total_loss

###########################################
# 2. Dataset for Pickle Files (Classification)
###########################################
class PklClassificationDataset(Dataset):
    def __init__(self, smoke_dir, nosmoke_dir, transform=None):
        """
        Args:
            smoke_dir (str): Directory containing pickle files for smoke samples.
            nosmoke_dir (str): Directory containing pickle files for nosmoke samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.samples = []  # list of tuples: (filepath, label)
        for fname in os.listdir(smoke_dir):
            self.samples.append((os.path.join(smoke_dir, fname), 1.0))
        for fname in os.listdir(nosmoke_dir):
            self.samples.append((os.path.join(nosmoke_dir, fname), 0.0))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        # Assume data_dict has keys 'rgb' and 'mask' (each shape: H x W x 3)
        # For this example, we convert each to grayscale by averaging the color channels.
        source = data_dict['rgb']            # (H, W, 3)
        seg = data_dict['mask']              # (H, W, 3)
        # Stack the two channels to create a 2-channel input.
        x = np.concatenate([source, seg], axis=2)        # shape: (H, W, 6)
        # Convert to tensor and change to (C, H, W)
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor([label], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return {'input': x, 'label': label}
        
###########################################
# 3. Logger (for saving console output)
###########################################
class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, "logfile.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

###########################################
# 4. Utility Functions
###########################################
def adjust_learning_rate(optimizer, epoch, max_epochs, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - epoch / max_epochs, power), 8)

def save_checkpoint(state, save_dir, filename='checkpoint.pth.tar', max_model_num=4):
    ckpt_path = os.path.join(save_dir, filename)
    torch.save(state, ckpt_path)
    ckpt_list = natsorted(glob.glob(os.path.join(save_dir, '*')))
    while len(ckpt_list) > max_model_num:
        os.remove(ckpt_list[0])
        ckpt_list = natsorted(glob.glob(os.path.join(save_dir, '*')))
        
###########################################
# New function to save epoch information to a JSON file.
###########################################
def save_epoch_info(epoch_info, save_path):
    with open(save_path, 'w') as f:
        json.dump(epoch_info, f, indent=4)

###########################################
# 5. Main Training Script
###########################################
def main():
    # Hyperparameters and paths
    batch_size = 2
    lr = 1e-4
    epoch_start = 0
    max_epochs = 100
    cont_training = False
    # Initialize BCEWithLogitsLoss for classification.
    criterion = nn.BCEWithLogitsLoss()

    # Set your training and validation directories (adjust these paths)
    train_smoke_dir = '/home/khanm/workfolder/UAST/Train_data/train/smoke/'
    train_nosmoke_dir = '/home/khanm/workfolder/UAST/Train_data/train/nosmoke/'
    val_smoke_dir   = '/home/khanm/workfolder/UAST/Train_data/val/smoke/'
    val_nosmoke_dir = '/home/khanm/workfolder/UAST/Train_data/val/nosmoke/'

    # Directories for saving checkpoints and logs
    save_dir = os.path.join('experiments', 'UAST_classification')
    log_dir = os.path.join('logs', 'UAST_classification')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Logger(log_dir)

    # Initialize the model from configuration.
    config = CONFIGS_TM['UASTNet']
    model = UAST_Net.UASTNet(config)
    model.cuda()

    # Optionally resume training from a checkpoint.
    if cont_training:
        ckpt_list = natsorted(os.listdir(save_dir))
        if ckpt_list:
            ckpt_path = os.path.join(save_dir, ckpt_list[-1])
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['state_dict'])
            epoch_start = checkpoint['epoch']
            print(f"Resuming training from epoch {epoch_start}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)

    # Create the training and validation datasets and loaders.
    train_dataset = PklClassificationDataset(train_smoke_dir, train_nosmoke_dir)
    val_dataset   = PklClassificationDataset(val_smoke_dir, val_nosmoke_dir)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    writer = SummaryWriter(log_dir=log_dir)
    
    # This list will store per-epoch info for saving as JSON.
    epoch_history = []
    json_save_path = os.path.join(save_dir, "epoch_info.json")


    best_val_acc = 0.0
    for epoch in range(epoch_start, max_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        adjust_learning_rate(optimizer, epoch, max_epochs, lr)
        
        # Wrap the train_loader with tqdm
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
        
        for batch in train_bar:
            optimizer.zero_grad()
            inputs = batch['input'].cuda()   # shape: [B, 6, H, W]
            labels = batch['label'].cuda()     # shape: [B, 1]

            # Forward pass - note that our updated UAST-Net returns four outputs.
            mean_pred, std_pred, mean_scores, var_scores = model(inputs)
            loss = custom_loss(mean_pred, std_pred, mean_scores, var_scores, labels, criterion)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            # For classification, threshold mean_scores at 0.5.
            preds = (torch.sigmoid(mean_scores) > 0.5).float()
            # preds = (mean_scores > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}")

        # Validation phase.
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Optionally wrap the validation loop with tqdm as well.
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch in val_bar:
                inputs = batch['input'].cuda()
                labels = batch['label'].cuda()
                mean_pred, std_pred, mean_scores, var_scores = model(inputs)
                loss = custom_loss(mean_pred, std_pred, mean_scores, var_scores, labels, criterion)
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(mean_scores) > 0.5).float()
                # preds = (mean_scores > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss = val_loss / total
        val_acc = correct / total
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        print(f"Epoch {epoch}: Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

        # Save the model checkpoint if validation accuracy improves.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }
            save_checkpoint(checkpoint, save_dir=save_dir,
                            filename=f"epoch_{epoch+1}_acc_{val_acc:.4f}.pth.tar")
                            
        # Save epoch statistics to JSON.
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        epoch_history.append(epoch_data)
        save_epoch_info(epoch_history, json_save_path)

        
    writer.close()

###########################################
# 6. Script Entry Point and GPU Setup
###########################################
if __name__ == '__main__':
    # Optionally set the GPU device.
    GPU_iden = 0
    torch.cuda.set_device(GPU_iden)
    print('Using GPU:', torch.cuda.get_device_name(GPU_iden))
    main()