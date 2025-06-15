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

import random
import torchvision.transforms.functional as TF

###########################################
# 0. Custom Transformation Class
###########################################
class Custom6ChannelTransform:
    def __init__(self, output_size=(256, 256), rotation=15, hflip_prob=0.5):
        self.output_size = output_size
        self.rotation = rotation
        self.hflip_prob = hflip_prob

    def __call__(self, x):
        x = TF.resize(x, self.output_size)
        if random.random() < self.hflip_prob:
            x = TF.hflip(x)
        angle = random.uniform(-self.rotation, self.rotation)
        x = TF.rotate(x, angle)
        return x

###########################################
# 1. Custom Loss Function
###########################################
def custom_loss(mean_pred, log_var_pred, mean_scores, var_scores, target, criterion,
                lambda_mean=1.0, lambda_var=0.0, lambda_stdvar=1.0, lambda_meanpred=1.0,
                ema_decay=0.99, eps=1e-6):
    """
    Original composite loss for phase 1.
    Note: Here log_var_pred is used in place of the previous std_pred.
    """
    loss_mean     = criterion(mean_scores, target)
    loss_var      = F.smooth_l1_loss(var_scores, torch.zeros_like(var_scores))
    loss_stdvar   = F.smooth_l1_loss(log_var_pred, var_scores)
    loss_meanpred = criterion(mean_pred, target)
    if not hasattr(custom_loss, 'running_mean_loss'):
        custom_loss.running_mean_loss = loss_mean.item()
        custom_loss.running_var_loss = loss_var.item()
        custom_loss.running_stdvar_loss = loss_stdvar.item()
        custom_loss.running_meanpred_loss = loss_meanpred.item()
    else:
        custom_loss.running_mean_loss = ema_decay * custom_loss.running_mean_loss + (1 - ema_decay) * loss_mean.item()
        custom_loss.running_var_loss  = ema_decay * custom_loss.running_var_loss  + (1 - ema_decay) * loss_var.item()
        custom_loss.running_stdvar_loss = ema_decay * custom_loss.running_stdvar_loss + (1 - ema_decay) * loss_stdvar.item()
        custom_loss.running_meanpred_loss = ema_decay * custom_loss.running_meanpred_loss + (1 - ema_decay) * loss_meanpred.item()

    normalized_loss_mean     = loss_mean / (custom_loss.running_mean_loss + eps)
    normalized_loss_var      = loss_var / (custom_loss.running_var_loss + eps)
    normalized_loss_stdvar   = loss_stdvar / (custom_loss.running_stdvar_loss + eps)
    normalized_loss_meanpred = loss_meanpred / (custom_loss.running_meanpred_loss + eps)
    total_loss = (lambda_mean * normalized_loss_mean +
                  lambda_var * normalized_loss_var +
                  lambda_stdvar * normalized_loss_stdvar +
                  lambda_meanpred * normalized_loss_meanpred) / (lambda_mean + lambda_var + lambda_stdvar + lambda_meanpred)
    return total_loss

###########################################
# Heteroscedastic Loss Function (Phase 2)
###########################################

def std_regression_loss(log_var_pred, var_scores, eps=1e-6):
    """
    Regress the predicted log-variance onto the empirical variance.
    
    - log_var_pred: s = log(sigma^2) output by your std branch
    - var_scores:   empirical variance (std of MC-dropout) from your forward()
    
    Returns a Smooth L1 loss between sqrt(exp(s)) and var_scores.
    """
    # convert log-variance -> std
    pred_std = torch.exp(0.5 * log_var_pred)
    # Smooth L1 is robust to outliers
    return F.smooth_l1_loss(pred_std, var_scores)

###########################################
# 2. Dataset for Pickle Files (Classification)
###########################################
class PklClassificationDataset(Dataset):
    def __init__(self, smoke_dir, nosmoke_dir, transform=None):
        self.samples = []
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
        source = data_dict['rgb']  # shape: (H, W, 3)
        seg = data_dict['mask']    # shape: (H, W, 3)
        x = np.concatenate([source, seg], axis=2)  # shape: (H, W, 6)
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

def save_epoch_info(epoch_info, save_path):
    with open(save_path, 'w') as f:
        json.dump(epoch_info, f, indent=4)

###########################################
# 5. Main Training Script
###########################################
def main():
    # Hyperparameters and paths
    batch_size = 4
    lr = 1e-4
    epoch_start = 0
    max_epochs = 100
    cont_training = False
    criterion = nn.BCEWithLogitsLoss()
    transform = Custom6ChannelTransform(output_size=(256, 256), rotation=15, hflip_prob=0.5)

    # Set directories (adjust these paths as necessary)
    train_smoke_dir = '/home/khanm/workfolder/UAST/Train_data/train_Copy/smoke/'
    train_nosmoke_dir = '/home/khanm/workfolder/UAST/Train_data/train_Copy/nosmoke/'
    val_smoke_dir   = '/home/khanm/workfolder/UAST/Train_data/val_Copy/smoke/'
    val_nosmoke_dir = '/home/khanm/workfolder/UAST/Train_data/val_Copy/nosmoke/'

    # Directories for saving checkpoints and logs
    save_dir = os.path.join('experiments', 'UAST_classification')
    log_dir = os.path.join('logs', 'UAST_classification')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Logger(log_dir)

    # Initialize the model using your configuration.
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

    # Create the datasets and loaders.
    train_dataset = PklClassificationDataset(train_smoke_dir, train_nosmoke_dir, transform=transform)
    val_dataset   = PklClassificationDataset(val_smoke_dir, val_nosmoke_dir, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    writer = SummaryWriter(log_dir=log_dir)
    epoch_history = []
    json_save_path = os.path.join(save_dir, "epoch_info.json")

    # Initialize flag and threshold for phase transition.
    activate_stdvar = False
    loss_threshold = 0.2
    loss_checkpoint_saved = False
    best_val_acc = 0.0
    least_val_loss = float('inf')

    for epoch in range(epoch_start, max_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        correct_difference = 0.0
        total = 0

        adjust_learning_rate(optimizer, epoch, max_epochs, lr)
        
        # For phase 1 (activate_stdvar==False) we use the custom_loss;
        # for phase 2 (activate_stdvar==True) we use the heteroscedastic loss.
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
        for batch in train_bar:
            optimizer.zero_grad()
            inputs = batch['input'].cuda()
            labels = batch['label'].cuda()

            # Get outputs from model.
            # Note: The model forward method now returns:
            # final_mean_pred, final_log_var_pred, mean_scores, var_scores
            final_mean_pred, final_log_var_pred, mean_scores, var_scores = model(inputs)
            
            if activate_stdvar:
                loss = std_regression_loss(final_log_var_pred, var_scores)
            else:
                # Use custom composite loss for phase 1.
                loss = custom_loss(final_mean_pred, final_log_var_pred, mean_scores, var_scores,
                                   labels, criterion,
                                   lambda_mean=1.0, lambda_var=0.0, lambda_stdvar=0.0, lambda_meanpred=1.0)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(mean_scores) > 0.5).float()
            correct += (preds == labels).sum().item()
            probs = torch.sigmoid(mean_scores)             # shape [B,1]
            rounded = torch.round(probs)                   # shape [B,1]
            diffs = torch.abs(probs - rounded)             # shape [B,1]
            correct_difference += diffs.sum().item()       # add up all |p - round(p)|
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_acc_diff = correct_difference / total

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy_difference/train', train_acc_diff, epoch)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Train Acc Diff {train_acc_diff: .4f}")

        # Check for phase transition: if not in phase 2 and training loss drops below threshold.
        if not activate_stdvar and train_loss < loss_threshold:
            activate_stdvar = True
            print("Activating phase 2: freezing all weights except class_std_pred and switching to heteroscedastic loss.")
            model.freeze_except_std()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr, weight_decay=0, amsgrad=True)

        # Validation phase.
        model.eval()
        val_loss = 0.0
        correct = 0
        correct_difference_val = 0.0
        total = 0
        
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                inputs = batch['input'].cuda()
                labels = batch['label'].cuda()
                final_mean_pred, final_log_var_pred, mean_scores, var_scores = model(inputs)
                if activate_stdvar:
                    loss = std_regression_loss(final_log_var_pred, var_scores)
                else:
                    loss = custom_loss(final_mean_pred, final_log_var_pred, mean_scores, var_scores,
                                       labels, criterion,
                                       lambda_mean=1.0, lambda_var=0.0, lambda_stdvar=0.0, lambda_meanpred=1.0)
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(mean_scores) > 0.5).float()
                
                probs_val = torch.sigmoid(mean_scores)             # shape [B,1]
                rounded_val = torch.round(probs_val)                   # shape [B,1]
                diffs_val = torch.abs(probs_val - rounded_val)             # shape [B,1]
                correct_difference_val += diffs_val.sum().item()       # add up all |p - round(p)|
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss = val_loss / total
        val_acc = correct / total
        val_acc_diff = correct_difference_val / total

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Accuracy_difference/val', val_acc_diff, epoch)
        print(f"Epoch {epoch}: Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Val Acc Diff {val_acc_diff:.4f}")

        # Save checkpoint if validation accuracy improves.
        # 1) Save on accuracy improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'least_val_loss': least_val_loss,
            }
            save_checkpoint(
                checkpoint, save_dir=save_dir,
                filename=f"epoch_{epoch+1:03d}_acc_{val_acc:.4f}_loss_{val_loss:.4f}.pth.tar"
            )

        # 2) Save on loss improvement
        if val_loss < least_val_loss:
            least_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'least_val_loss': least_val_loss,
            }
            save_checkpoint(
                checkpoint, save_dir=save_dir,
                filename=f"epoch_{epoch+1:03d}_acc_{val_acc:.4f}_loss_{val_loss:.4f}.pth.tar"
            )
                            
        # Save epoch statistics to JSON.
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_acc_diff": train_acc_diff,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_acc_diff": val_acc_diff,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        epoch_history.append(epoch_data)
        save_epoch_info(epoch_history, json_save_path)

    writer.close()

###########################################
# 6. Script Entry Point and GPU Setup
###########################################
if __name__ == '__main__':
    GPU_iden = 0
    torch.cuda.set_device(GPU_iden)
    print('Using GPU:', torch.cuda.get_device_name(GPU_iden))
    main()
