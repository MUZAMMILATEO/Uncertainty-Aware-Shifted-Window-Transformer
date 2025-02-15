#!/usr/bin/env python3
import os
import sys
import glob
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, confusion_matrix, classification_report)
import seaborn as sns  # for a nicer confusion matrix plot (optional)

# Import your model configuration and UAST-Net model.
from models.UAST_Net import CONFIGS as CONFIGS_TM
import models.UAST_Net as UAST_Net

###########################################
# Dataset for Pickle Files (Test)
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
        # Assign label 1.0 for smoke and 0.0 for nosmoke.
        for fname in os.listdir(smoke_dir):
            if fname.endswith('.pkl'):
                self.samples.append((os.path.join(smoke_dir, fname), 1.0))
        for fname in os.listdir(nosmoke_dir):
            if fname.endswith('.pkl'):
                self.samples.append((os.path.join(nosmoke_dir, fname), 0.0))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        # Assume data_dict has keys 'rgb' and 'mask' (each shape: H x W x 3)
        # Here we simply stack the two channels without conversion to grayscale.
        source = data_dict['rgb']  # shape: (H, W, 3)
        seg = data_dict['mask']    # shape: (H, W, 3)
        # Create a multi-channel input by concatenating along the channel dimension.
        # (If your model expects 6 channels, adjust accordingly. Here we assume 6 channels.)
        x = np.concatenate([source, seg], axis=2)  # shape: (H, W, 6)
        # Convert to tensor and permute to (C, H, W)
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor([label], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return {'input': x, 'label': label}

###########################################
# Utility Functions for Metrics and Plotting
###########################################
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_probability_histograms(probs, labels, save_path=None):
    """
    Plot the distribution of predicted probabilities for the two classes.
      - probs: numpy array of predicted probabilities (N,)
      - labels: numpy array of ground truth labels (N,) with values 0 or 1.
    """
    smoke_probs = probs[labels==1]
    nosmoke_probs = probs[labels==0]

    plt.figure(figsize=(10, 5))
    plt.hist(smoke_probs, bins=20, alpha=0.7, label='Smoke (GT label=1)', color='red')
    plt.hist(nosmoke_probs, bins=20, alpha=0.7, label='No Smoke (GT label=0)', color='blue')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
def plot_samples_with_uncertainty(images, mean_scores, std_scores, num_samples=8, save_path=None):
    """
    Plots a grid of sample images with predicted mean probability and uncertainty (std).

    Args:
        images (list or numpy array): A list or numpy array of images (each image as a tensor or array in shape (C, H, W)).
                                      If in channel-first order (C, H, W) and C >= 3, the first 3 channels are used as RGB.
        mean_scores (array-like): Array of predicted mean scores (raw logits) for each sample.
        std_scores (array-like): Array of predicted standard deviations (uncertainty) for each sample.
        num_samples (int): Number of samples to plot (default: 8).
        save_path (str, optional): If provided, saves the figure to this path.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # If inputs are PyTorch tensors, convert to NumPy.
    if torch.is_tensor(images):
        images = images.cpu().numpy()
    if torch.is_tensor(mean_scores):
        mean_scores = mean_scores.cpu().numpy()
    if torch.is_tensor(std_scores):
        std_scores = std_scores.cpu().numpy()

    # Ensure arrays are flattened along the sample dimension.
    mean_scores = mean_scores.flatten()
    std_scores = std_scores.flatten()

    # Determine how many samples to plot.
    num_samples = min(num_samples, len(images))
    
    # Create a grid (e.g., 2 columns)
    ncols = 2
    nrows = (num_samples + ncols - 1) // ncols

    plt.figure(figsize=(ncols * 5, nrows * 4))
    for i in range(num_samples):
        # Get the image: assume shape (C, H, W). We'll take the first 3 channels for RGB.
        img = images[i]
        if img.shape[0] >= 3:
            # Convert from channel-first (C, H, W) to (H, W, C)
            rgb_img = np.transpose(img[:3, :, :], (1, 2, 0))
            # Normalize to [0,1] if necessary (assuming values are in 0-1 range already)
        else:
            # If only one channel, duplicate for visualization.
            rgb_img = np.transpose(img, (1, 2, 0))
            rgb_img = np.repeat(rgb_img, 3, axis=2)

        # Apply sigmoid to mean score to get probability.
        prob = 1 / (1 + np.exp(-mean_scores[i]))
        uncertainty = std_scores[i]
        
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.imshow(rgb_img)
        ax.axis('off')
        ax.set_title(f"Prob: {prob:.2f}, Unc: {uncertainty:.2f}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

###########################################
# Main Inference/Test Function
###########################################
def run_inference(test_smoke_dir, test_nosmoke_dir, checkpoint_path, batch_size=2):
    # Create the test dataset and DataLoader.
    test_dataset = PklClassificationDataset(test_smoke_dir, test_nosmoke_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load the model configuration and initialize the model.
    config = CONFIGS_TM['UASTNet']
    model = UAST_Net.UASTNet(config)
    model.cuda()

    # Load the checkpoint.
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint file {checkpoint_path} not found!")
        sys.exit(1)

    model.eval()

    # Lists to collect ground truth and predictions.
    all_labels = []
    all_preds = []
    all_probs = []  # predicted probabilities from sigmoid on mean_scores
    
    # For visualization, collect a subset of images, mean scores, and std scores.
    sample_images = []
    sample_mean_scores = []
    sample_std_scores = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch['input'].cuda()   # shape: [B, 6, H, W] if 3+3 channels
            labels = batch['label'].cuda()     # shape: [B, 1]
            # Run inference; here we assume that mean_scores is the classification output.
            # (Our model returns: mean_pred, std_pred, mean_scores, var_scores)
            mean_scores, std_scores, _, _ = model(inputs)
            # Apply sigmoid to get probability.
            probs = torch.sigmoid(mean_scores)
            preds = (probs > 0.5).float()
            
            # Save the first batch (or a subset) for visualization.
            if i == 0:
                sample_images.append(inputs.cpu())
                sample_mean_scores.append(mean_scores.cpu())
                sample_std_scores.append(std_scores.cpu())

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Concatenate results from all batches.
    all_labels = np.concatenate(all_labels).flatten()
    all_preds = np.concatenate(all_preds).flatten()
    all_probs = np.concatenate(all_probs).flatten()
    
    # Concatenate sample batches if needed.
    sample_images = torch.cat(sample_images, dim=0).numpy()
    sample_mean_scores = torch.cat(sample_mean_scores, dim=0).numpy().flatten()
    sample_std_scores = torch.cat(sample_std_scores, dim=0).numpy().flatten()
    
    # Plot a few samples with their uncertainty.
    plot_samples_with_uncertainty(sample_images, sample_mean_scores, sample_std_scores,
                                  num_samples=8, save_path="sample_uncertainty.png")


    # Compute Metrics.
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("Test Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    # Plot confusion matrix.
    plot_confusion_matrix(cm, classes=['No Smoke', 'Smoke'], title='Confusion Matrix')

    # Plot probability distribution histograms.
    plot_probability_histograms(all_probs, all_labels)

    return all_labels, all_preds, all_probs

###########################################
# Main Script with Argument Parsing
###########################################
def main():
    parser = argparse.ArgumentParser(description="Test script for UAST-Net classification")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to the test folder containing subfolders 'smoke' and 'nosmoke'")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (pth file)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for testing")
    args = parser.parse_args()

    # Test folder structure:
    # test_dir/smoke/   -> contains pickle files for smoke samples
    # test_dir/nosmoke/ -> contains pickle files for nosmoke samples
    test_smoke_dir = os.path.join(args.test_dir, "smoke")
    test_nosmoke_dir = os.path.join(args.test_dir, "nosmoke")

    run_inference(test_smoke_dir, test_nosmoke_dir, args.checkpoint, batch_size=args.batch_size)

if __name__ == '__main__':
    # Optionally set GPU device.
    GPU_iden = 0
    torch.cuda.set_device(GPU_iden)
    print('Using GPU:', torch.cuda.get_device_name(GPU_iden))
    main()
