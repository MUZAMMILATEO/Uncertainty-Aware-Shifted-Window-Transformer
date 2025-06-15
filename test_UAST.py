import os
import sys
import glob
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from natsort import natsorted
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
import seaborn as sns

# Import for confusion matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Import for calibration curve
from sklearn.calibration import calibration_curve

# Import your model configuration and UAST-Net model.
from models.UAST_Net import CONFIGS as CONFIGS_TM
import models.UAST_Net as UAST_Net

def display_classification_metrics(true_labels_for_cm, predicted_probs_for_cm, digits=4):
    """
    Computes and prints accuracy, precision, recall and F1 score.

    Args:
        y_true (list or array): Ground-truth binary labels (0 or 1).
        y_pred (list or array): Predicted binary labels (0 or 1).
        digits (int): Number of decimals to show.
    """
    predicted_probs_for_cm = [1 if p >= 0.5 else 0 for p in predicted_probs_for_cm]
    
    acc  = accuracy_score(true_labels_for_cm, predicted_probs_for_cm)
    prec = precision_score(true_labels_for_cm, predicted_probs_for_cm, zero_division=0)
    rec  = recall_score(true_labels_for_cm, predicted_probs_for_cm, zero_division=0)
    f1   = f1_score(true_labels_for_cm, predicted_probs_for_cm, zero_division=0)

    print("\n--- Classification Metrics ---")
    print(f"Accuracy : {acc:.{digits}f}")
    print(f"Precision: {prec:.{digits}f}")
    print(f"Recall   : {rec:.{digits}f}")
    print(f"F1 Score : {f1:.{digits}f}")
    print("-" * 30)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def calibration_curve_plotting(true_labels_for_cm, predicted_probs_for_cm, n_bin=10):
        # `y_true` are your true binary labels
        # `y_prob` are your predicted probabilities (mean_pred_for_dist)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true=true_labels_for_cm,
            y_prob=predicted_probs_for_cm,
            n_bins=n_bin  # You can adjust the number of bins
        )

        plt.figure(figsize=(10, 10)) # Increased figure size for better visibility
        ax = plt.gca() # Get current axes for tick adjustments

        # Plot the perfectly calibrated line
        plt.plot([0, 1], [0, 1], linestyle='--', linewidth=3.0, color='green', label='Perfectly Calibrated')

        # Plot the model's calibration curve
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', markersize=8, label='Model Calibration', color='blue', linewidth=2.0)
  
        # Set labels with increased font size and bold
        plt.xlabel('Mean Predicted Probability', fontsize=14, fontweight='bold')
        plt.ylabel('Fraction of Positives', fontsize=14, fontweight='bold')
        plt.title('Calibration Curve', fontsize=18, fontweight='bold')

        # Make tick marks bigger and bold
        ax.tick_params(axis='both', which='major', labelsize=12, width=2) # 'width' for tick line thickness
    
        # Improve legend appearance
        plt.legend(fontsize=12)
    
        plt.grid(True, linestyle=':', alpha=0.7) # Added stylistic grid
        plt.show()
        
def plot_uncertainty_hist(std_pred_list: list, title: str = 'Histogram of Predicted Uncertainty in Probability Space'):
    """
    Calculates and plots a histogram of the predicted uncertainties (standard deviations).

    Args:
        std_pred_list (list): A list of standard deviation values in probability space.
        title (str): The title for the histogram plot.
    """
    if not std_pred_list:
        print("No uncertainty data to plot for histogram.")
        return

    plt.figure(figsize=(10, 6))
    
    # You can adjust the number of bins or let matplotlib decide.
    # A common approach is to use a fixed number or a dynamic calculation like np.sqrt(len(std_pred_list)).
    num_bins = 200 # Example: using 20 bins. You can try other values.
    
    plt.hist(std_pred_list, bins=num_bins, edgecolor='black', alpha=0.7)

    plt.xlabel('Predicted Uncertainty', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')

    # Make tick marks bigger and bold
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=12, width=2)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
def plot_unc_vs_error(true_labels_for_cm, predicted_probs_for_cm, std_preds_for_hist):
    if len(true_labels_for_cm) == 0 or len(predicted_probs_for_cm) == 0 or len(std_preds_for_hist) == 0:
        print("Not enough data for uncertainty vs. error plot.")
        return
    # Convert lists to NumPy arrays for element-wise operations
    predicted_probs_np = np.array(predicted_probs_for_cm)
    true_labels_np = np.array(true_labels_for_cm)
    # errs = ((predicted_probs_np>0.5).astype(int) != true_labels_np).astype(float)
    errs = np.abs(predicted_probs_np - true_labels_np)
    plt.figure(figsize=(5,3))
    sc = plt.scatter(predicted_probs_np, std_preds_for_hist, c=errs, cmap="coolwarm", alpha=0.7,
                     edgecolor='black', linewidth=0.8)
    plt.colorbar(sc, label="Error")
    plt.xlabel("Predicted probability", fontweight='bold')
    plt.ylabel("Uncertainty", fontweight='bold')
    plt.title("Uncertainty vs. Error", fontweight='bold')
    plt.tight_layout()
    plt.show()
    
def plot_box_unc(true_labels_for_cm, std_preds_for_hist):
    if len(true_labels_for_cm) == 0 or len(std_preds_for_hist) == 0:
        print("Not enough data for uncertainty box plot.")
        return
        
    classes = np.where(np.array(true_labels_for_cm) > 0.5, "Smoke", "No Smoke")
    df = pd.DataFrame({"Uncertainty": std_preds_for_hist, "Class": classes})
    plt.figure(figsize=(4,3))
    sns.boxplot(x="Class", y="Uncertainty", data=df, palette="Set2",
                 linewidth=1.5, fliersize=3)
    plt.xlabel("Class", fontweight='bold')
    plt.ylabel("Uncertainty", fontweight='bold')
    plt.title("Uncertainty by Class", fontweight='bold')
    plt.tight_layout()
    plt.show()   
 
def plot_plausibility_histogram(z_score_for_cm):
    if len(z_score_for_cm) == 0:
        print("Not enough data for plausibility histogram.")
        return
        
    z_scores = np.array(z_score_for_cm, dtype=float)
    confidence = np.exp(-0.5 * z_scores**2)

    plt.figure(figsize=(10, 5))

    # Z-score histogram
    plt.subplot(1, 2, 1)
    plt.hist(z_scores, bins=50, edgecolor='black', alpha=0.8, density=True)
    plt.axvline(x=0, color='red', linestyle='--', label='Z=0')
    plt.axvline(x=1, color='green', linestyle=':', label='Z=1')
    plt.axvline(x=-1, color='green', linestyle=':')
    plt.axvline(x=2, color='orange', linestyle=':', label='Z=2')
    plt.axvline(x=-2, color='orange', linestyle=':')
    plt.axvline(x=3, color='purple', linestyle=':', label='Z=3')
    plt.axvline(x=-3, color='purple', linestyle=':')
    plt.xlabel("Z-score", fontweight='bold')
    plt.ylabel("Density", fontweight='bold')
    plt.title("Distribution of Z-scores (Plausibility)", fontweight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Confidence distribution
    plt.subplot(1, 2, 2)
    plt.hist(confidence, bins=30, edgecolor='black', alpha=0.8, density=True)
    
    plt.axvline(x=np.exp(-0.5 * 0**2), color='red', linestyle='--', label='Mean (Z=0)')
    plt.axvline(x=np.exp(-0.5 * 1**2), color='green', linestyle=':', label='Z=1')
    plt.axvline(x=np.exp(-0.5 * 2**2), color='orange', linestyle=':', label='Z=2')
    plt.axvline(x=np.exp(-0.5 * 3**2), color='purple', linestyle=':', label='Z=3')
    
    plt.xlabel(r"Confidence $(e^{-Z^2/2})$", fontweight='bold')
    plt.ylabel("Density", fontweight='bold')
    plt.title("Distribution of Plausibility Confidence", fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    
def analyze_model_outputs(mean_pred_logit: float, std_pred_logit: float, sample_logit: float) -> dict:
    """
    Analyzes model outputs (logits and standard deviation) to provide corresponding
    probabilities and a Z-score.

    Args:
        mean_pred_logit (float): The mean prediction from the model's logit output.
                                     This is the 'mean' of the distribution in logit space.
        std_pred_logit (float): The standard deviation of the prediction in logit space.
                                 This value is assumed to be positive (e.g., already
                                 transformed from log-variance using torch.exp(0.5 * log_var)).
        sample_logit (float): A specific sample value (logit) from the distribution,
                                 which could be a random draw or another specific logit
                                 value you want to evaluate the Z-score for.

    Returns:
        dict: A dictionary containing the following calculated values:
            - 'predicted_probability_mean': The sigmoid-transformed probability of the mean_pred_logit.
            - 'std_probability_space': The approximate standard deviation in probability space
                                         calculated using the Delta Method.
            - 'sample_probability': The sigmoid-transformed probability of the sample_logit.
            - 'z_score': The Z-score of the sample_logit relative to the mean_pred_logit,
                         normalized by std_pred_logit.
    """
    # Ensure inputs are torch tensors for operations, then convert back to float for return
    mean_pred_logit_tensor = torch.tensor(mean_pred_logit, dtype=torch.float32)
    std_pred_logit_tensor = torch.tensor(std_pred_logit, dtype=torch.float32)
    sample_logit_tensor = torch.tensor(sample_logit, dtype=torch.float32)

    # 1. Predicted Probability (Mean)
    predicted_probability_mean = torch.sigmoid(mean_pred_logit_tensor).item()

    # 2. Standard Deviation in Probability Space (Delta Method)
    # Derivative of sigmoid: sigma(x) * (1 - sigma(x))
    sigmoid_derivative_at_mean = predicted_probability_mean * (1 - predicted_probability_mean)
    
    # Ensure std_pred_logit is positive for calculation, though it should be by design
    # Add a small epsilon for numerical stability in case std_pred_logit is extremely close to zero
    epsilon = 1e-6 
    if std_pred_logit_tensor.item() < 0:
        print("Warning: std_pred_logit is negative. It should be positive. Using its absolute value.")
        std_pred_logit_tensor = torch.abs(std_pred_logit_tensor)

    std_probability_space = abs(sigmoid_derivative_at_mean) * std_pred_logit_tensor.item()

    # 3. Sample Probability
    sample_probability = torch.sigmoid(sample_logit_tensor).item()

    # 4. Z-Score Calculation
    # Z = (sample_logits - mean_pred) / std_pred
    # Using the provided inputs directly for the Z-score formula
    z_score_raw = (sample_probability - predicted_probability_mean) / (std_probability_space + epsilon)
    z_score = abs(z_score_raw)

    return {
        'predicted_probability_mean': predicted_probability_mean,
        'std_probability_space': std_probability_space,
        'sample_probability': sample_probability,
        'z_score_raw': z_score_raw,
        'z_score': z_score,
    }


###########################################
# 1. Dataset for Pickle Files (Classification)
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
        source = data_dict['rgb']  # shape: (H, W, 3) - This is what we want to display
        seg = data_dict['mask']    # shape: (H, W, 3)
        x = np.concatenate([source, seg], axis=2)  # shape: (H, W, 6)
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1) # Permute for model input (C, H, W)
        
        if self.transform:
            x = self.transform(x)

        label = torch.tensor([label], dtype=torch.float32)
        return {'input': x, 'label': label, 'filename': os.path.basename(filepath), 'rgb_image': source}


###########################################
# 2. Main Evaluation Script
###########################################
def main():
    # Hyperparameters and paths
    batch_size = 1 # Use batch size 1 for individual sample evaluation
    
    # Set directories (adjust these paths as necessary)
    val_smoke_dir   = '/home/khanm/workfolder/UAST/Train_data/test/smoke/'
    val_nosmoke_dir = '/home/khanm/workfolder/UAST/Train_data/test/nosmoke/'
    
    # Directory where your trained model checkpoints are saved
    checkpoint_dir = 'experiments/UAST_classification/' 

    # Find the latest checkpoint
    ckpt_list = natsorted(glob.glob(os.path.join(checkpoint_dir, '*.pth.tar')))
    if not ckpt_list:
        print("Error: No checkpoint found in the specified directory.")
        sys.exit(1)
    
    # Load the latest checkpoint
    latest_checkpoint_path = ckpt_list[-1]
    print(f"Loading model from checkpoint: {latest_checkpoint_path}")

    # Initialize the model using your configuration.
    config = CONFIGS_TM['UASTNet']
    model = UAST_Net.UASTNet(config)
    
    # Load model state dict
    checkpoint = torch.load(latest_checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval() # Set model to evaluation mode

    # Create the dataset and loader.
    transform = transforms.Resize((256, 256)) # Only resizing for consistency, no data augmentation

    val_dataset = PklClassificationDataset(val_smoke_dir, val_nosmoke_dir, transform=transform)
    val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("\n--- Model Evaluation Results ---")
    print("{:<30} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "Sample Name", "True Label", "Pred Label", "Pred Mean", "Pred STD", "MC-Dropout Mean", "Z-Score", "Trust Label"
    ))
    print("-" * 140)

    all_sample_results = []
    
    true_labels_for_cm = []
    predicted_probs_for_cm = []
    std_preds_for_hist = [] 
    sample_pred_for_cm = []
    z_score_for_cm = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            inputs = batch['input'].cuda()
            labels = batch['label'].cuda()
            filenames = batch['filename']
            rgb_images = batch['rgb_image']

            mean_pred, var_pred, randm_sample_val, _ = model(inputs)
            
            analysis_results = analyze_model_outputs(
                mean_pred_logit=mean_pred.item(),
                std_pred_logit=abs(var_pred.item()),
                sample_logit=randm_sample_val.item() 
            )

            mean_pred_for_dist = analysis_results['predicted_probability_mean'] 
            randm_sample_for_z = analysis_results['sample_probability']    
            std_pred = analysis_results['std_probability_space']
            z_score = analysis_results['z_score']
            z_score_raw = analysis_results['z_score_raw']

            predicted_label = float(mean_pred_for_dist > 0.5) 
            
            if z_score < 2.0:
                predicted_z_label = "High Confidence"
            elif 2.0 <= z_score < 2.25:
                predicted_z_label = "Moderate Confidence"
            elif 2.25 <= z_score < 3.0:
                predicted_z_label = "Low Confidence" # Renamed from "Unlikely" for consistency with the prompt.
            else: # z_score >= 3.0
                predicted_z_label = "Very Low Confidence" # Renamed from "Highly Unlikely" for consistency with the prompt.

            print(f"{filenames[0]:<30} {labels.item():<15.0f} {predicted_label:<15.0f} {mean_pred_for_dist:<15.4f} {std_pred:<15.4f} {randm_sample_for_z:<15.4f} {z_score:<15.4f} {predicted_z_label:<15}")

            all_sample_results.append({
                'filename': filenames[0],
                'true_label': labels.item(),
                'predicted_label': predicted_label,
                'predicted_probability': mean_pred_for_dist,
                'z_score': z_score,
                'trust_label': predicted_z_label,
                'rgb_image': rgb_images[0].cpu().numpy()
            })

            true_labels_for_cm.append(labels.item())
            predicted_probs_for_cm.append(mean_pred_for_dist)
            std_preds_for_hist.append(std_pred) 
            sample_pred_for_cm.append(randm_sample_for_z)
            z_score_for_cm.append(z_score_raw)

    print("-" * 140) 
    print("Evaluation complete.")

    # --- Confusion Matrix Display ---
    print("\n--- Confusion Matrix ---")
    
    binary_predictions_for_cm = [1 if prob > 0.5 else 0 for prob in predicted_probs_for_cm]

    cm = confusion_matrix(true_labels_for_cm, binary_predictions_for_cm)
    
    class_names = ['No Smoke', 'Smoke'] 
    
    # Create a Figure + Axes
    fig, ax = plt.subplots(figsize=(6,6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False, text_kw={'fontsize':16, 'fontweight':'bold'})
    # Tweak tick label size & weight
    ax.tick_params(axis='x', labelsize=14)   # increase x-tick (class) size
    ax.tick_params(axis='y', labelsize=14)   # increase y-tick size

    for lbl in ax.get_xticklabels():
        lbl.set_fontweight('bold')
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')

    # (Optional) bold the axis labels themselves  
    ax.xaxis.label.set_fontsize(16)
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontweight('bold')
    
    plt.title('Confusion Matrix', fontsize=18, fontweight='bold') # Adjusted title slightly
    plt.show()
    
    # --- EVALUATION METRICS ---
    display_classification_metrics(true_labels_for_cm, predicted_probs_for_cm)
    
    # --- Calibration Curve Display ---
    print("\n--- Calibration Curve ---")
    calibration_curve_plotting(true_labels_for_cm, predicted_probs_for_cm, n_bin=10)
    
    # --- UNCERTAINTY HISTOGRAM ---
    print("\n--- Uncertainty Histogram ---")
    plot_uncertainty_hist(std_preds_for_hist)
    
    # --- SCATTER PLOT SECTION ---
    plot_unc_vs_error(true_labels_for_cm, predicted_probs_for_cm, std_preds_for_hist)
    
    # --- UNCERTAINTY BOX PLOT ---
    plot_box_unc(true_labels_for_cm, std_preds_for_hist)
    
    # --- Z-SCORE DISTRIBUTION ---
    plot_plausibility_histogram(z_score_for_cm)
    
    # --- Image Display Section ---
    print("\n--- Displaying Sample Images ---")

    trust_samples = [s for s in all_sample_results if s['trust_label'] == 'High Confidence']
    doubtful_samples = [s for s in all_sample_results if s['trust_label'] != 'High Confidence']

    num_trust_to_sample = min(4, len(trust_samples))
    num_doubtful_to_sample = min(4, len(doubtful_samples))

    images_to_display = random.sample(trust_samples, num_trust_to_sample) + random.sample(doubtful_samples, num_doubtful_to_sample)

    random.shuffle(images_to_display) # Shuffle the combined list to mix them up

    titles = [
        f"{s['filename']}\nTrue: {s['true_label']:.0f}, Pred: {s['predicted_label']:.0f}, Pred prob: {s['predicted_probability']:.4f}\nPlausibility Confidence: {np.exp(-0.5 * s['z_score']**2):.2f}, {s['trust_label']}" 
        for s in images_to_display
    ]

    if not images_to_display:
        print("No images to display based on filtering criteria. Make sure your validation sets contain enough samples for all categories.")
        return

    num_images = len(images_to_display)
    cols = 4 
    rows = math.ceil(num_images / cols) 

    plt.figure(figsize=(cols * 4, rows * 4)) 

    for i, sample_data in enumerate(images_to_display):
        ax = plt.subplot(rows, cols, i + 1)
        img = sample_data['rgb_image']
        # Matplotlib expects image data as (H, W, C) for color images.
        # Ensure the image data is correctly typed (e.g., uint8) if it's 0-255.
        # If it's float 0-1, it's also fine. If it's float > 1, you might need normalization.
        # Assuming your 'rgb_image' is directly plotable as is from the pickle.
        ax.imshow(img) 
        ax.set_title(titles[i], fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle("Randomly Selected Samples by Z-Score Trust Categories", fontsize=16, y=1.02) 
    plt.show()


###########################################
# 3. Script Entry Point and GPU Setup
###########################################
if __name__ == '__main__':
    GPU_iden = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(GPU_iden)
        print('Using GPU:', torch.cuda.get_device_name(GPU_iden))
    else:
        print("CUDA is not available. Running on CPU.")
        torch.device('cpu') 
    main()
