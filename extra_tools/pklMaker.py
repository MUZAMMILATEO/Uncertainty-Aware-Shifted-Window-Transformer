#!/usr/bin/env python3
import os
import cv2
import pickle
import numpy as np
import argparse

def process_images(rgb_folder, mask_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop over all files in the RGB image folder
    for filename in os.listdir(rgb_folder):
        rgb_path = os.path.join(rgb_folder, filename)
        mask_path = os.path.join(mask_folder, filename)
        
        # Check if the corresponding mask exists
        if not os.path.exists(mask_path):
            print(f"Warning: No mask found for {filename}. Skipping this file.")
            continue

        # Read the RGB image (OpenCV reads images in BGR by default)
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_img is None:
            print(f"Error reading RGB image {rgb_path}. Skipping.")
            continue

        # Read the segmentation mask as a three-channel image
        mask_img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask_img is None:
            print(f"Error reading mask image {mask_path}. Skipping.")
            continue

        # Convert images from BGR to RGB
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        
        # Normalize images to the [0, 1] range as float32
        rgb_normalized = rgb_img.astype(np.float32) / 255.0
        mask_normalized = mask_img.astype(np.float32) / 255.0

        # Create a dictionary to hold the normalized images
        data_pair = {'rgb': rgb_normalized, 'mask': mask_normalized}

        # Define the output pickle file path, using the same base name
        base_name = os.path.splitext(filename)[0]
        pkl_filename = base_name + '.pkl'
        pkl_path = os.path.join(output_folder, pkl_filename)

        # Save the data pair into a pickle file
        with open(pkl_path, 'wb') as f:
            pickle.dump(data_pair, f)

        print(f"Saved normalized pair for {filename} to {pkl_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare pickle files containing normalized RGB and segmented images for deep learning training."
    )
    parser.add_argument(
        "--rgb_folder",
        type=str,
        required=True,
        help="Path to the folder containing RGB images."
    )
    parser.add_argument(
        "--mask_folder",
        type=str,
        required=True,
        help="Path to the folder containing segmentation masks."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder where pickle files will be saved."
    )
    
    args = parser.parse_args()
    process_images(args.rgb_folder, args.mask_folder, args.output_folder)

if __name__ == "__main__":
    main()
