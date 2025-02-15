#!/usr/bin/env python3
import os
import random
import shutil
import argparse

def sample_val_and_test(train_dir, val_dir, test_dir, val_proportion=0.2, test_proportion=0.1, move_files=True, seed=42):
    """
    For each category (e.g., 'smoke' and 'nosmoke') inside train_dir, this function:
      - Lists all .pkl files.
      - Computes the number of files to sample for validation (val_count) and test (test_count)
        based on the original total number of files.
      - Samples val_count files and then (from the remaining files) samples test_count files.
      - Moves (or copies) the sampled files to the corresponding subdirectories in val_dir and test_dir.
      
    Args:
        train_dir (str): Path to the training directory that contains subdirectories 'smoke' and 'nosmoke'.
        val_dir (str): Path to the validation directory. Subdirectories (smoke, nosmoke) will be created if needed.
        test_dir (str): Path to the test directory. Subdirectories (smoke, nosmoke) will be created if needed.
        val_proportion (float): Proportion of files (with respect to the original total) to sample for validation.
        test_proportion (float): Proportion of files (with respect to the original total) to sample for testing.
        move_files (bool): If True, the files will be moved from train_dir; otherwise, they are copied.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)
    categories = ['smoke', 'nosmoke']
    
    for cat in categories:
        train_cat_dir = os.path.join(train_dir, cat)
        val_cat_dir   = os.path.join(val_dir, cat)
        test_cat_dir  = os.path.join(test_dir, cat)
        os.makedirs(val_cat_dir, exist_ok=True)
        os.makedirs(test_cat_dir, exist_ok=True)
        
        # List all .pkl files in the current training category directory.
        all_files = [os.path.join(train_cat_dir, f)
                     for f in os.listdir(train_cat_dir) if f.endswith('.pkl')]
        total_files = len(all_files)
        if total_files == 0:
            print(f"No .pkl files found in {train_cat_dir}")
            continue
        
        # Compute number of validation and test files (with respect to the original total).
        val_count = int(total_files * val_proportion)
        test_count = int(total_files * test_proportion)
        
        print(f"[{cat}] Total files: {total_files}, Val: {val_count}, Test: {test_count}")
        
        # Randomly sample the validation files.
        val_files = random.sample(all_files, val_count)
        # Remove the validation files from the pool.
        remaining_files = [f for f in all_files if f not in val_files]
        # Randomly sample the test files from the remaining ones.
        if len(remaining_files) < test_count:
            test_count = len(remaining_files)
        test_files = random.sample(remaining_files, test_count)
        
        # Process validation files.
        for file_path in val_files:
            dest_path = os.path.join(val_cat_dir, os.path.basename(file_path))
            if move_files:
                shutil.move(file_path, dest_path)
                print(f"Moved to val: {file_path} -> {dest_path}")
            else:
                shutil.copy(file_path, dest_path)
                print(f"Copied to val: {file_path} -> {dest_path}")
        
        # Process test files.
        for file_path in test_files:
            dest_path = os.path.join(test_cat_dir, os.path.basename(file_path))
            if move_files:
                shutil.move(file_path, dest_path)
                print(f"Moved to test: {file_path} -> {dest_path}")
            else:
                shutil.copy(file_path, dest_path)
                print(f"Copied to test: {file_path} -> {dest_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample a proportion of .pkl files from train (with subfolders 'smoke' and 'nosmoke') into separate validation and test folders."
    )
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Path to the train directory containing 'smoke' and 'nosmoke' subfolders.")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Path to the validation directory where 'smoke' and 'nosmoke' subfolders will be created.")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to the test directory where 'smoke' and 'nosmoke' subfolders will be created.")
    parser.add_argument("--val_proportion", type=float, default=0.2,
                        help="Proportion of files (relative to the original total in each category) to sample for validation.")
    parser.add_argument("--test_proportion", type=float, default=0.1,
                        help="Proportion of files (relative to the original total in each category) to sample for testing.")
    parser.add_argument("--move", action="store_true",
                        help="If set, the files will be moved from train to val/test; otherwise, they are copied.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    sample_val_and_test(args.train_dir, args.val_dir, args.test_dir,
                        val_proportion=args.val_proportion,
                        test_proportion=args.test_proportion,
                        move_files=args.move,
                        seed=args.seed)

if __name__ == "__main__":
    main()
