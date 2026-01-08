"""
Visual Feature Extraction Script

This script extracts ResNet50 visual features from all thumbnail images and saves them to a CSV file.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# Add parent directory to path to import v_cnn_extraction
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v_cnn_extraction import create_feature_extractor, preprocess_image, get_v_cnn


def extract_visual_features_batch(thumbnail_dirs, video_ids, labels, batch_size=32):
    """
    Extract ResNet50 visual features from all thumbnails in batch mode.

    Args:
        thumbnail_dirs (dict): Dictionary mapping labels to thumbnail directory paths.
                               e.g., {0: 'path/to/NMTV_Thumbnails', 1: 'path/to/MTV_Thumbnails'}
        video_ids (dict): Dictionary mapping labels to lists of video IDs.
        labels (list): List of label values (0 for NMTV, 1 for MTV).
        batch_size (int): Number of images to process in each batch.

    Returns:
        df (pd.DataFrame): DataFrame with columns: video_id, v_cnn_0, v_cnn_1, ..., v_cnn_2047, label
    """
    # Initialize ResNet50
    print("Loading ResNet50 model...")
    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights)
    feature_extractor = create_feature_extractor(resnet)
    feature_extractor.eval()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    feature_extractor = feature_extractor.to(device)

    all_features = []
    all_video_ids = []
    all_labels = []
    failed_images = []

    # Process each label category
    for label in labels:
        thumbnail_dir = thumbnail_dirs[label]
        video_id_list = video_ids[label]

        print(f"\nProcessing label {label} ({len(video_id_list)} images) from {thumbnail_dir}...")

        # Process in batches
        for i in tqdm(range(0, len(video_id_list), batch_size), desc=f"Label {label}"):
            batch_video_ids = video_id_list[i:i+batch_size]
            batch_tensors = []
            batch_valid_ids = []

            # Load and preprocess batch
            for video_id in batch_video_ids:
                image_path = os.path.join(thumbnail_dir, f"{video_id}.jpg")

                try:
                    # Preprocess image
                    img_tensor = preprocess_image(image_path, weights)
                    batch_tensors.append(img_tensor)
                    batch_valid_ids.append(video_id)
                except Exception as e:
                    failed_images.append((video_id, str(e)))
                    print(f"\nWarning: Failed to process {image_path}: {e}")
                    continue

            if len(batch_tensors) == 0:
                continue

            # Stack tensors into batch
            batch = torch.cat(batch_tensors, dim=0).to(device)

            # Extract features for batch
            with torch.no_grad():
                features = feature_extractor(batch)
                features = features.view(features.size(0), -1)  # Flatten

            # Convert to numpy and store
            features_np = features.cpu().numpy()

            for j, video_id in enumerate(batch_valid_ids):
                all_features.append(features_np[j])
                all_video_ids.append(video_id)
                all_labels.append(label)

    # Create DataFrame
    print("\nCreating DataFrame...")
    feature_columns = [f"v_cnn_{i}" for i in range(2048)]
    df = pd.DataFrame(all_features, columns=feature_columns)
    df.insert(0, 'video_id', all_video_ids)
    df['label'] = all_labels

    # Report failed images
    if failed_images:
        print(f"\nWarning: Failed to process {len(failed_images)} images:")
        for video_id, error in failed_images[:10]:  # Show first 10
            print(f"  - {video_id}: {error}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")

    return df


def main():
    """Main function to extract visual features and save to CSV."""
    # Define paths (relative to CreateML directory)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), 'data', 'ThumbnailTruthData')

    mtv_csv = os.path.join(data_dir, 'mtv_cleaned.csv')
    nmtv_csv = os.path.join(data_dir, 'nmtv_cleaned.csv')

    mtv_thumbnails = os.path.join(data_dir, 'MTV_Thumbnails')
    nmtv_thumbnails = os.path.join(data_dir, 'NMTV_Thumbnails')

    output_dir = os.path.join(base_dir, 'features')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'visual_features.csv')

    # Load video IDs
    print("Loading video IDs from cleaned CSV files...")
    mtv_df = pd.read_csv(mtv_csv)
    nmtv_df = pd.read_csv(nmtv_csv)

    mtv_video_ids = mtv_df['video_id'].tolist()
    nmtv_video_ids = nmtv_df['video_id'].tolist()

    print(f"Found {len(mtv_video_ids)} MTV videos and {len(nmtv_video_ids)} NMTV videos")

    # Set up directories and labels
    thumbnail_dirs = {
        0: nmtv_thumbnails,  # NMTV = label 0
        1: mtv_thumbnails    # MTV = label 1
    }

    video_ids = {
        0: nmtv_video_ids,
        1: mtv_video_ids
    }

    labels = [0, 1]

    # Extract features
    df = extract_visual_features_batch(thumbnail_dirs, video_ids, labels, batch_size=32)

    # Save to CSV
    print(f"\nSaving features to {output_file}...")
    df.to_csv(output_file, index=False)

    print(f"\nSuccess! Extracted features for {len(df)} videos")
    print(f"Output shape: {df.shape}")
    print(f"Columns: {list(df.columns[:5])} ... {list(df.columns[-5:])}")

    # Display summary statistics
    print("\nLabel distribution:")
    print(df['label'].value_counts().sort_index())


if __name__ == "__main__":
    main()
