"""
Multimodal Dataset Creation Script

This script merges visual and NLP features, validates the data, and creates
train/validation/test splits for CreateML.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_features(visual_features_path, nlp_features_path, clip_features_path=None, vllm_features_path=None):
    """
    Load visual, NLP, and optionally CLIP and VLLM features from CSV files.

    Args:
        visual_features_path (str): Path to visual features CSV.
        nlp_features_path (str): Path to NLP features CSV.
        clip_features_path (str, optional): Path to CLIP features CSV.
        vllm_features_path (str, optional): Path to VLLM features CSV.

    Returns:
        tuple: (visual_df, nlp_df, clip_df, vllm_df) where clip_df and vllm_df may be None
    """
    print("Loading visual features...")
    visual_df = pd.read_csv(visual_features_path)
    print(f"Loaded {len(visual_df)} visual feature vectors (2048-dim)")

    print("Loading NLP features...")
    nlp_df = pd.read_csv(nlp_features_path)
    print(f"Loaded {len(nlp_df)} NLP feature vectors (406-dim)")

    clip_df = None
    if clip_features_path and os.path.exists(clip_features_path):
        print("Loading CLIP features...")
        clip_df = pd.read_csv(clip_features_path)
        print(f"Loaded {len(clip_df)} CLIP alignment scores (1-dim)")
    else:
        print("CLIP features not found - skipping")

    vllm_df = None
    if vllm_features_path and os.path.exists(vllm_features_path):
        print("Loading VLLM features...")
        vllm_df = pd.read_csv(vllm_features_path)
        print(f"Loaded {len(vllm_df)} VLLM incongruence features (769-dim)")
    else:
        print("VLLM features not found - skipping")

    return visual_df, nlp_df, clip_df, vllm_df


def merge_features(visual_df, nlp_df, clip_df=None, vllm_df=None):
    """
    Merge visual, NLP, and optionally CLIP and VLLM features on video_id.

    Args:
        visual_df (pd.DataFrame): Visual features DataFrame.
        nlp_df (pd.DataFrame): NLP features DataFrame.
        clip_df (pd.DataFrame, optional): CLIP features DataFrame.
        vllm_df (pd.DataFrame, optional): VLLM features DataFrame.

    Returns:
        merged_df (pd.DataFrame): Merged DataFrame with all features.
    """
    print("\nMerging features on video_id...")

    # Remove label column from individual feature sets (we'll use one consistent label)
    visual_no_label = visual_df.drop(columns=['label'])
    nlp_no_label = nlp_df.drop(columns=['label'])

    # Start with visual and NLP
    merged_df = pd.merge(visual_no_label, nlp_no_label, on='video_id', how='inner')
    print(f"  Merged visual + NLP: {len(merged_df)} videos")

    # Merge CLIP if available
    if clip_df is not None:
        clip_no_label = clip_df.drop(columns=['label'])
        merged_df = pd.merge(merged_df, clip_no_label, on='video_id', how='inner')
        print(f"  Added CLIP features: {len(merged_df)} videos remaining")

    # Merge VLLM if available
    if vllm_df is not None:
        vllm_no_label = vllm_df.drop(columns=['label'])
        merged_df = pd.merge(merged_df, vllm_no_label, on='video_id', how='inner')
        print(f"  Added VLLM features: {len(merged_df)} videos remaining")

    # Add label back from visual_df (all should have the same labels)
    label_map = dict(zip(visual_df['video_id'], visual_df['label']))
    merged_df['label'] = merged_df['video_id'].map(label_map)

    print(f"\nFinal merged dataset: {len(merged_df)} videos with {len(merged_df.columns)-2} features")

    # Check for videos that didn't match
    all_sources = [visual_df, nlp_df]
    source_names = ['visual', 'NLP']

    if clip_df is not None:
        all_sources.append(clip_df)
        source_names.append('CLIP')

    if vllm_df is not None:
        all_sources.append(vllm_df)
        source_names.append('VLLM')

    final_ids = set(merged_df['video_id'])
    for source_df, name in zip(all_sources, source_names):
        missing = set(source_df['video_id']) - final_ids
        if missing:
            print(f"  Note: {len(missing)} videos from {name} not in final dataset")

    return merged_df


def validate_dataset(df):
    """
    Validate the merged dataset for missing values, duplicates, and class balance.

    Args:
        df (pd.DataFrame): Merged dataset DataFrame.

    Returns:
        is_valid (bool): True if dataset passes all validation checks.
    """
    print("\nValidating dataset...")
    is_valid = True

    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: Found {missing_values} missing values")
        print("Columns with missing values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        is_valid = False
    else:
        print("✓ No missing values")

    # Check for duplicate video_ids
    duplicates = df['video_id'].duplicated().sum()
    if duplicates > 0:
        print(f"Warning: Found {duplicates} duplicate video IDs")
        is_valid = False
    else:
        print("✓ No duplicate video IDs")

    # Check label values
    unique_labels = df['label'].unique()
    if not set(unique_labels).issubset({0, 1}):
        print(f"Warning: Labels contain unexpected values: {unique_labels}")
        is_valid = False
    else:
        print(f"✓ Labels are valid: {sorted(unique_labels)}")

    # Check class balance
    label_counts = df['label'].value_counts().sort_index()
    print("\nClass distribution:")
    print(label_counts)
    print(f"Balance ratio: {label_counts.min() / label_counts.max():.2f}")

    # Check feature ranges
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'label']

    inf_count = np.isinf(df[numeric_cols].values).sum()
    if inf_count > 0:
        print(f"Warning: Found {inf_count} infinite values")
        is_valid = False
    else:
        print("✓ No infinite values")

    return is_valid


def create_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Create stratified train/validation/test splits.

    Args:
        df (pd.DataFrame): Merged dataset DataFrame.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_df (pd.DataFrame): Training set.
        val_df (pd.DataFrame): Validation set.
        test_df (pd.DataFrame): Test set.
    """
    print(f"\nCreating splits: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test...")

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df['label'],
        random_state=random_state
    )

    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_adjusted,
        stratify=temp_df['label'],
        random_state=random_state
    )

    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # Verify no overlap
    train_ids = set(train_df['video_id'])
    val_ids = set(val_df['video_id'])
    test_ids = set(test_df['video_id'])

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        print("Warning: Overlap detected between splits!")
    else:
        print("✓ No overlap between splits")

    # Check class balance in each split
    print("\nClass distribution in each split:")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split_df['label'].value_counts().sort_index()
        print(f"{name}: {dict(counts)}")

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir):
    """
    Save train/validation/test splits to CSV files.

    Args:
        train_df (pd.DataFrame): Training set.
        val_df (pd.DataFrame): Validation set.
        test_df (pd.DataFrame): Test set.
        output_dir (str): Directory to save CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'validation.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    print(f"\nSaving splits to {output_dir}...")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✓ Saved train.csv ({len(train_df)} rows)")
    print(f"✓ Saved validation.csv ({len(val_df)} rows)")
    print(f"✓ Saved test.csv ({len(test_df)} rows)")


def save_summary(df, train_df, val_df, test_df, output_dir, has_clip, has_vllm):
    """
    Save dataset summary to JSON file.

    Args:
        df (pd.DataFrame): Full merged dataset.
        train_df (pd.DataFrame): Training set.
        val_df (pd.DataFrame): Validation set.
        test_df (pd.DataFrame): Test set.
        output_dir (str): Directory to save summary file.
        has_clip (bool): Whether CLIP features are included.
        has_vllm (bool): Whether VLLM features are included.
    """
    feature_breakdown = {
        "visual_features_vcnn": 2048,
        "nlp_sentence_embeddings": 384,
        "nlp_statistical_features": 15,
        "nlp_sentiment_features": 3,
        "nlp_complexity_features": 4
    }

    if has_clip:
        feature_breakdown["clip_alignment_sclip"] = 1

    if has_vllm:
        feature_breakdown["vllm_embeddings_tvllm"] = 768
        feature_breakdown["vllm_incongruence_score"] = 1

    summary = {
        "total_samples": len(df),
        "num_features": len(df.columns) - 2,  # Exclude video_id and label
        "feature_breakdown": feature_breakdown,
        "gemini_features_included": {
            "clip_alignment": has_clip,
            "vllm_incongruence": has_vllm
        },
        "splits": {
            "train": {"count": len(train_df), "label_0": int((train_df['label'] == 0).sum()), "label_1": int((train_df['label'] == 1).sum())},
            "validation": {"count": len(val_df), "label_0": int((val_df['label'] == 0).sum()), "label_1": int((val_df['label'] == 1).sum())},
            "test": {"count": len(test_df), "label_0": int((test_df['label'] == 0).sum()), "label_1": int((test_df['label'] == 1).sum())}
        },
        "class_distribution": {
            "label_0_count": int((df['label'] == 0).sum()),
            "label_1_count": int((df['label'] == 1).sum()),
            "balance_ratio": float((df['label'] == 0).sum() / (df['label'] == 1).sum())
        }
    }

    summary_path = os.path.join(output_dir, 'dataset_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved dataset_summary.json")


def main():
    """Main function to create multimodal dataset."""
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(base_dir, 'features')
    output_dir = os.path.join(base_dir, 'datasets')

    visual_features_path = os.path.join(features_dir, 'visual_features.csv')
    nlp_features_path = os.path.join(features_dir, 'nlp_features.csv')
    clip_features_path = os.path.join(features_dir, 'clip_features.csv')
    vllm_features_path = os.path.join(features_dir, 'vllm_features.csv')

    # Check if required feature files exist
    if not os.path.exists(visual_features_path):
        print(f"Error: Visual features file not found at {visual_features_path}")
        print("Please run extract_visual_features.py first")
        return

    if not os.path.exists(nlp_features_path):
        print(f"Error: NLP features file not found at {nlp_features_path}")
        print("Please run extract_nlp_features.py first")
        return

    print("="*60)
    print("Multimodal Dataset Creation")
    print("="*60)

    # Load features (CLIP and VLLM are optional)
    visual_df, nlp_df, clip_df, vllm_df = load_features(
        visual_features_path,
        nlp_features_path,
        clip_features_path,
        vllm_features_path
    )

    # Track which Gemini features are included
    has_clip = clip_df is not None
    has_vllm = vllm_df is not None

    print(f"\nFeature Set Summary:")
    print(f"  ✓ Visual (Vcnn): 2048 features")
    print(f"  ✓ NLP: 406 features")
    print(f"  {'✓' if has_clip else '✗'} CLIP (Sclip): {1 if has_clip else 0} feature")
    print(f"  {'✓' if has_vllm else '✗'} VLLM (Tvllm + Incongruence): {769 if has_vllm else 0} features")

    total_features = 2048 + 406
    if has_clip:
        total_features += 1
    if has_vllm:
        total_features += 769

    print(f"  Total: {total_features} features")

    # Merge features
    merged_df = merge_features(visual_df, nlp_df, clip_df, vllm_df)

    # Validate dataset
    is_valid = validate_dataset(merged_df)

    if not is_valid:
        print("\nWarning: Dataset validation failed. Proceeding anyway, but please review warnings above.")

    # Create splits
    train_df, val_df, test_df = create_splits(merged_df)

    # Save splits
    save_splits(train_df, val_df, test_df, output_dir)

    # Save summary
    save_summary(merged_df, train_df, val_df, test_df, output_dir, has_clip, has_vllm)

    print("\n" + "="*60)
    print("SUCCESS! Multimodal dataset created successfully")
    print("="*60)
    print(f"\nDataset files are ready in: {output_dir}")
    print(f"Feature configuration:")
    print(f"  - Base features (Vcnn + NLP): 2,454")
    if has_clip:
        print(f"  - CLIP alignment (Sclip): +1")
    if has_vllm:
        print(f"  - VLLM incongruence (Tvllm + score): +769")
    print(f"  - Total: {total_features} features")

    print("\nNext steps:")
    print("1. Run: python createml_dataset_validator.py")
    print("2. Run: python analyze_features.py (optional)")
    print("3. Import datasets/train.csv to CreateML Tabular Classifier")

    if not has_clip:
        print("\nNote: CLIP features not included. To add them:")
        print("  - Run: python extract_clip_features.py")
        print("  - Then re-run this script")

    if not has_vllm:
        print("\nNote: VLLM features not included. To add them:")
        print("  - Configure ANTHROPIC_API_KEY in .env file")
        print("  - Run: python extract_vllm_features_claude.py (~1-2 hours, ~$13 cost)")
        print("  - Then re-run this script")


if __name__ == "__main__":
    main()
