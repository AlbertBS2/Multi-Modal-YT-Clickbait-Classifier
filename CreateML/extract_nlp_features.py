"""
NLP Feature Extraction Script

This script extracts NLP features from all video transcripts and saves them to a CSV file.
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from nlp_feature_extraction import extract_all_nlp_features


def extract_nlp_features_batch(transcript_files, labels, model, batch_size=64):
    """
    Extract NLP features from all transcripts in batch mode.

    Args:
        transcript_files (dict): Dictionary mapping labels to transcript TSV file paths.
        labels (list): List of label values (0 for NMTV, 1 for MTV).
        model (SentenceTransformer): Pre-trained sentence transformer model.
        batch_size (int): Number of transcripts to process in each batch.

    Returns:
        df (pd.DataFrame): DataFrame with columns: video_id, sent_emb_0, ..., sent_emb_383,
                           transcript_length, word_count, ..., label
    """
    all_features = []
    all_video_ids = []
    all_labels = []
    failed_transcripts = []

    # Process each label category
    for label in labels:
        transcript_file = transcript_files[label]

        print(f"\nLoading transcripts for label {label} from {transcript_file}...")
        df = pd.read_csv(transcript_file, sep='\t')

        print(f"Processing {len(df)} transcripts...")

        # Process in batches
        for i in tqdm(range(0, len(df), batch_size), desc=f"Label {label}"):
            batch = df.iloc[i:i+batch_size]

            for _, row in batch.iterrows():
                video_id = row['video_id']
                transcript = row['transcript']

                try:
                    # Extract all NLP features
                    features = extract_all_nlp_features(transcript, model)
                    all_features.append(features)
                    all_video_ids.append(video_id)
                    all_labels.append(label)

                except Exception as e:
                    failed_transcripts.append((video_id, str(e)))
                    print(f"\nWarning: Failed to process transcript for {video_id}: {e}")
                    continue

    # Create DataFrame
    print("\nCreating DataFrame...")

    # Column names: 384 embeddings + 15 statistical + 3 sentiment + 4 complexity
    embedding_cols = [f"sent_emb_{i}" for i in range(384)]
    statistical_cols = [
        'transcript_length', 'word_count', 'avg_word_length', 'sentence_count',
        'avg_sentence_length', 'exclamation_count', 'question_count',
        'uppercase_word_ratio', 'capitalized_word_ratio', 'number_count',
        'unique_word_ratio', 'stopword_ratio', 'punctuation_density',
        'all_caps_ratio', 'ellipsis_count'
    ]
    sentiment_cols = ['sentiment_polarity', 'sentiment_subjectivity', 'sentiment_intensity']
    complexity_cols = ['lexical_density', 'avg_syllables_per_word', 'flesch_reading_ease', 'automated_readability_index']

    all_columns = embedding_cols + statistical_cols + sentiment_cols + complexity_cols

    df = pd.DataFrame(all_features, columns=all_columns)
    df.insert(0, 'video_id', all_video_ids)
    df['label'] = all_labels

    # Report failed transcripts
    if failed_transcripts:
        print(f"\nWarning: Failed to process {len(failed_transcripts)} transcripts:")
        for video_id, error in failed_transcripts[:10]:  # Show first 10
            print(f"  - {video_id}: {error}")
        if len(failed_transcripts) > 10:
            print(f"  ... and {len(failed_transcripts) - 10} more")

    return df


def main():
    """Main function to extract NLP features and save to CSV."""
    # Define paths (relative to CreateML directory)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), 'data', 'ThumbnailTruthData')

    mtv_transcripts = os.path.join(data_dir, 'MTV_transcripts.tsv')
    nmtv_transcripts = os.path.join(data_dir, 'NMTV_transcripts.tsv')

    output_dir = os.path.join(base_dir, 'features')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'nlp_features.csv')

    # Load sentence transformer model
    print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
    print("Note: First run will download the model (~80MB)")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Set up transcript files and labels
    transcript_files = {
        0: nmtv_transcripts,  # NMTV = label 0
        1: mtv_transcripts    # MTV = label 1
    }

    labels = [0, 1]

    # Extract features
    df = extract_nlp_features_batch(transcript_files, labels, model, batch_size=64)

    # Save to CSV
    print(f"\nSaving features to {output_file}...")
    df.to_csv(output_file, index=False)

    print(f"\nSuccess! Extracted NLP features for {len(df)} videos")
    print(f"Output shape: {df.shape}")
    print(f"Columns: {list(df.columns[:5])} ... {list(df.columns[-5:])}")

    # Display summary statistics
    print("\nLabel distribution:")
    print(df['label'].value_counts().sort_index())

    # Show sample statistics
    print("\nSample feature statistics:")
    print(df[['transcript_length', 'word_count', 'sentence_count', 'sentiment_polarity']].describe())


if __name__ == "__main__":
    main()
