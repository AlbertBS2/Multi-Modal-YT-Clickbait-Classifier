"""
CLIP Feature Extraction Script

This script extracts semantic alignment scores between thumbnails and transcripts using CLIP.
The alignment score (Sclip) measures how well the thumbnail matches the video content.
"""

import os
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


def extract_clip_similarity(image_path, transcript, model, processor, device):
    """
    Calculate semantic alignment between thumbnail and transcript using CLIP.

    Args:
        image_path (str): Path to thumbnail image.
        transcript (str): Video transcript text.
        model: CLIP model.
        processor: CLIP processor.
        device: Torch device.

    Returns:
        float: Sclip score (-1 to 1, higher = better alignment)
    """
    image = Image.open(image_path).convert('RGB')

    # Truncate transcript to avoid exceeding CLIP's token limit
    # CLIP has a max of 77 tokens, roughly ~500 characters
    transcript_truncated = transcript[:500] if len(transcript) > 500 else transcript

    inputs = processor(
        text=[transcript_truncated],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Calculate cosine similarity between image and text embeddings
        similarity = torch.nn.functional.cosine_similarity(
            outputs.image_embeds,
            outputs.text_embeds
        )

    return similarity.item()


def process_videos(video_data, thumbnail_dir, model, processor, device, label):
    """
    Process a batch of videos to extract CLIP features.

    Args:
        video_data (pd.DataFrame): DataFrame with video_id and transcript columns.
        thumbnail_dir (str): Directory containing thumbnail images.
        model: CLIP model.
        processor: CLIP processor.
        device: Torch device.
        label (int): Label for these videos (0 or 1).

    Returns:
        list: List of dictionaries with video_id, clip_similarity, and label.
    """
    results = []
    failed_count = 0

    for _, row in tqdm(video_data.iterrows(), total=len(video_data), desc=f"Label {label}"):
        video_id = row['video_id']
        transcript = row['transcript']
        thumb_path = os.path.join(thumbnail_dir, f"{video_id}.jpg")

        try:
            sclip = extract_clip_similarity(thumb_path, transcript, model, processor, device)
            results.append({
                'video_id': video_id,
                'clip_similarity': sclip,
                'label': label
            })
        except Exception as e:
            failed_count += 1
            if failed_count <= 10:  # Show first 10 errors
                print(f"\nWarning: Failed for {video_id}: {e}")

    if failed_count > 10:
        print(f"\n... and {failed_count - 10} more failures")

    return results


def main():
    """Main function to extract CLIP features and save to CSV."""
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), 'data', 'ThumbnailTruthData')

    mtv_transcripts_path = os.path.join(data_dir, 'MTV_transcripts.tsv')
    nmtv_transcripts_path = os.path.join(data_dir, 'NMTV_transcripts.tsv')

    mtv_thumb_dir = os.path.join(data_dir, 'MTV_Thumbnails')
    nmtv_thumb_dir = os.path.join(data_dir, 'NMTV_Thumbnails')

    output_dir = os.path.join(base_dir, 'features')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'clip_features.csv')

    # Load CLIP model
    print("Loading CLIP model (openai/clip-vit-base-patch32)...")
    print("Note: First run will download the model (~600MB)")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    # Load transcripts
    print("\nLoading transcripts...")
    mtv_transcripts = pd.read_csv(mtv_transcripts_path, sep='\t')
    nmtv_transcripts = pd.read_csv(nmtv_transcripts_path, sep='\t')

    print(f"Found {len(mtv_transcripts)} MTV transcripts")
    print(f"Found {len(nmtv_transcripts)} NMTV transcripts")

    # Process videos
    all_results = []

    print("\nProcessing MTV videos (clickbait)...")
    mtv_results = process_videos(mtv_transcripts, mtv_thumb_dir, model, processor, device, label=1)
    all_results.extend(mtv_results)

    print("\nProcessing NMTV videos (non-clickbait)...")
    nmtv_results = process_videos(nmtv_transcripts, nmtv_thumb_dir, model, processor, device, label=0)
    all_results.extend(nmtv_results)

    # Create DataFrame and save
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print("SUCCESS! CLIP features extracted")
    print('='*60)
    print(f"\nSaved to: {output_file}")
    print(f"Total videos processed: {len(df)}")
    print(f"Output shape: {df.shape}")

    # Statistics
    print("\nCLIP Similarity Statistics:")
    print(f"  Range: [{df['clip_similarity'].min():.3f}, {df['clip_similarity'].max():.3f}]")
    print(f"  Mean: {df['clip_similarity'].mean():.3f}")
    print(f"  Std: {df['clip_similarity'].std():.3f}")

    print("\nBy Label:")
    for label in [0, 1]:
        subset = df[df['label'] == label]
        label_name = "Non-Clickbait" if label == 0 else "Clickbait"
        print(f"  {label_name}: mean={subset['clip_similarity'].mean():.3f}, std={subset['clip_similarity'].std():.3f}")

    print("\nInterpretation:")
    print("  - Higher scores (closer to 1): Thumbnail aligns with transcript content")
    print("  - Lower scores (closer to -1): Thumbnail mismatches transcript content")
    print("  - Hypothesis: Clickbait videos should have LOWER alignment scores")


if __name__ == "__main__":
    main()