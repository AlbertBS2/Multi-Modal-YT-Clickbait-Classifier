"""
VLLM Feature Extraction using Claude API (Vision)

This script uses Claude's vision capabilities to:
1. Generate descriptions of what thumbnails actually show
2. Compare descriptions to actual video transcripts
3. Calculate incongruence score (mismatch between thumbnail promise and content delivery)

Uses Anthropic's Claude API with vision (no GPU required).

Requirements:
- Anthropic API key (set as ANTHROPIC_API_KEY environment variable or in .env file)
- pip install anthropic python-dotenv

Output: Tvllm (768-dim BERT embeddings) + Incongruence score (1-dim)
"""

import os
import pandas as pd
import numpy as np
import base64
import time
from pathlib import Path
from anthropic import Anthropic
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def encode_image_base64(image_path):
    """Encode image to base64 for Claude API."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def describe_thumbnail_with_claude(image_path, client):
    """
    Use Claude's vision API to describe what the thumbnail shows.

    Args:
        image_path (str): Path to thumbnail image.
        client (Anthropic): Anthropic API client.

    Returns:
        str: Description of what's shown in the thumbnail.
    """
    # Encode image
    image_data = encode_image_base64(image_path)

    # Determine image type
    ext = Path(image_path).suffix.lower()
    media_type = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"

    # Call Claude API with vision
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",  # Claude 3.5 Sonnet has vision
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe what you see in this YouTube thumbnail in one detailed sentence. Focus on the main subject, their expression or action, and any visible text. Be objective and specific."
                    }
                ],
            }
        ],
    )

    return message.content[0].text


def calculate_incongruence(claude_description, transcript, bert_model, bert_tokenizer, device):
    """
    Calculate incongruence by checking the thumbnail description against
    ALL chunks of the transcript and keeping the best match.
    """
    # 1. Encode the Thumbnail Description (The "Promise")
    inputs_desc = bert_tokenizer(claude_description, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        desc_output = bert_model(**inputs_desc)
        # Standard mean pooling for the short description
        tvllm_embedding = desc_output.last_hidden_state.mean(dim=1).squeeze()

    # 2. Chunk the Transcript (The "Delivery")
    words = transcript.split()
    chunk_size = 300  # BERT limit is 512, 300 is a safe "paragraph" size
    stride = 150  # 50% overlap to ensure no context is lost

    chunks = [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), stride)]

    # 3. Compare Description to every chunk
    max_similarity = -1.0

    for chunk in chunks:
        inputs_trans = bert_tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(device)

        with torch.no_grad():
            trans_output = bert_model(**inputs_trans)
            chunk_embedding = trans_output.last_hidden_state.mean(dim=1).squeeze()

            # Calculate similarity for THIS chunk
            sim = torch.nn.functional.cosine_similarity(
                tvllm_embedding.unsqueeze(0),
                chunk_embedding.unsqueeze(0)
            ).item()

            # Keep the BEST match found in the whole video
            if sim > max_similarity:
                max_similarity = sim

    # Incongruence = 1 - (Best Match Found)
    incongruence_score = 1 - max_similarity

    return incongruence_score, tvllm_embedding.cpu().numpy()

def process_videos(video_data, thumbnail_dir, claude_client, bert_model, bert_tokenizer, device, label):
    """
    Process a batch of videos to extract Claude-based VLLM incongruence features.

    Args:
        video_data (pd.DataFrame): DataFrame with video_id and transcript columns.
        thumbnail_dir (str): Directory containing thumbnail images.
        claude_client: Anthropic API client.
        bert_model: BERT model.
        bert_tokenizer: BERT tokenizer.
        device: Torch device.
        label (int): Label for these videos (0 or 1).

    Returns:
        list: List of dictionaries with features.
    """
    results = []
    failed_count = 0
    rate_limit_delay = 0.5  # Delay between API calls to avoid rate limits

    for idx, row in tqdm(video_data.iterrows(), total=len(video_data), desc=f"Label {label}"):
        video_id = row['video_id']
        transcript = row['transcript']
        thumb_path = os.path.join(thumbnail_dir, f"{video_id}.jpg")

        try:
            # Generate thumbnail description using Claude
            claude_desc = describe_thumbnail_with_claude(thumb_path, claude_client)

            # Small delay to respect rate limits
            time.sleep(rate_limit_delay)

            # Calculate incongruence and get Tvllm embedding
            incong_score, tvllm_emb = calculate_incongruence(
                claude_desc, transcript, bert_model, bert_tokenizer, device
            )

            # Store results
            result = {
                'video_id': video_id,
                'vllm_description': claude_desc,  # Store for inspection
                'incongruence_score': incong_score,
                'label': label
            }

            # Add Tvllm embedding columns (768 dimensions)
            for i, val in enumerate(tvllm_emb):
                result[f'tvllm_{i}'] = float(val)

            results.append(result)

            # Progress update every 50 videos
            if (idx + 1) % 50 == 0:
                print(f"\n  Progress: {idx + 1}/{len(video_data)} videos processed")
                print(f"  Latest incongruence: {incong_score:.3f}")
                print(f"  Latest description: {claude_desc[:100]}...")

        except Exception as e:
            failed_count += 1
            if failed_count <= 10:
                print(f"\nWarning: Failed for {video_id}: {e}")

            # If rate limited, increase delay
            if "rate_limit" in str(e).lower():
                rate_limit_delay = min(rate_limit_delay * 2, 5.0)
                print(f"  Increasing delay to {rate_limit_delay}s")
                time.sleep(rate_limit_delay)

    if failed_count > 10:
        print(f"\n... and {failed_count - 10} more failures")

    return results


def main():
    """Main function to extract VLLM incongruence features using Claude API."""
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), 'data', 'ThumbnailTruthData')

    mtv_transcripts_path = os.path.join(data_dir, 'MTV_transcripts.tsv')
    nmtv_transcripts_path = os.path.join(data_dir, 'NMTV_transcripts.tsv')

    mtv_thumb_dir = os.path.join(data_dir, 'MTV_Thumbnails')
    nmtv_thumb_dir = os.path.join(data_dir, 'NMTV_Thumbnails')

    output_dir = os.path.join(base_dir, 'features')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'vllm_features.csv')
    descriptions_file = os.path.join(output_dir, 'vllm_descriptions.csv')

    print("="*60)
    print("VLLM Feature Extraction (Claude API)")
    print("="*60)

    # Check cache - if output exists, reuse it
    if os.path.exists(output_file):
        print("\n✓ Found existing VLLM features, reusing cached results")
        print(f"   {output_file}")
        return True

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n✗ Error: ANTHROPIC_API_KEY not found")
        print("\nPlease configure your API key:")
        print("  1. Copy .env.example to .env in project root")
        print("  2. Add your API key to .env file")
        print("  3. Get API key from: https://console.anthropic.com/settings/keys")
        print("\nAlternatively, set environment variable:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        return False

    # Initialize Claude client
    print("\nInitializing Claude API client...")
    claude_client = Anthropic(api_key=api_key)
    print("✓ Claude client ready")

    # Load BERT model for embeddings
    print("\nLoading BERT model for embeddings...")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)
    bert_model.eval()
    print(f"✓ BERT model loaded (device: {device})")

    # Load transcripts
    print("\nLoading transcripts...")
    mtv_transcripts = pd.read_csv(mtv_transcripts_path, sep='\t')
    nmtv_transcripts = pd.read_csv(nmtv_transcripts_path, sep='\t')

    print(f"Found {len(mtv_transcripts)} MTV transcripts")
    print(f"Found {len(nmtv_transcripts)} NMTV transcripts")
    print(f"Total: {len(mtv_transcripts) + len(nmtv_transcripts)} videos")

    # Estimate cost
    total_videos = len(mtv_transcripts) + len(nmtv_transcripts)
    # Claude 3.5 Sonnet: ~$3 per 1M input tokens, ~$15 per 1M output tokens
    # Image ~1600 tokens, prompt ~50 tokens, output ~50 tokens
    # Rough estimate: ~$0.006 per image
    estimated_cost = total_videos * 0.006
    print(f"\nEstimated API cost: ~${estimated_cost:.2f}")
    print("(Actual cost may vary based on image sizes and response lengths)")

    response = input("\nProceed with Claude API extraction? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted by user")
        return False

    # Process videos
    all_results = []

    print("\n" + "="*60)
    print("Processing MTV videos (clickbait)...")
    print("="*60)
    mtv_results = process_videos(
        mtv_transcripts, mtv_thumb_dir,
        claude_client, bert_model, bert_tokenizer,
        device, label=1
    )
    all_results.extend(mtv_results)

    print("\n" + "="*60)
    print("Processing NMTV videos (non-clickbait)...")
    print("="*60)
    nmtv_results = process_videos(
        nmtv_transcripts, nmtv_thumb_dir,
        claude_client, bert_model, bert_tokenizer,
        device, label=0
    )
    all_results.extend(nmtv_results)

    # Create DataFrame
    print("\nCreating DataFrame...")
    df = pd.DataFrame(all_results)

    # Separate descriptions from features for readability
    descriptions_df = df[['video_id', 'vllm_description', 'label']]
    descriptions_file = os.path.join(output_dir, 'vllm_descriptions.csv')
    descriptions_df.to_csv(descriptions_file, index=False)

    # Save features (drop description column for main features file)
    features_df = df.drop(columns=['vllm_description'])
    features_df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print("SUCCESS! VLLM features extracted with Claude")
    print('='*60)
    print(f"\nFeatures saved to: {output_file}")
    print(f"Descriptions saved to: {descriptions_file}")
    print(f"Total videos processed: {len(features_df)}")
    print(f"Output shape: {features_df.shape}")

    # Statistics
    print("\nIncongruence Score Statistics:")
    print(f"  Range: [{features_df['incongruence_score'].min():.3f}, {features_df['incongruence_score'].max():.3f}]")
    print(f"  Mean: {features_df['incongruence_score'].mean():.3f}")
    print(f"  Std: {features_df['incongruence_score'].std():.3f}")

    print("\nBy Label:")
    for label in [0, 1]:
        subset = features_df[features_df['label'] == label]
        label_name = "Non-Clickbait" if label == 0 else "Clickbait"
        print(f"  {label_name}: mean={subset['incongruence_score'].mean():.3f}, std={subset['incongruence_score'].std():.3f}")

    print("\nInterpretation:")
    print("  - Higher incongruence (closer to 1): Thumbnail shows X, but video is about Y")
    print("  - Lower incongruence (closer to 0): Thumbnail accurately represents content")
    print("  - Hypothesis: Clickbait videos should have HIGHER incongruence scores")

    # Show sample descriptions
    print("\nSample Claude Descriptions (first 5):")
    for idx, row in descriptions_df.head(5).iterrows():
        label_name = "Clickbait" if row['label'] == 1 else "Non-Clickbait"
        print(f"\n  Video: {row['video_id']} ({label_name})")
        print(f"  Description: {row['vllm_description']}")

    return True


if __name__ == "__main__":
    success = main()
    if success is False:
        import sys
        sys.exit(1)