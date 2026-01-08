"""
VLLM Feature Extraction using Claude Batch API (Vision)

This script uses Claude's Batch API with vision capabilities to:
1. Generate descriptions of what thumbnails actually show
2. Compare descriptions to actual video transcripts
3. Calculate incongruence score (mismatch between thumbnail promise and content delivery)

Uses Anthropic's Batch API (50% cost savings compared to standard API).

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

# LOAD .ENV FILE
load_dotenv()


# ENCODE IMAGE
def encode_image_base64(image_path):
    """Encode image to base64 for Claude API."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def load_video_info(video_data, thumbnail_dir, label):
    """Load video info without encoding images."""
    video_info = []
    for _, row in video_data.iterrows():
        video_id = row['video_id']
        thumb_path = os.path.join(thumbnail_dir, f"{video_id}.jpg")
        if os.path.exists(thumb_path):
            video_info.append({
                'video_id': video_id,
                'transcript': row['transcript'],
                'label': label
            })
    return video_info


# BATCH
def prepare_batch_requests(video_data, thumbnail_dir, label):
    """Prepare batch requests for all videos."""
    requests = []
    video_info = []

    for _, row in video_data.iterrows():
        video_id = row['video_id']
        thumb_path = os.path.join(thumbnail_dir, f"{video_id}.jpg")

        if not os.path.exists(thumb_path):
            continue

        image_data = encode_image_base64(thumb_path)
        ext = Path(thumb_path).suffix.lower()
        media_type = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"

        requests.append({
            "custom_id": video_id,
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 150,
                "messages": [{
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
                }]
            }
        })

        video_info.append({
            'video_id': video_id,
            'transcript': row['transcript'],
            'label': label
        })

    return requests, video_info


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

def process_batch_results(batch_results, video_info_map, bert_model, bert_tokenizer, device):
    """Process batch API results and calculate incongruence scores."""
    results = []

    for result in tqdm(batch_results, desc="Processing results"):
        video_id = result.custom_id
        if video_id not in video_info_map:
            continue

        video = video_info_map[video_id]
        claude_desc = result.result.message.content[0].text

        # Calculate incongruence and get Tvllm embedding
        incong_score, tvllm_emb = calculate_incongruence(
            claude_desc, video['transcript'], bert_model, bert_tokenizer, device
        )

        # Store results
        result_dict = {
            'video_id': video_id,
            'vllm_description': claude_desc,
            'incongruence_score': incong_score,
            'label': video['label']
        }

        # Add Tvllm embedding columns (768 dimensions)
        for i, val in enumerate(tvllm_emb):
            result_dict[f'tvllm_{i}'] = float(val)

        results.append(result_dict)

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
    batch_id_file = os.path.join(output_dir, 'vllm_batch_id.txt')

    print("="*60)
    print("VLLM Feature Extraction (Claude Batch API)")
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
    claude_client = Anthropic(api_key=api_key)

    # Check for pending batch
    if os.path.exists(batch_id_file):
        with open(batch_id_file, 'r') as f:
            batch_id = f.read().strip()

        print(f"\n✓ Found pending batch: {batch_id}")
        print("Checking status...")

        batch = claude_client.messages.batches.retrieve(batch_id)

        if batch.processing_status == 'ended':
            succeeded = batch.request_counts.succeeded
            total = succeeded + batch.request_counts.errored
            print(f"✓ Batch complete: {succeeded}/{total} succeeded")
        else:
            print(f"⏳ Batch still processing, run again later to retrieve results")
            return True

        # Load necessary data to process results
        print("\nLoading BERT model...")
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model = bert_model.to(device)
        bert_model.eval()

        print("Loading transcripts...")
        mtv_transcripts = pd.read_csv(mtv_transcripts_path, sep='\t')
        nmtv_transcripts = pd.read_csv(nmtv_transcripts_path, sep='\t')

        print("Loading video info...")
        mtv_info = load_video_info(mtv_transcripts, mtv_thumb_dir, label=1)
        nmtv_info = load_video_info(nmtv_transcripts, nmtv_thumb_dir, label=0)
        video_info_map = {v['video_id']: v for v in mtv_info + nmtv_info}

        # Retrieve and process results
        print("Retrieving results...")
        batch_results = []
        for result in claude_client.messages.batches.results(batch_id):
            if result.result.type == 'succeeded':
                batch_results.append(result)

        print(f"Processing {len(batch_results)} results...")
        all_results = process_batch_results(batch_results, video_info_map, bert_model, bert_tokenizer, device)

        # Clean up batch ID file
        os.remove(batch_id_file)

        # Create DataFrame
        print("\nCreating DataFrame...")
        df = pd.DataFrame(all_results)

        # Separate descriptions from features for readability
        descriptions_df = df[['video_id', 'vllm_description', 'label']]
        descriptions_df.to_csv(descriptions_file, index=False)

        # Save features (drop description column for main features file)
        features_df = df.drop(columns=['vllm_description'])
        features_df.to_csv(output_file, index=False)

        print(f"\n{'='*60}")
        print("SUCCESS! VLLM features extracted")
        print('='*60)
        print(f"Features saved to: {output_file}")
        print(f"Total videos: {len(features_df)}")
        print(f"Output shape: {features_df.shape}")

        # Statistics
        print(f"\nIncongruence: mean={features_df['incongruence_score'].mean():.3f}, std={features_df['incongruence_score'].std():.3f}")

        return True

    else:
        # New batch submission
        print("\nLoading BERT model...")
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model = bert_model.to(device)
        bert_model.eval()

        print("Loading transcripts...")
        mtv_transcripts = pd.read_csv(mtv_transcripts_path, sep='\t')
        nmtv_transcripts = pd.read_csv(nmtv_transcripts_path, sep='\t')

        print(f"Found {len(mtv_transcripts)} MTV + {len(nmtv_transcripts)} NMTV transcripts")

        print("Preparing batch requests...")
        mtv_requests, mtv_info = prepare_batch_requests(mtv_transcripts, mtv_thumb_dir, label=1)
        nmtv_requests, nmtv_info = prepare_batch_requests(nmtv_transcripts, nmtv_thumb_dir, label=0)

        all_requests = mtv_requests + nmtv_requests

        print(f"Prepared {len(all_requests)} requests")

        # Estimate cost (Batch API is 50% cheaper)
        estimated_cost = len(all_requests) * 0.003
        print(f"Estimated cost: ~${estimated_cost:.2f} (50% savings)")

        response = input("\nSubmit batch? [y/N]: ").strip().lower()
        if response != 'y':
            return False

        # Submit batch
        print("\nSubmitting batch...")
        batch = claude_client.messages.batches.create(requests=all_requests)
        batch_id = batch.id

        # Save batch ID
        with open(batch_id_file, 'w') as f:
            f.write(batch_id)

        print(f"✓ Batch submitted: {batch_id}")
        print("Run this script again later to retrieve results")
        return True


if __name__ == "__main__":
    success = main()
    if success is False:
        import sys
        sys.exit(1)