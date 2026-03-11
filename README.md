# A Multi-Modal Clickbait (Misleading) Classifier for YouTube Thumbnails

> Can we accurately detect YouTube clickbait by quantifying the incongruence between thumbnails and transcripts? Is it more effective than only using the thumbnails?

## Overview

This project builds a **multi-modal late-fusion classifier** that detects misleading ("clickbait") YouTube thumbnails by combining visual features, semantic alignment scores, and thumbnail–transcript incongruence signals. Rather than relying on thumbnails alone, the system jointly reasons over what the thumbnail *shows*, what the video *contains*, and how much the two *disagree* — mimicking the way human annotators judge clickbait.

## Dataset

Based on the **ThumbnailTruth** dataset (from *ThumbnailTruth: A Multi-Modal LLM Approach for Detecting Misleading YouTube Thumbnails Across Diverse Cultural Settings*).

| Property | Value |
|---|---|
| Total videos | 2 222 (subset of original; limited by video/transcript availability) |
| Clickbait (MTV) | 1 114 (50.1 %) |
| Non-Clickbait (NMTV) | 1 108 (49.9 %) |
| Countries represented | 8 (high-income and middle-income economies) |

Each sample provides a **video transcript**, a **thumbnail image**, and a **clickbait label** (yes / no).

### What counts as clickbait?

A thumbnail is considered clickbait if it:

- Depicts a subject or event that is not actually present or central to the video.
- Asserts or strongly implies a factual outcome that does not occur.
- Uses doctored images or real images in a false context without disclosure.
- Shows hyperbolic visual/text claims framed as factual (not clearly comedic/satirical) that are unsupported by the video.

## Method

### Feature Extraction

| Feature | Dimensionality | Description |
|---|---|---|
| **V_cnn** (ResNet50) | 2 048 | Visual feature vector from the penultimate layer. Captures colors, composition, facial expressions, and visual emphasis cues. |
| **S_clip** (CLIP) | 2 | Max and mean cosine similarity between the thumbnail and transcript chunks. Measures thematic overlap between thumbnail and video. |
| **T_vllm** (Claude → BERT) | 768 | BERT embedding of a Claude-generated textual description of the thumbnail. Enables textual comparison of what the thumbnail depicts. |
| **I_score** (Incongruence) | 1 | Cosine similarity between the VLLM thumbnail description and the video transcript. Quantifies contradiction between thumbnail and video. |
| **NLP features** | 406 | Sentence-transformer embeddings (384-d) plus statistical, sentiment, and complexity features extracted from the transcript. |

### Late Fusion

All features are concatenated into a single **Master Vector**:

$$F_{\text{comb}} = [V_{\text{cnn}} \oplus S_{\text{clip}} \oplus T_{\text{vllm}} \oplus I_{\text{score}}]$$

This vector is fed into a fully-connected neural network:

- **Input**: 3 224 dimensions
- **Hidden layers**: 512 → 128 (ReLU activation)
- **Regularization**: Dropout
- **Output**: Sigmoid scalar (clickbait probability)

### Why a Master Vector?

Clickbait is multi-dimensional: a shocked face alone (V_cnn) is common in both clickbait and comedy; low CLIP similarity may stem from creative content rather than deception. Individual signals are weak in isolation. By concatenating all features, the network can learn **conditional logic** — flagging clickbait only when multiple signals co-occur.  This mirrors how human annotators judge clickbait: checking the thumbnail, watching the video, and assessing incongruence.

## Results

### Final Model (Late Fusion)

| Metric | Value |
|---|---|
| Loss | 0.4452 |
| Accuracy | 88.32 % |
| Precision | 90.06 % |
| Recall | 86.31 % |
| F1 Score | 88.15 % |
| AUC-ROC | 93.55 % |

### Baseline (CNN-only, thumbnails only)

| Metric | Value |
|---|---|
| Accuracy | 83.46 % |
| Precision | 84.05 % |
| Recall | 78.29 % |
| F1 Score | 81.07 % |

The multi-modal model improves F1 by **~7 percentage points** over the visual-only baseline and substantially boosts recall, confirming that quantifying thumbnail–transcript incongruence is more effective than using thumbnails alone.

### Known Failure Modes

The model tends to misclassify when:

- Thumbnails contain exaggerated faces/expressions or objects (common in non-clickbait entertainment).
- Titles use exaggerated text or ALL CAPS (style overlap between clickbait and legitimate content).

### Reflection on Representations

- **Representational bias**: ResNet50 was pretrained on ImageNet (predominantly Western photography), which may cause cultural visual cues from non-Western creators to be misinterpreted.
- **Reductionism**: Compressing a 20-minute transcript into a single vector or incongruence score inevitably loses nuance.

## Project Structure

```
├── main.py                          # Master pipeline – runs all extraction steps in sequence
├── fetch_comments.py                # Scrape YouTube comments via AJAX API
├── fetch_thumbnails.py              # Download HQ thumbnails (1280×720) by video ID
├── fetch_transcript.py              # Extract transcripts (manual preferred over auto-generated)
├── remove_missing.py                # Remove videos with missing thumbnails or transcripts
├── v_cnn_extraction.py              # Utility: ResNet50 feature extractor (2048-d vectors)
├── v_cnn_example.ipynb              # Notebook example for CNN feature extraction
│
├── data/
│   └── ThumbnailTruthData/
│       ├── mtv.csv / nmtv.csv                   # Raw labeled datasets (URL-based)
│       ├── mtv_cleaned.csv / nmtv_cleaned.csv   # Cleaned datasets (video_id-based)
│       ├── MTV_transcripts.tsv / NMTV_transcripts.tsv  # Extracted transcripts
│       ├── mtv_no-thumb.csv / nmtv_no-thumb.csv        # Failed thumbnail downloads
│       ├── mtv_no-transcripts.csv / nmtv_no-transcripts.csv  # Missing transcripts
│       ├── MTV_Thumbnails/                       # Clickbait thumbnail images
│       └── NMTV_Thumbnails/                      # Non-clickbait thumbnail images
│
├── CreateML/
│   ├── extract_visual_features.py     # Batch ResNet50 feature extraction (GPU/MPS)
│   ├── extract_nlp_features.py        # Transcript embeddings + linguistic features (406-d)
│   ├── extract_clip_features.py       # CLIP thumbnail–transcript similarity (sliding window)
│   ├── extract_vllm_features_claude.py  # Claude thumbnail descriptions + BERT encoding + incongruence
│   ├── nlp_feature_extraction.py      # NLP utility: 22 feature extractors (sentiment, complexity, etc.)
│   ├── create_multimodal_dataset.py   # Merge all features + train/val/test split (70/15/15)
│   ├── master_merge.py               # Alternative merge with interactive feature selection UI
│   ├── analyze_features.py           # PCA, t-SNE, Random Forest feature importance
│   ├── createml_dataset_validator.py  # Pre-training validation (types, balance, column names)
│   ├── test.py                        # CoreML .mlmodel inference and evaluation
│   ├── requirements.txt               # Python dependencies
│   ├── features/                      # Intermediate feature CSVs
│   │   ├── visual_features.csv          # 2048-d ResNet50 vectors
│   │   ├── nlp_features.csv             # 406-d transcript features
│   │   ├── clip_features.csv            # Max + mean CLIP similarity
│   │   ├── vllm_features.csv            # 769-d VLLM feature vectors
│   │   ├── vllm_descriptions.csv        # Claude-generated thumbnail descriptions
│   │   └── incongruence_scores.csv      # Thumbnail–transcript incongruence
│   ├── datasets/                      # Final merged datasets (train/val/test splits)
│   └── analysis/                      # Feature statistics and importance rankings
│
└── models/
    ├── Base_Model/
    │   └── cnn_classifier.py          # Visual-only ResNet50 baseline with data augmentation
    └── Late_Fusion/
        ├── late_fusion.py             # Multi-modal fusion model (3224-d → 512 → 128 → 1)
        ├── evaluate_model.py          # Inference, ROC curves, misclassification visualization
        ├── cnn_vllm.pth               # Trained model weights (CNN + VLLM features)
        └── cnn_vllm_clip_max_mean_incong.pth  # Trained model weights (all features)
```

## Pipeline

The full extraction and training pipeline (orchestrated by `main.py`):

1. **Data collection** — `fetch_thumbnails.py`, `fetch_transcript.py`, `fetch_comments.py`
2. **Data cleaning** — `remove_missing.py`
3. **Visual feature extraction** — `extract_visual_features.py` (ResNet50, ~2 048-d)
4. **NLP feature extraction** — `extract_nlp_features.py` (sentence embeddings + linguistic metrics)
5. **CLIP alignment** — `extract_clip_features.py` (thumbnail ↔ transcript similarity)
6. **VLLM description + incongruence** — `extract_vllm_features_claude.py` (Claude API, ~\$13)
7. **Feature merging** — `create_multimodal_dataset.py` or `master_merge.py`
8. **Training** — `models/Late_Fusion/late_fusion.py`
9. **Evaluation** — `models/Late_Fusion/evaluate_model.py`

## Setup

### Dependencies

```bash
pip install -r CreateML/requirements.txt
```

Key libraries: PyTorch, torchvision, transformers, sentence-transformers, anthropic, nltk, scikit-learn, Pillow, matplotlib.

### API Keys

- **Anthropic API key** — required for VLLM thumbnail description generation (`extract_vllm_features_claude.py`). Set via `.env` or environment variable.

### Running

```bash
# Run the full feature extraction pipeline
python main.py

# Or run individual steps
python CreateML/extract_visual_features.py
python CreateML/extract_nlp_features.py
python CreateML/extract_clip_features.py
python CreateML/extract_vllm_features_claude.py
python CreateML/create_multimodal_dataset.py

# Train the late fusion model
python models/Late_Fusion/late_fusion.py

# Evaluate
python models/Late_Fusion/evaluate_model.py
```

## Acknowledgments

This project builds on the **ThumbnailTruth** dataset and methodology. The original dataset and labeling criteria are described in:

> *ThumbnailTruth: A Multi-Modal LLM Approach for Detecting Misleading YouTube Thumbnails Across Diverse Cultural Settings*
