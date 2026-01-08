# Enhanced Multimodal Feature Extraction for CreateML

This directory contains a complete pipeline for extracting multimodal features from YouTube videos and preparing them for training in CreateML's Tabular Classification template or PyTorch.

## Overview

This pipeline combines **two approaches**:

### Base Features (Always Extracted)
- **2048 visual features (Vcnn)** from thumbnails using ResNet50
- **406 NLP features** from transcripts:
  - 384-dim sentence embeddings (semantic meaning)
  - 15 statistical features (text patterns)
  - 3 sentiment features (emotional tone)
  - 4 complexity features (readability)

### Gemini Enhancement Features (Optional but Recommended)
- **1 CLIP alignment score (Sclip)**: Measures semantic match between thumbnail and transcript
- **769 VLLM incongruence features**:
  - 768-dim BERT embeddings of thumbnail description (Tvllm)
  - 1 incongruence score (thumbnail promise vs actual content)

**Total Options:**
- **Base only**: 2,454 features
- **Base + CLIP**: 2,455 features
- **Base + CLIP + VLLM**: 3,224 features (Full Gemini approach)

## Quick Start

### 1. Install Dependencies

```bash
cd CreateML
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Configure API Key (Optional - for VLLM features)

VLLM features require Claude API. Skip this if you only want base features.

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
# Get key from: https://console.anthropic.com/settings/keys
nano .env  # or use any text editor
```

Your `.env` file should contain:
```
ANTHROPIC_API_KEY=your-actual-api-key-here
```

### 3. Extract Features

**Option A: Automated - Full Pipeline (Recommended)**
```bash
# Run from project root
python main.py

# With API key configured:
#   - Creates 3,224 features (Vcnn + NLP + CLIP + VLLM)
#   - Time: ~1-2 hours (no GPU required)
#   - Cost: ~$13 (Claude API usage)

# Without API key:
#   - Creates 2,455 features (Vcnn + NLP + CLIP)
#   - Skips VLLM extraction
#   - Time: ~45-60 minutes
#   - Cost: $0
```

**Option B: Manual - Step by Step**
```bash
# From CreateML directory
cd CreateML

# Extract visual features (~10-30 minutes)
python extract_visual_features.py

# Extract NLP features (~15 minutes)
python extract_nlp_features.py

# Extract CLIP alignment (~30 minutes)
python extract_clip_features.py

# Extract VLLM incongruence (~1-2 hours, requires API key, optional)
python extract_vllm_features_claude.py

# Merge all features
python create_multimodal_dataset.py
```

### 4. Validate and Analyze

```bash
# Validate CreateML compatibility
python createml_dataset_validator.py

# Analyze features (optional)
python analyze_features.py
```

### 5. Train Model

**Option A: CreateML (GUI)**
1. Open Xcode
2. Create new CreateML project
3. Select "Tabular Classifier" template
4. Import `datasets/train.csv` as training data
5. Import `datasets/validation.csv` as validation data
6. Set "label" as target column
7. Set "video_id" as metadata (exclude from training)
8. Train model
9. Evaluate on `datasets/test.csv`

**Option B: PyTorch (Code)**
```bash
# Coming soon: train_pytorch.py
python train_pytorch.py --data datasets/train.csv --epochs 20
```

## File Structure

```
CreateML/
├── nlp_feature_extraction.py        # Core NLP feature extraction functions
├── extract_visual_features.py       # Visual features (Vcnn) - ResNet50
├── extract_nlp_features.py          # NLP features - embeddings + stats
├── extract_clip_features.py         # CLIP alignment (Sclip)
├── extract_vllm_features_claude.py  # VLLM incongruence (Claude API)
├── create_multimodal_dataset.py     # Feature fusion and data splitting
├── createml_dataset_validator.py    # CreateML compatibility validation
├── analyze_features.py              # Feature analysis and visualization
├── requirements.txt                 # Python dependencies
├── features/                        # Extracted features (created by scripts)
│   ├── visual_features.csv          # Vcnn (2048-dim)
│   ├── nlp_features.csv             # NLP (406-dim)
│   ├── clip_features.csv            # Sclip (1-dim) - optional
│   ├── vllm_features.csv            # Tvllm + incongruence (769-dim) - optional
│   └── vllm_descriptions.csv        # Human-readable descriptions
├── datasets/                        # Train/val/test splits (created by scripts)
│   ├── train.csv                    # 2,454 to 3,224 features
│   ├── validation.csv
│   ├── test.csv
│   └── dataset_summary.json         # Feature breakdown
└── analysis/                        # Feature analysis results (created by scripts)
    ├── feature_statistics.csv
    ├── pca_plot.png
    ├── tsne_plot.png
    ├── feature_importance_top50.csv
    └── feature_importance_plot.png
```

## Feature Details

### Base Features

#### Visual Features - Vcnn (2048 dimensions)
- **Model**: ResNet50 (pre-trained on ImageNet)
- **Extraction**: Layer before final classification
- **Captures**: Visual patterns, colors, text overlays, composition, thumbnail style
- **Why**: General-purpose visual understanding

#### NLP Features (406 dimensions)

**Sentence Embeddings (384-dim):**
- Model: sentence-transformers/all-MiniLM-L6-v2
- Captures semantic meaning and context
- Better than keyword-based methods (TF-IDF)

**Statistical Features (15-dim):**
- transcript_length, word_count, avg_word_length
- sentence_count, avg_sentence_length
- exclamation_count, question_count, ellipsis_count
- uppercase_word_ratio, capitalized_word_ratio, all_caps_ratio
- number_count, unique_word_ratio, stopword_ratio, punctuation_density

**Sentiment Features (3-dim):**
- sentiment_polarity (-1 to 1)
- sentiment_subjectivity (0 to 1)
- sentiment_intensity (absolute polarity)

**Complexity Features (4-dim):**
- lexical_density
- avg_syllables_per_word
- flesch_reading_ease
- automated_readability_index

### Gemini Enhancement Features (Optional)

#### CLIP Alignment - Sclip (1 dimension)
- **Model**: CLIP (openai/clip-vit-base-patch32)
- **What it measures**: Cosine similarity between thumbnail image and transcript text
- **Range**: -1 to 1 (higher = better alignment)
- **Why it matters**: Directly detects if thumbnail matches content
  - **High score**: Thumbnail accurately represents video topic
  - **Low score**: Thumbnail misleads about video content → **clickbait signal!**
- **Example**: Thumbnail shows screaming face, transcript about gardening → Low alignment → Likely clickbait

#### VLLM Incongruence - Tvllm + Score (769 dimensions)

**Tvllm Embedding (768-dim):**
- **Model**: Claude 3.5 Sonnet (Vision) + BERT-base
- **Process**:
  1. Claude API generates description of what thumbnail actually shows
  2. BERT encodes description into 768-dim embedding
- **Why**: Captures what the thumbnail "promises" to the viewer
- **Note**: Uses Claude API (no GPU required, ~$13 for 2,223 images)

**Incongruence Score (1-dim):**
- **Calculation**: 1 - cosine_similarity(Tvllm, transcript_embedding)
- **Range**: 0 to 1 (higher = more mismatch)
- **Why it matters**: Core clickbait detection mechanism
  - **Low incongruence (0-0.3)**: Thumbnail and content align → Trustworthy
  - **High incongruence (0.7-1.0)**: Thumbnail shows X but video delivers Y → **Clickbait!**
- **Example**:
  - Thumbnail: Person shocked by spider (Claude describes: "shocked person pointing")
  - Transcript: "Today we'll learn Python web development"
  - Incongruence: 0.85 (very high) → Clear clickbait

**Why Gemini Features are Powerful:**
Unlike base features that learn "what clickbait looks like," these features directly measure the **clickbait mechanism**: deception through misalignment between promise (thumbnail) and delivery (content).

## Expected Performance

| Configuration | Features | Expected Accuracy |
|---------------|----------|-------------------|
| **Base only** | 2,454 | 65-75% (baseline) |
| **Base + CLIP** | 2,455 | 72-80% (recommended) |
| **Base + CLIP + VLLM** | 3,224 | 78-88% (best) |

**Why Gemini features improve performance:**
- CLIP catches semantic misalignment
- VLLM directly models the clickbait deception mechanism
- Combined: Both "what" (visual style) and "why" (intentional misleading)

## Why This Approach is Better

1. **Pre-trained visual features** capture patterns better than training from scratch (only 2,223 samples)
2. **Sentence embeddings** capture semantic meaning that keyword methods miss
3. **Multimodal fusion** combines visual style + language patterns
4. **Tabular classifier** optimal for fixed-dimension features
5. **Interpretable** - can analyze which features matter most

## Troubleshooting

### Out of Memory Error
- Reduce batch_size in extract_visual_features.py (default: 32)
- Reduce batch_size in extract_nlp_features.py (default: 64)

### CUDA/GPU Issues
- The scripts will automatically use CPU if GPU is unavailable
- Visual feature extraction will be slower on CPU (~30 min vs 10 min)

### Missing Transcripts/Thumbnails
- The pipeline automatically skips videos with missing data
- Check the console output for warnings about failed files

### VLLM Issues (Claude API)

**API Key Not Found:**
- Ensure `.env` file exists in project root
- Verify `ANTHROPIC_API_KEY` is set correctly in `.env`
- Get API key from: https://console.anthropic.com/settings/keys

**API Rate Limits:**
- Script includes automatic rate limiting (0.5s delay between requests)
- If you hit rate limits, the delay automatically increases
- Free tier has lower limits - consider upgrading if needed

**High API Costs:**
- Estimated ~$13 for 2,223 images
- To reduce costs, test on a subset first (modify script to process fewer videos)
- Monitor usage at: https://console.anthropic.com/usage

**BERT Model Issues:**
- BERT model downloads automatically (~500MB)
- Model cached at: `~/.cache/huggingface/hub/`
- Try: `pip install --upgrade transformers`

## Alternative Approaches

**If dimensionality is too high:**
- Use PCA to reduce visual features from 2048 to 512 dimensions
- Use feature selection to keep top 500 most important features

**If performance is insufficient:**
- Add video title features (if available)
- Use larger sentence embeddings (768-dim with all-mpnet-base-v2)
- Ensemble separate visual and text models

**If computation is too slow:**
- Use smaller sentence transformer model
- Skip sentence embeddings, use only statistical features (22-dim)
- Process on cloud GPU (Google Colab, AWS, etc.)

## Questions?

This approach allows you to:
1. Extract rich features in Python (where you have full control)
2. Train in CreateML (optimized for Apple hardware)
3. Compare both Python and CreateML approaches easily
4. Iterate on features without retraining entire networks

The features are ready for CreateML's Tabular Classification template!
