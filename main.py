"""
Main Pipeline - Full Multimodal Feature Extraction

This script runs all feature extraction scripts to create the complete dataset:
1. Visual features (Vcnn) - 2048 dimensions
2. NLP features - 406 dimensions
3. CLIP features (Sclip) - 1 dimension
4. VLLM features (Tvllm + Incongruence) - 769 dimensions
5. Merge and create train/val/test splits

Output: 3,224-feature dataset ready for CreateML or PyTorch

Usage: python main.py
"""

import subprocess
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_script(script_path):
    """Run a Python script and print its output."""
    print(f"\n{'='*60}")
    print(f"Running: {os.path.basename(script_path)}")
    print('='*60)

    result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(script_path))

    if result.returncode != 0:
        print(f"\n✗ Script failed with exit code {result.returncode}")
        return False
    print(f"\n✓ Completed successfully")
    return True


def main():
    createml_dir = os.path.join(os.path.dirname(__file__), 'CreateML')

    # All scripts to run
    scripts = [
        ('extract_visual_features.py', 'Visual Features (Vcnn)'),
        ('extract_nlp_features.py', 'NLP Features'),
        ('extract_clip_features.py', 'CLIP Alignment (Sclip)'),
        ('extract_vllm_features_claude.py', 'VLLM Incongruence (Tvllm)'),
        ('create_multimodal_dataset.py', 'Dataset Creation'),
    ]

    print("="*60)
    print("MULTIMODAL FEATURE EXTRACTION PIPELINE")
    print("="*60)
    print("Extracting ALL features (3,224 dimensions)")

    # Check API key availability
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not found - VLLM will use cached features if available")

    print("="*60)

    for script, name in scripts:
        script_path = os.path.join(createml_dir, script)
        print(f"\nStep: {name}")
        if not run_script(script_path):
            print(f"\n✗ Pipeline failed at: {name}")
            sys.exit(1)

    print("\n" + "="*60)
    print("✓✓✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nDatasets created in: CreateML/datasets/")
    print("  - train.csv (3,224 features: Vcnn + NLP + CLIP + VLLM)")
    print("  - validation.csv")
    print("  - test.csv")
    print("\nReady for training in CreateML or PyTorch!")


if __name__ == "__main__":
    main()