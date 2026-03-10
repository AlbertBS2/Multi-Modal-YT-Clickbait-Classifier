import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import matplotlib.pyplot as plt
from PIL import Image

# Import from late_fusion
from late_fusion import ClickbaitClassifier, load_data, prepare_datasets, create_dataloaders, evaluate


# Get the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "CreateML", "datasets", "cnn_vllm_clip_max_mean_incong")
THUMBNAIL_DIR = os.path.join(PROJECT_ROOT, "data", "ThumbnailTruthData")
MTV_THUMBNAILS = os.path.join(THUMBNAIL_DIR, "MTV_Thumbnails")  # Clickbait
NMTV_THUMBNAILS = os.path.join(THUMBNAIL_DIR, "NMTV_Thumbnails")  # Non-clickbait

MODEL_PATH = os.path.join(SCRIPT_DIR, "cnn_vllm_clip_max_mean_incong.pth")
val_path = os.path.join(DATA_DIR, "valid.parquet")
train_path = os.path.join(DATA_DIR, "train.parquet")


def load_model(model_path, input_dim, device):
    """Load a trained model from disk."""
    model = ClickbaitClassifier(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def run_inference(model, data_loader, device):
    """Run inference on data and return predictions and probabilities."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            
            outputs = model(X_batch)
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(y_batch.numpy().flatten())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def save_misclassified_thumbnails(video_ids, labels, predictions, probabilities, output_path, title, max_images=10):
    """
    Save a grid image showing misclassified thumbnails.
    
    Args:
        video_ids: Array of video IDs
        labels: True labels (1=clickbait, 0=non-clickbait)
        predictions: Model predictions
        probabilities: Model output probabilities
        output_path: Path to save the output image
        title: Title for the figure
        max_images: Maximum number of misclassified images to show
    """
    n_images = min(len(video_ids), max_images)
    
    if n_images == 0:
        print(f"No samples found for: {title}")
        return
    
    print(f"\nFound {len(video_ids)} samples for '{title}', showing {n_images}")
    
    # Calculate grid dimensions
    cols = min(5, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Flatten axes for easy iteration
    if n_images == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    for i in range(n_images):
        video_id = video_ids[i]
        true_label = int(labels[i])
        pred_label = int(predictions[i])
        prob = probabilities[i]
        
        # Determine thumbnail path based on true label
        # True label 1 = clickbait = MTV, True label 0 = non-clickbait = NMTV
        if true_label == 1:
            thumbnail_path = os.path.join(MTV_THUMBNAILS, f"{video_id}.jpg")
        else:
            thumbnail_path = os.path.join(NMTV_THUMBNAILS, f"{video_id}.jpg")
        
        ax = axes[i]
        
        if os.path.exists(thumbnail_path):
            img = Image.open(thumbnail_path)
            ax.imshow(img)
        else:
            # Show placeholder if image not found
            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', 
                    fontsize=12, transform=ax.transAxes)
            ax.set_facecolor('lightgray')
        
        # Create title with classification info
        true_class = "Clickbait" if true_label == 1 else "Non-Clickbait"
        pred_class = "Clickbait" if pred_label == 1 else "Non-Clickbait"
        subtitle = f"True: {true_class}\nPred: {pred_class} ({prob:.2f})"
        ax.set_title(subtitle, fontsize=10, color='red')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load validation data
    print(f"\nLoading data from: {DATA_DIR}")
    df_train, df_val = load_data(train_path, val_path)
    label_col = "clickbait_label"
    
    # Get video IDs from validation set before preparing datasets
    video_ids = df_val["video_id"].values
    
    (X_train, y_train), (X_val, y_val) = prepare_datasets(df_train, df_val, label_col)
    
    # Create validation dataloader
    _, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
    
    # Get input dimension from training data
    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")
    print(f"Validation samples: {len(X_val)}")
    
    # Load the trained model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH, input_dim, device)
    print("Model loaded successfully!")
    
    # Run inference
    print("\n" + "="*50)
    print("Running Inference on Validation Set")
    print("="*50)
    
    predictions, probabilities, labels = run_inference(model, val_loader, device)
    
    # Calculate and display metrics
    criterion = nn.BCELoss()
    metrics = evaluate(model, val_loader, criterion, device)
    
    print(f"\nResults:")
    print(f"  Loss:      {metrics['loss']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # Print sample predictions
    print(f"\n" + "="*50)
    print("Sample Predictions (first 10)")
    print("="*50)
    print(f"{'Index':<8} {'True Label':<12} {'Predicted':<12} {'Probability':<12}")
    print("-" * 44)
    for i in range(min(10, len(predictions))):
        print(f"{i:<8} {int(labels[i]):<12} {int(predictions[i]):<12} {probabilities[i]:.4f}")
    
    # Class distribution in predictions
    print(f"\n" + "="*50)
    print("Prediction Distribution")
    print("="*50)
    print(f"Predicted Clickbait:     {np.sum(predictions == 1)} ({100*np.mean(predictions == 1):.1f}%)")
    print(f"Predicted Non-Clickbait: {np.sum(predictions == 0)} ({100*np.mean(predictions == 0):.1f}%)")
    
    # Save misclassified thumbnails (False Positives and False Negatives)
    # False Positives: True=Non-Clickbait (0), Predicted=Clickbait (1)
    fp_mask = (labels == 0) & (predictions == 1)
    fp_indices = np.where(fp_mask)[0][:10]
    
    # False Negatives: True=Clickbait (1), Predicted=Non-Clickbait (0)
    fn_mask = (labels == 1) & (predictions == 0)
    fn_indices = np.where(fn_mask)[0][:10]
    
    # Save False Positives
    fp_output_path = os.path.join(SCRIPT_DIR, "false_positives.png")
    save_misclassified_thumbnails(
        video_ids=video_ids[fp_indices],
        labels=labels[fp_indices],
        predictions=predictions[fp_indices],
        probabilities=probabilities[fp_indices],
        output_path=fp_output_path,
        title="False Positives (Non-Clickbait predicted as Clickbait)",
        max_images=10
    )
    
    # Save False Negatives
    fn_output_path = os.path.join(SCRIPT_DIR, "false_negatives.png")
    save_misclassified_thumbnails(
        video_ids=video_ids[fn_indices],
        labels=labels[fn_indices],
        predictions=predictions[fn_indices],
        probabilities=probabilities[fn_indices],
        output_path=fn_output_path,
        title="False Negatives (Clickbait predicted as Non-Clickbait)",
        max_images=10
    )