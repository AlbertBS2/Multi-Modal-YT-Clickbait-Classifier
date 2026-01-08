"""
Feature Analysis Script

This script analyzes the extracted multimodal features and generates visualizations
and statistics to understand feature distributions and class separability.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_dataset(file_path):
    """Load the multimodal dataset."""
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df


def generate_feature_statistics(df, output_dir):
    """
    Generate and save feature statistics.

    Args:
        df (pd.DataFrame): Dataset DataFrame.
        output_dir (str): Directory to save statistics.
    """
    print("\nGenerating feature statistics...")

    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['video_id', 'label']]
    X = df[feature_cols]
    y = df['label']

    # Overall statistics
    stats = X.describe().T
    stats['missing'] = X.isnull().sum()

    # Class-wise statistics
    clickbait_stats = X[y == 1].describe().T
    non_clickbait_stats = X[y == 0].describe().T

    clickbait_stats.columns = ['clickbait_' + col for col in clickbait_stats.columns]
    non_clickbait_stats.columns = ['non_clickbait_' + col for col in non_clickbait_stats.columns]

    # Combine statistics
    combined_stats = pd.concat([stats, clickbait_stats, non_clickbait_stats], axis=1)

    # Save to CSV
    stats_path = os.path.join(output_dir, 'feature_statistics.csv')
    combined_stats.to_csv(stats_path)
    print(f"✓ Saved feature statistics to {stats_path}")

    # Print summary for some key features
    print("\nSample feature statistics (first 10 features):")
    print(combined_stats[['mean', 'std', 'min', 'max']].head(10))


def plot_pca(df, output_dir):
    """
    Generate PCA visualization.

    Args:
        df (pd.DataFrame): Dataset DataFrame.
        output_dir (str): Directory to save plot.
    """
    print("\nGenerating PCA visualization...")

    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['video_id', 'label']]
    X = df[feature_cols].values
    y = df['label'].values

    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot each class
    colors = ['blue', 'red']
    labels = ['Non-Clickbait', 'Clickbait']

    for label_val, color, label_name in zip([0, 1], colors, labels):
        mask = y == label_val
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            label=label_name,
            alpha=0.6,
            s=20
        )

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Projection of Multimodal Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    pca_path = os.path.join(output_dir, 'pca_plot.png')
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved PCA plot to {pca_path}")
    print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")


def plot_tsne(df, output_dir):
    """
    Generate t-SNE visualization.

    Args:
        df (pd.DataFrame): Dataset DataFrame.
        output_dir (str): Directory to save plot.
    """
    print("\nGenerating t-SNE visualization (this may take a few minutes)...")

    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['video_id', 'label']]
    X = df[feature_cols].values
    y = df['label'].values

    # Apply t-SNE (use sample if dataset is large)
    if len(X) > 1000:
        print("  Using sample of 1000 points for t-SNE...")
        indices = np.random.choice(len(X), 1000, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_sample)

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot each class
    colors = ['blue', 'red']
    labels = ['Non-Clickbait', 'Clickbait']

    for label_val, color, label_name in zip([0, 1], colors, labels):
        mask = y_sample == label_val
        plt.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            c=color,
            label=label_name,
            alpha=0.6,
            s=20
        )

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Projection of Multimodal Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    tsne_path = os.path.join(output_dir, 'tsne_plot.png')
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved t-SNE plot to {tsne_path}")


def analyze_feature_importance(df, output_dir):
    """
    Analyze feature importance using Random Forest.

    Args:
        df (pd.DataFrame): Dataset DataFrame.
        output_dir (str): Directory to save results.
    """
    print("\nAnalyzing feature importance with Random Forest...")

    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['video_id', 'label']]
    X = df[feature_cols]
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    print(f"  Random Forest accuracy: {accuracy:.2%}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Clickbait', 'Clickbait']))

    # Get feature importances
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Save top 50 features
    top_features = importances.head(50)
    importance_path = os.path.join(output_dir, 'feature_importance_top50.csv')
    top_features.to_csv(importance_path, index=False)

    print(f"\n✓ Saved top 50 feature importances to {importance_path}")
    print("\nTop 10 most important features:")
    print(top_features.head(10))

    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    top_20 = importances.head(20)
    plt.barh(range(20), top_20['importance'].values)
    plt.yticks(range(20), top_20['feature'].values)
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Features (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Save plot
    importance_plot_path = os.path.join(output_dir, 'feature_importance_plot.png')
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved feature importance plot to {importance_plot_path}")


def plot_statistical_features(df, output_dir):
    """
    Plot distributions of key statistical features.

    Args:
        df (pd.DataFrame): Dataset DataFrame.
        output_dir (str): Directory to save plots.
    """
    print("\nPlotting statistical feature distributions...")

    # Select key statistical features
    stat_features = [
        'transcript_length', 'word_count', 'exclamation_count', 'question_count',
        'uppercase_word_ratio', 'sentiment_polarity', 'sentiment_subjectivity',
        'flesch_reading_ease'
    ]

    # Check which features exist
    available_features = [f for f in stat_features if f in df.columns]

    if not available_features:
        print("  Warning: No statistical features found in dataset")
        return

    # Create subplots
    n_features = len(available_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(available_features):
        ax = axes[idx]

        # Plot histograms for each class
        clickbait_values = df[df['label'] == 1][feature]
        non_clickbait_values = df[df['label'] == 0][feature]

        ax.hist(non_clickbait_values, bins=30, alpha=0.6, label='Non-Clickbait', color='blue')
        ax.hist(clickbait_values, bins=30, alpha=0.6, label='Clickbait', color='red')

        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save plot
    dist_path = os.path.join(output_dir, 'statistical_features_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved statistical feature distributions to {dist_path}")


def main():
    """Main function to analyze features."""
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(base_dir, 'datasets')
    analysis_dir = os.path.join(base_dir, 'analysis')

    os.makedirs(analysis_dir, exist_ok=True)

    train_path = os.path.join(datasets_dir, 'train.csv')

    # Check if dataset exists
    if not os.path.exists(train_path):
        print(f"Error: Training dataset not found at {train_path}")
        print("Please run create_multimodal_dataset.py first")
        return

    # Load dataset
    df = load_dataset(train_path)

    # Generate analyses
    generate_feature_statistics(df, analysis_dir)
    plot_pca(df, analysis_dir)
    plot_tsne(df, analysis_dir)
    analyze_feature_importance(df, analysis_dir)
    plot_statistical_features(df, analysis_dir)

    print("\n" + "="*60)
    print("SUCCESS! Feature analysis complete")
    print("="*60)
    print(f"\nAnalysis files saved to: {analysis_dir}")
    print("\nGenerated files:")
    print("- feature_statistics.csv: Detailed feature statistics")
    print("- pca_plot.png: PCA visualization")
    print("- tsne_plot.png: t-SNE visualization")
    print("- feature_importance_top50.csv: Top 50 important features")
    print("- feature_importance_plot.png: Visual feature importance")
    print("- statistical_features_distribution.png: Feature distributions")


if __name__ == "__main__":
    main()
