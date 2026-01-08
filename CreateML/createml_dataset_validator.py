"""
CreateML Dataset Validator

This script validates that the dataset CSVs are compatible with CreateML Tabular Classification.
"""

import os
import pandas as pd
import numpy as np
import re


def validate_column_names(df, file_name):
    """
    Validate that column names are CreateML-compatible (no special characters).

    Args:
        df (pd.DataFrame): DataFrame to validate.
        file_name (str): Name of the file being validated.

    Returns:
        is_valid (bool): True if all column names are valid.
    """
    print(f"\n[{file_name}] Validating column names...")
    is_valid = True

    # Check for special characters (CreateML prefers alphanumeric and underscore)
    invalid_cols = []
    for col in df.columns:
        if not re.match(r'^[a-zA-Z0-9_]+$', col):
            invalid_cols.append(col)
            is_valid = False

    if invalid_cols:
        print(f"  ✗ Found {len(invalid_cols)} columns with invalid names:")
        for col in invalid_cols[:5]:
            print(f"    - {col}")
        if len(invalid_cols) > 5:
            print(f"    ... and {len(invalid_cols) - 5} more")
    else:
        print("  ✓ All column names are valid")

    return is_valid


def validate_missing_values(df, file_name):
    """
    Check for missing values in the dataset.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        file_name (str): Name of the file being validated.

    Returns:
        is_valid (bool): True if no missing values found.
    """
    print(f"\n[{file_name}] Checking for missing values...")

    missing_count = df.isnull().sum().sum()

    if missing_count > 0:
        print(f"  ✗ Found {missing_count} missing values")
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        print(f"  Columns with missing values: {cols_with_missing[:10]}")
        return False
    else:
        print("  ✓ No missing values")
        return True


def validate_label_column(df, file_name):
    """
    Validate the label column contains only 0 and 1.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        file_name (str): Name of the file being validated.

    Returns:
        is_valid (bool): True if label column is valid.
    """
    print(f"\n[{file_name}] Validating label column...")

    if 'label' not in df.columns:
        print("  ✗ 'label' column not found")
        return False

    unique_labels = df['label'].unique()

    if not set(unique_labels).issubset({0, 1}):
        print(f"  ✗ Label column contains invalid values: {unique_labels}")
        return False

    label_counts = df['label'].value_counts().sort_index()
    print(f"  ✓ Label column is valid")
    print(f"    Label 0: {label_counts.get(0, 0)} samples")
    print(f"    Label 1: {label_counts.get(1, 0)} samples")

    return True


def validate_numeric_columns(df, file_name):
    """
    Validate that all feature columns are numeric.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        file_name (str): Name of the file being validated.

    Returns:
        is_valid (bool): True if all feature columns are numeric.
    """
    print(f"\n[{file_name}] Validating feature columns are numeric...")

    # Exclude video_id (string) and label (target) from numeric check
    feature_cols = [col for col in df.columns if col not in ['video_id', 'label']]

    non_numeric = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)

    if non_numeric:
        print(f"  ✗ Found {len(non_numeric)} non-numeric feature columns:")
        for col in non_numeric[:5]:
            print(f"    - {col}: {df[col].dtype}")
        if len(non_numeric) > 5:
            print(f"    ... and {len(non_numeric) - 5} more")
        return False
    else:
        print(f"  ✓ All {len(feature_cols)} feature columns are numeric")
        return True


def validate_infinite_values(df, file_name):
    """
    Check for infinite values in numeric columns.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        file_name (str): Name of the file being validated.

    Returns:
        is_valid (bool): True if no infinite values found.
    """
    print(f"\n[{file_name}] Checking for infinite values...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols].values).sum()

    if inf_count > 0:
        print(f"  ✗ Found {inf_count} infinite values")
        return False
    else:
        print("  ✓ No infinite values")
        return True


def validate_file_encoding(file_path):
    """
    Validate that the file is UTF-8 encoded.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        is_valid (bool): True if file is UTF-8 encoded.
    """
    print(f"\n[{os.path.basename(file_path)}] Validating file encoding...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        print("  ✓ File is UTF-8 encoded")
        return True
    except UnicodeDecodeError:
        print("  ✗ File is not UTF-8 encoded")
        return False


def validate_csv_format(file_path):
    """
    Validate that the file is a valid CSV with proper formatting.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        is_valid (bool): True if CSV format is valid.
    """
    print(f"\n[{os.path.basename(file_path)}] Validating CSV format...")

    try:
        # Try to read with default CSV settings
        df = pd.read_csv(file_path)
        print(f"  ✓ CSV format is valid ({len(df)} rows, {len(df.columns)} columns)")
        return True
    except Exception as e:
        print(f"  ✗ CSV format error: {e}")
        return False


def validate_dataset_file(file_path):
    """
    Run all validation checks on a single dataset file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        is_valid (bool): True if all validations pass.
    """
    file_name = os.path.basename(file_path)
    print("\n" + "="*60)
    print(f"Validating: {file_name}")
    print("="*60)

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"  ✗ File not found: {file_path}")
        return False

    all_valid = True

    # Validate encoding
    all_valid &= validate_file_encoding(file_path)

    # Validate CSV format
    all_valid &= validate_csv_format(file_path)

    # Load DataFrame
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  ✗ Failed to load CSV: {e}")
        return False

    # Run all DataFrame validations
    all_valid &= validate_column_names(df, file_name)
    all_valid &= validate_missing_values(df, file_name)
    all_valid &= validate_label_column(df, file_name)
    all_valid &= validate_numeric_columns(df, file_name)
    all_valid &= validate_infinite_values(df, file_name)

    # Summary
    print(f"\n[{file_name}] Summary:")
    if all_valid:
        print("  ✓✓✓ All validations passed! Ready for CreateML")
    else:
        print("  ✗✗✗ Some validations failed. Please review errors above.")

    return all_valid


def main():
    """Main function to validate all dataset files."""
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(base_dir, 'datasets')

    train_path = os.path.join(datasets_dir, 'train.csv')
    val_path = os.path.join(datasets_dir, 'validation.csv')
    test_path = os.path.join(datasets_dir, 'test.csv')

    print("="*60)
    print("CreateML Dataset Validator")
    print("="*60)

    # Validate each file
    results = {}
    for name, path in [("Training", train_path), ("Validation", val_path), ("Test", test_path)]:
        results[name] = validate_dataset_file(path)

    # Final summary
    print("\n" + "="*60)
    print("FINAL VALIDATION SUMMARY")
    print("="*60)

    all_passed = all(results.values())

    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name} set: {status}")

    if all_passed:
        print("\n🎉 All datasets are ready for CreateML!")
        print("\nNext steps:")
        print("1. Open Xcode")
        print("2. Create new CreateML project")
        print("3. Select 'Tabular Classifier' template")
        print(f"4. Import training data: {train_path}")
        print(f"5. Import validation data: {val_path}")
        print("6. Set 'label' as target column")
        print("7. Set 'video_id' as metadata (exclude from training)")
        print("8. Train model")
        print(f"9. Evaluate on test data: {test_path}")
    else:
        print("\n⚠️  Some validations failed. Please fix errors and run again.")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
