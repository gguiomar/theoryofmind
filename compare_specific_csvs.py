#!/usr/bin/env python3
"""
Script to compare column names of specific CSV files:
- dataset.csv
- dataset_with_idea_density.csv
- test_output.csv
"""

import pandas as pd
import os
from collections import Counter

def load_csv_columns(file_path):
    """Load CSV file and return its column names."""
    try:
        df = pd.read_csv(file_path, nrows=0)
        return list(df.columns)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    # Define the specific CSV files to analyze
    csv_files = {
        'dataset.csv': './dataset.csv',
        'dataset_with_idea_density.csv': './Idea_Density/dataset_with_idea_density.csv',
        'test_output.csv': './RST/test_output.csv'
    }
    
    print("COMPARING SPECIFIC CSV FILES")
    print("=" * 60)
    
    # Load column data for each file
    csv_data = {}
    for name, path in csv_files.items():
        if os.path.exists(path):
            columns = load_csv_columns(path)
            csv_data[name] = columns
            print(f"\n{name}:")
            print(f"  Path: {path}")
            if columns:
                print(f"  Columns ({len(columns)}): {columns}")
            else:
                print("  ERROR: Could not read file")
        else:
            print(f"\n{name}:")
            print(f"  Path: {path}")
            print("  ERROR: File not found")
    
    # Filter out files that couldn't be read
    valid_files = {k: v for k, v in csv_data.items() if v is not None}
    
    if len(valid_files) < 2:
        print("\nNeed at least 2 valid files to compare!")
        return
    
    # Find all unique columns
    all_columns = set()
    for columns in valid_files.values():
        all_columns.update(columns)
    
    # Find common columns (present in ALL files)
    common_columns = set(list(valid_files.values())[0])
    for columns in valid_files.values():
        common_columns = common_columns.intersection(set(columns))
    
    print(f"\nCOLUMN ANALYSIS:")
    print("=" * 60)
    print(f"Total unique columns across all files: {len(all_columns)}")
    print(f"Columns common to ALL files: {len(common_columns)}")
    
    if common_columns:
        print(f"Common columns: {sorted(list(common_columns))}")
    else:
        print("No columns are common to all files")
    
    # Show unique columns per file
    print(f"\nUNIQUE COLUMNS PER FILE:")
    print("-" * 40)
    for file_name, columns in valid_files.items():
        unique_to_file = set(columns) - common_columns
        if unique_to_file:
            print(f"\n{file_name}:")
            print(f"  Unique columns: {sorted(list(unique_to_file))}")
        else:
            print(f"\n{file_name}: No unique columns")
    
    # Column frequency analysis
    all_columns_list = []
    for columns in valid_files.values():
        all_columns_list.extend(columns)
    
    column_counts = Counter(all_columns_list)
    
    print(f"\nCOLUMN FREQUENCY:")
    print("-" * 40)
    for col, count in sorted(column_counts.items()):
        files_with_col = [name for name, cols in valid_files.items() if col in cols]
        print(f"'{col}': {count}/{len(valid_files)} files ({', '.join(files_with_col)})")
    
    # Detailed comparison matrix
    print(f"\nCOLUMN PRESENCE MATRIX:")
    print("-" * 40)
    print(f"{'Column':<30} | ", end="")
    for file_name in valid_files.keys():
        print(f"{file_name[:15]:<17}", end="")
    print()
    print("-" * (30 + 2 + 17 * len(valid_files)))
    
    for col in sorted(all_columns):
        print(f"{col[:29]:<30} | ", end="")
        for file_name, columns in valid_files.items():
            present = "✓" if col in columns else "✗"
            print(f"{present:<17}", end="")
        print()

if __name__ == "__main__":
    main()
