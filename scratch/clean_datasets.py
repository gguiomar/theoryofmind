#!/usr/bin/env python3
"""
Script to remove columns with .1 in their names from all datasets and create dataset_new.csv
"""

import pandas as pd
import os

def clean_dataset(file_path, output_path):
    """Remove columns with .1 in their names and save cleaned dataset."""
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Find columns with .1 in their names
        columns_to_remove = [col for col in df.columns if '.1' in col]
        
        if columns_to_remove:
            print(f"Removing columns from {file_path}: {columns_to_remove}")
            # Remove the columns
            df_cleaned = df.drop(columns=columns_to_remove)
        else:
            print(f"No columns with .1 found in {file_path}")
            df_cleaned = df.copy()
        
        # Save the cleaned dataset
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to: {output_path}")
        print(f"Original columns: {len(df.columns)}, Cleaned columns: {len(df_cleaned.columns)}")
        
        return df_cleaned
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    # Define the CSV files to process
    csv_files = {
        'dataset.csv': './dataset.csv',
        'dataset_with_idea_density.csv': './Idea_Density/dataset_with_idea_density.csv',
        'test_output.csv': './RST/test_output.csv'
    }
    
    print("CLEANING DATASETS - REMOVING COLUMNS WITH .1")
    print("=" * 60)
    
    cleaned_datasets = {}
    
    # Process each file
    for name, path in csv_files.items():
        if os.path.exists(path):
            print(f"\nProcessing {name}...")
            
            # Create output path
            base_name = name.replace('.csv', '')
            output_path = f'./{base_name}_new.csv'
            
            # Clean the dataset
            cleaned_df = clean_dataset(path, output_path)
            
            if cleaned_df is not None:
                cleaned_datasets[name] = {
                    'dataframe': cleaned_df,
                    'output_path': output_path,
                    'columns': list(cleaned_df.columns)
                }
        else:
            print(f"File not found: {path}")
    
    # Summary of cleaned datasets
    print(f"\nSUMMARY OF CLEANED DATASETS:")
    print("=" * 60)
    
    for name, info in cleaned_datasets.items():
        print(f"\n{name} -> {os.path.basename(info['output_path'])}")
        print(f"  Columns ({len(info['columns'])}): {info['columns'][:5]}..." if len(info['columns']) > 5 else f"  Columns ({len(info['columns'])}): {info['columns']}")
    
    # Check if all cleaned datasets have the same columns
    if len(cleaned_datasets) > 1:
        column_sets = [set(info['columns']) for info in cleaned_datasets.values()]
        common_columns = set.intersection(*column_sets)
        
        print(f"\nCOLUMN CONSISTENCY CHECK:")
        print("-" * 40)
        print(f"Common columns across all cleaned files: {len(common_columns)}")
        
        all_same = all(len(cols) == len(common_columns) for cols in column_sets)
        if all_same:
            print("✓ All cleaned datasets have identical column structure")
        else:
            print("✗ Cleaned datasets have different column structures")
            for name, info in cleaned_datasets.items():
                unique_cols = set(info['columns']) - common_columns
                if unique_cols:
                    print(f"  {name} has unique columns: {list(unique_cols)}")

if __name__ == "__main__":
    main()
