#!/usr/bin/env python3
"""
Fix the cleaning script to preserve the Meta Llama model and recreate dataset_joined.csv
"""

import pandas as pd
import os

def clean_dataset_properly(file_path, output_path):
    """Remove only columns that END with .1, not those that contain .1"""
    try:
        df = pd.read_csv(file_path)
        
        # Find columns that END with .1 (not just contain .1)
        columns_to_remove = [col for col in df.columns if col.endswith('.1')]
        
        if columns_to_remove:
            print(f"Removing columns from {file_path}: {columns_to_remove}")
            df_cleaned = df.drop(columns=columns_to_remove)
        else:
            print(f"No columns ending with .1 found in {file_path}")
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
    
    print("FIXING DATASET CLEANING - PRESERVING META LLAMA MODEL")
    print("=" * 60)
    
    cleaned_datasets = {}
    
    # Process each file
    for name, path in csv_files.items():
        if os.path.exists(path):
            print(f"\nProcessing {name}...")
            
            # Create output path
            base_name = name.replace('.csv', '')
            output_path = f'./{base_name}_fixed.csv'
            
            # Clean the dataset properly
            cleaned_df = clean_dataset_properly(path, output_path)
            
            if cleaned_df is not None:
                cleaned_datasets[name] = {
                    'dataframe': cleaned_df,
                    'output_path': output_path,
                    'columns': list(cleaned_df.columns)
                }
        else:
            print(f"File not found: {path}")
    
    # Check for Meta Llama model
    print(f"\nCHECKING FOR META LLAMA MODEL:")
    print("-" * 40)
    
    for name, info in cleaned_datasets.items():
        llama_cols = [col for col in info['columns'] if 'llama' in col.lower() or 'meta' in col.lower()]
        if llama_cols:
            print(f"✓ {name}: Found Meta Llama model: {llama_cols}")
        else:
            print(f"✗ {name}: No Meta Llama model found")
    
    # Use the best dataset (dataset.csv) as the base for the new joined analysis
    if 'dataset.csv' in cleaned_datasets:
        base_df = cleaned_datasets['dataset.csv']['dataframe']
        print(f"\nUsing dataset_fixed.csv as base for joined analysis")
        print(f"Model columns found:")
        model_cols = [col for col in base_df.columns if any(word in col.lower() for word in ['llama', 'qwen', 'olmo', 'mistral', 'phi', 'internlm', 'meta'])]
        for col in model_cols:
            print(f"  - {col}")

if __name__ == "__main__":
    main()
