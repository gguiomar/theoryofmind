#!/usr/bin/env python3
"""
Script to create dataset_v3 by combining columns from main_msvd and dataset_joined_corrected
"""

import pandas as pd
import numpy as np

def create_dataset_v3():
    """
    Combines columns from main_msvd and dataset_joined_corrected to create dataset_v3
    
    Strategy:
    1. Use dataset_joined_corrected as base (more comprehensive, 50 columns)
    2. Add unique columns from main_msvd (11 unique columns)
    3. Handle column name conflicts and clean up naming
    4. Create comprehensive dataset_v3 with all features
    """
    
    print("Loading datasets...")
    
    # Load both datasets
    df_main_msvd = pd.read_csv('main_with_msvd.csv')
    df_joined = pd.read_csv('Joined_Analysis/dataset_joined_corrected.csv')
    
    print(f"main_msvd shape: {df_main_msvd.shape}")
    print(f"dataset_joined_corrected shape: {df_joined.shape}")
    
    # Verify both datasets have same number of rows
    assert len(df_main_msvd) == len(df_joined), "Datasets must have same number of rows"
    
    # Start with dataset_joined_corrected as base
    df_v3 = df_joined.copy()
    
    # Clean up column names in base dataset (remove \n prefixes)
    column_mapping = {}
    for col in df_v3.columns:
        if col.startswith('\n'):
            new_col = col[1:]  # Remove \n prefix
            column_mapping[col] = new_col
    
    df_v3 = df_v3.rename(columns=column_mapping)
    
    print(f"Cleaned column names. Base dataset now has {len(df_v3.columns)} columns")
    
    # Identify unique columns from main_msvd to add
    main_msvd_cols = set(df_main_msvd.columns)
    joined_cols = set(df_v3.columns)
    
    # Columns unique to main_msvd that we want to add
    unique_cols_to_add = main_msvd_cols - joined_cols
    
    print(f"Adding {len(unique_cols_to_add)} unique columns from main_msvd:")
    print(f"Columns to add: {sorted(list(unique_cols_to_add))}")
    
    # Add unique columns from main_msvd
    for col in unique_cols_to_add:
        df_v3[col] = df_main_msvd[col]
    
    # Reorder columns for better organization
    # Put core identification columns first
    core_cols = ['Unnamed: 0', 'ABILITY', 'TASK', 'INDEX', 'STORY']
    
    # Question and answer columns
    qa_cols = ['QUESTION'] + [col for col in df_v3.columns if col.startswith('Q_')] + \
              ['Question_Complexity_Score'] + \
              ['OPTION-A', 'OPTION-B', 'OPTION-C', 'OPTION-D', 'ANSWER']
    
    # Model response columns
    model_cols = [col for col in df_v3.columns if any(model in col for model in 
                  ['meta_llama', 'Qwen', 'allenai', 'mistralai', 'microsoft', 'internlm'])]
    
    # Analysis columns
    analysis_cols = ['Idea_Density', 'Word_Count'] + \
                   [col for col in df_v3.columns if col.startswith('rel_')] + \
                   ['num_edus', 'tree_depth', 'RST_tree_depth'] + \
                   ['Volition', 'Cognition', 'Emotion'] + \
                   ['Volition_MSVD', 'Cognition_MSVD', 'Emotion_MSVD', 'Overall_MSVD']
    
    # Collect all organized columns
    organized_cols = []
    for col_group in [core_cols, qa_cols, model_cols, analysis_cols]:
        for col in col_group:
            if col in df_v3.columns and col not in organized_cols:
                organized_cols.append(col)
    
    # Add any remaining columns
    for col in df_v3.columns:
        if col not in organized_cols:
            organized_cols.append(col)
    
    # Reorder the dataframe
    df_v3 = df_v3[organized_cols]
    
    print(f"\nFinal dataset_v3 shape: {df_v3.shape}")
    print(f"Total columns: {len(df_v3.columns)}")
    
    # Save the combined dataset
    output_path = 'dataset_v3.csv'
    df_v3.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    
    # Display summary information
    print("\n=== DATASET_V3 SUMMARY ===")
    print(f"Rows: {len(df_v3)}")
    print(f"Columns: {len(df_v3.columns)}")
    
    print("\nColumn categories:")
    print(f"- Core identification: {len([c for c in core_cols if c in df_v3.columns])}")
    print(f"- Question analysis: {len([c for c in qa_cols if c in df_v3.columns])}")
    print(f"- Model responses: {len(model_cols)}")
    print(f"- Text analysis: {len([c for c in analysis_cols if c in df_v3.columns])}")
    
    print(f"\nFirst few columns: {list(df_v3.columns[:10])}")
    print(f"Last few columns: {list(df_v3.columns[-10:])}")
    
    return df_v3

if __name__ == "__main__":
    dataset_v3 = create_dataset_v3()
    
    # Display first few rows to verify
    print("\n=== FIRST 3 ROWS OF DATASET_V3 ===")
    print(dataset_v3.head(3))
