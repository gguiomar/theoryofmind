#!/usr/bin/env python3
"""
Apply advanced ToM metrics to dataset_v3 and create dataset_v4
"""

import pandas as pd
import numpy as np
from advanced_tom_metrics import analyze_text_for_advanced_tom
import warnings
warnings.filterwarnings('ignore')

def apply_advanced_tom_metrics():
    """Apply advanced ToM metrics to dataset_v3."""
    print("Loading dataset_v3...")
    df = pd.read_csv('dataset_v3.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Processing {len(df)} rows...")
    
    # Initialize lists to store results
    story_features_list = []
    question_features_list = []
    
    # Process each row
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"Processing row {idx + 1}/{len(df)}...")
        
        # Analyze story
        story = row['STORY']
        story_features = analyze_text_for_advanced_tom(story, 'Story')
        story_features_list.append(story_features)
        
        # Analyze question
        question = row['QUESTION']
        question_features = analyze_text_for_advanced_tom(question, 'Q')
        question_features_list.append(question_features)
    
    # Convert to DataFrames
    story_features_df = pd.DataFrame(story_features_list)
    question_features_df = pd.DataFrame(question_features_list)
    
    # Combine all features
    advanced_features_df = pd.concat([story_features_df, question_features_df], axis=1)
    
    # Combine with original dataset
    df_v4 = pd.concat([df, advanced_features_df], axis=1)
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Advanced features shape: {advanced_features_df.shape}")
    print(f"Final dataset_v4 shape: {df_v4.shape}")
    
    # Save the enhanced dataset
    output_path = 'dataset_v4.csv'
    df_v4.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    
    # Print summary of new features
    print("\n=== NEW ADVANCED TOM FEATURES SUMMARY ===")
    
    new_cols = advanced_features_df.columns.tolist()
    print(f"Total new features: {len(new_cols)}")
    
    # Group by feature type
    story_features = [col for col in new_cols if col.startswith('Story_')]
    question_features = [col for col in new_cols if col.startswith('Q_')]
    
    print(f"Story-level features: {len(story_features)}")
    print(f"Question-level features: {len(question_features)}")
    
    print("\nStory feature categories:")
    for category in ['MS_Embedding', 'Perspective', 'Recursive', 'Coref', 'False_Belief', 'MS_Arg', 'Temporal']:
        category_features = [col for col in story_features if category in col]
        if category_features:
            print(f"  {category}: {len(category_features)} features")
    
    print("\nQuestion feature categories:")
    for category in ['MS_Embedding', 'Perspective', 'Recursive', 'Coref', 'False_Belief', 'MS_Arg', 'Temporal']:
        category_features = [col for col in question_features if category in col]
        if category_features:
            print(f"  {category}: {len(category_features)} features")
    
    # Show some sample statistics
    print("\n=== SAMPLE FEATURE STATISTICS ===")
    
    key_features = [
        'Story_MS_Embedding_Max_Depth',
        'Story_Recursive_MS_Max_Depth', 
        'Story_False_Belief_Score',
        'Q_MS_Embedding_Max_Depth',
        'Q_Perspective_Complexity',
        'Q_False_Belief_Score'
    ]
    
    for feature in key_features:
        if feature in df_v4.columns:
            values = df_v4[feature]
            print(f"{feature}:")
            print(f"  Mean: {values.mean():.3f}")
            print(f"  Std:  {values.std():.3f}")
            print(f"  Max:  {values.max()}")
            print(f"  Non-zero: {(values > 0).sum()} ({(values > 0).sum()/len(values)*100:.1f}%)")
    
    return df_v4

if __name__ == "__main__":
    dataset_v4 = apply_advanced_tom_metrics()
    
    print("\n=== FIRST 3 ROWS OF NEW FEATURES ===")
    new_feature_cols = [col for col in dataset_v4.columns if col.startswith(('Story_', 'Q_')) and 'MS_' in col or 'Perspective' in col or 'Recursive' in col][:10]
    print(dataset_v4[new_feature_cols].head(3))
