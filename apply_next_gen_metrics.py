#!/usr/bin/env python3
"""
Apply Next-Generation ToM Metrics to Dataset
Creates dataset_v11 with sophisticated multiplicative and dynamic complexity measures
"""

import pandas as pd
import numpy as np
from next_generation_tom_metrics import analyze_text_next_gen
import warnings
warnings.filterwarnings('ignore')

def load_dataset():
    """Load the latest dataset"""
    print("Loading dataset_v10_ultimate...")
    df = pd.read_csv('./dataset_v10_ultimate.csv')
    df.columns = df.columns.str.strip()
    
    # Clean data
    df_clean = df[df['ABILITY'].notna()].copy()
    df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
    
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Categories: {df_clean['Main_Category'].unique()}")
    
    return df_clean

def apply_next_gen_metrics(df):
    """Apply next-generation metrics to stories and questions"""
    print(f"Processing {len(df)} rows with next-generation metrics...")
    
    # Initialize lists to store features
    story_features_list = []
    question_features_list = []
    
    # Process in batches for progress tracking
    batch_size = 100
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df))
        
        print(f"Processing batch {batch_idx + 1}/{total_batches} (rows {start_idx}-{end_idx})...")
        
        batch_story_features = []
        batch_question_features = []
        
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]
            
            # Analyze story
            story_text = row.get('STORY', '')
            story_features = analyze_text_next_gen(story_text, 'Story')
            batch_story_features.append(story_features)
            
            # Analyze question
            question_text = row.get('QUESTION', '')
            question_features = analyze_text_next_gen(question_text, 'Question')
            batch_question_features.append(question_features)
        
        story_features_list.extend(batch_story_features)
        question_features_list.extend(batch_question_features)
    
    # Convert to DataFrames
    story_features_df = pd.DataFrame(story_features_list)
    question_features_df = pd.DataFrame(question_features_list)
    
    print(f"Story features shape: {story_features_df.shape}")
    print(f"Question features shape: {question_features_df.shape}")
    
    return story_features_df, question_features_df

def combine_datasets(original_df, story_features_df, question_features_df):
    """Combine original dataset with new features"""
    print("Combining datasets...")
    
    # Reset indices to ensure alignment
    original_df = original_df.reset_index(drop=True)
    story_features_df = story_features_df.reset_index(drop=True)
    question_features_df = question_features_df.reset_index(drop=True)
    
    # Concatenate horizontally
    combined_df = pd.concat([original_df, story_features_df, question_features_df], axis=1)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df

def analyze_new_features(story_features_df, question_features_df):
    """Analyze the new next-generation features"""
    print("\n" + "="*60)
    print("NEXT-GENERATION FEATURES ANALYSIS")
    print("="*60)
    
    total_new_features = len(story_features_df.columns) + len(question_features_df.columns)
    print(f"Total new features: {total_new_features}")
    print(f"Story-level features: {len(story_features_df.columns)}")
    print(f"Question-level features: {len(question_features_df.columns)}")
    
    # Analyze feature categories
    story_categories = {}
    question_categories = {}
    
    # Categorize story features
    for col in story_features_df.columns:
        if 'Cognitive_Load_Index' in col or 'Mental_State_Interaction' in col or 'Inference_Complexity' in col:
            category = 'Multiplicative_Complexity'
        elif 'Complexity_Gradient' in col or 'Complexity_Variance' in col or 'Working_Memory' in col:
            category = 'Dynamic_Complexity'
        elif 'Recursive' in col or 'Meta_' in col or 'Cognitive_Interference' in col:
            category = 'Meta_Cognitive'
        elif 'Entity_Density' in col or 'Causal_Depth' in col or 'Uncertainty_Level' in col:
            category = 'Core_Dimensions'
        else:
            category = 'Component_Features'
        
        if category not in story_categories:
            story_categories[category] = []
        story_categories[category].append(col)
    
    # Categorize question features similarly
    for col in question_features_df.columns:
        if 'Cognitive_Load_Index' in col or 'Mental_State_Interaction' in col or 'Inference_Complexity' in col:
            category = 'Multiplicative_Complexity'
        elif 'Complexity_Gradient' in col or 'Complexity_Variance' in col or 'Working_Memory' in col:
            category = 'Dynamic_Complexity'
        elif 'Recursive' in col or 'Meta_' in col or 'Cognitive_Interference' in col:
            category = 'Meta_Cognitive'
        elif 'Entity_Density' in col or 'Causal_Depth' in col or 'Uncertainty_Level' in col:
            category = 'Core_Dimensions'
        else:
            category = 'Component_Features'
        
        if category not in question_categories:
            question_categories[category] = []
        question_categories[category].append(col)
    
    print(f"\nStory feature categories:")
    for category, features in story_categories.items():
        print(f"  {category}: {len(features)} features")
    
    print(f"\nQuestion feature categories:")
    for category, features in question_categories.items():
        print(f"  {category}: {len(features)} features")
    
    # Sample statistics for key multiplicative features
    print(f"\n=== KEY MULTIPLICATIVE FEATURES STATISTICS ===")
    
    # Story multiplicative features
    key_story_features = [
        'Story_Cognitive_Load_Index',
        'Story_Mental_State_Interaction',
        'Story_Inference_Complexity',
        'Story_Complexity_Gradient',
        'Story_Working_Memory_Load'
    ]
    
    for feature in key_story_features:
        if feature in story_features_df.columns:
            values = story_features_df[feature]
            non_zero_count = (values > 0).sum()
            non_zero_pct = non_zero_count / len(values) * 100
            print(f"{feature}:")
            print(f"  Mean: {values.mean():.6f}")
            print(f"  Std:  {values.std():.6f}")
            print(f"  Max:  {values.max():.6f}")
            print(f"  Non-zero: {non_zero_count} ({non_zero_pct:.1f}%)")
    
    # Question multiplicative features
    key_question_features = [
        'Question_Cognitive_Load_Index',
        'Question_Mental_State_Interaction',
        'Question_Inference_Complexity',
        'Question_Meta_Uncertainty_Index'
    ]
    
    for feature in key_question_features:
        if feature in question_features_df.columns:
            values = question_features_df[feature]
            non_zero_count = (values > 0).sum()
            non_zero_pct = non_zero_count / len(values) * 100
            print(f"{feature}:")
            print(f"  Mean: {values.mean():.6f}")
            print(f"  Std:  {values.std():.6f}")
            print(f"  Max:  {values.max():.6f}")
            print(f"  Non-zero: {non_zero_count} ({non_zero_pct:.1f}%)")

def main():
    """Main function"""
    print("Starting next-generation metrics application...")
    
    # Load dataset
    df = load_dataset()
    
    # Apply next-generation metrics
    story_features_df, question_features_df = apply_next_gen_metrics(df)
    
    # Combine datasets
    final_df = combine_datasets(df, story_features_df, question_features_df)
    
    # Analyze new features
    analyze_new_features(story_features_df, question_features_df)
    
    # Save the new dataset
    output_file = 'dataset_v11_next_gen.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")
    
    # Show first few rows of new features
    print(f"\n=== FIRST 3 ROWS OF NEW MULTIPLICATIVE FEATURES (sample) ===")
    multiplicative_cols = [col for col in story_features_df.columns if 'Index' in col or 'Interaction' in col or 'Complexity' in col][:5]
    if multiplicative_cols:
        print(final_df[multiplicative_cols].head(3))
    
    return final_df, story_features_df, question_features_df

if __name__ == "__main__":
    final_df, story_features_df, question_features_df = main()
