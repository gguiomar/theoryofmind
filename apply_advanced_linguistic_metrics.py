#!/usr/bin/env python3
"""
Apply Advanced Linguistic Metrics to Dataset
Creates dataset_v13 with syntactic, semantic, and embedding-based metrics
"""

import pandas as pd
import numpy as np
from advanced_linguistic_metrics import analyze_text_advanced
import warnings
warnings.filterwarnings('ignore')

def load_dataset():
    """Load the latest dataset"""
    print("Loading dataset_v12_final_universal...")
    df = pd.read_csv('./dataset_v12_final_universal.csv')
    df.columns = df.columns.str.strip()
    
    # Clean data
    df_clean = df[df['ABILITY'].notna()].copy()
    df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
    
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Categories: {df_clean['Main_Category'].unique()}")
    
    return df_clean

def apply_advanced_linguistic_metrics(df):
    """Apply advanced linguistic metrics to stories and questions"""
    print(f"Processing {len(df)} rows with advanced linguistic metrics...")
    
    # Initialize lists to store features
    story_features_list = []
    question_features_list = []
    
    # Process in batches for progress tracking
    batch_size = 50  # Smaller batches due to computational complexity
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
            try:
                story_features = analyze_text_advanced(story_text, 'Story')
            except Exception as e:
                print(f"Error analyzing story at row {idx}: {e}")
                story_features = get_empty_advanced_features('Story')
            batch_story_features.append(story_features)
            
            # Analyze question
            question_text = row.get('QUESTION', '')
            try:
                question_features = analyze_text_advanced(question_text, 'Question')
            except Exception as e:
                print(f"Error analyzing question at row {idx}: {e}")
                question_features = get_empty_advanced_features('Question')
            batch_question_features.append(question_features)
        
        story_features_list.extend(batch_story_features)
        question_features_list.extend(batch_question_features)
    
    # Convert to DataFrames
    story_features_df = pd.DataFrame(story_features_list)
    question_features_df = pd.DataFrame(question_features_list)
    
    print(f"Story features shape: {story_features_df.shape}")
    print(f"Question features shape: {question_features_df.shape}")
    
    return story_features_df, question_features_df

def get_empty_advanced_features(text_type):
    """Return empty advanced features for error cases"""
    return {
        # Syntactic features
        f'{text_type}_Parse_Max_Depth': 0,
        f'{text_type}_Parse_Avg_Depth': 0,
        f'{text_type}_Parse_Total_Depth': 0,
        f'{text_type}_Constituency_Depth': 0,
        f'{text_type}_Clause_Count': 0,
        f'{text_type}_Subordinate_Clauses': 0,
        f'{text_type}_Coordinate_Clauses': 0,
        f'{text_type}_Yngve_Depth': 0,
        f'{text_type}_Avg_Arc_Length': 0,
        f'{text_type}_Max_Arc_Length': 0,
        
        # Semantic features
        f'{text_type}_Verb_Argument_Complexity': 0,
        f'{text_type}_Predicate_Count': 0,
        f'{text_type}_Entity_Relation_Depth': 0,
        f'{text_type}_Semantic_Mental_State_Depth': 0,
        f'{text_type}_Causal_Chain_Depth': 0,
        f'{text_type}_Temporal_Semantic_Complexity': 0,
        
        # Embedding features
        f'{text_type}_Embedding_Variance': 0,
        f'{text_type}_Embedding_Std': 0,
        f'{text_type}_Semantic_Coherence_Avg': 0,
        f'{text_type}_Semantic_Coherence_Min': 0,
        f'{text_type}_Embedding_Trajectory_Length': 0,
        f'{text_type}_PCA_First_Component': 0,
        f'{text_type}_PCA_Cumulative_Variance': 0,
        f'{text_type}_Avg_Distance_To_Centroid': 0,
        f'{text_type}_Max_Distance_To_Centroid': 0,
        
        # Readability features
        f'{text_type}_Flesch_Reading_Ease': 0,
        f'{text_type}_Flesch_Kincaid_Grade': 0,
        f'{text_type}_Gunning_Fog': 0,
        f'{text_type}_SMOG_Index': 0,
        f'{text_type}_Automated_Readability': 0,
        f'{text_type}_Coleman_Liau': 0,
        f'{text_type}_Difficult_Words': 0,
        f'{text_type}_Dale_Chall': 0,
        f'{text_type}_Linsear_Write': 0
    }

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
    """Analyze the new advanced linguistic features"""
    print("\n" + "="*60)
    print("ADVANCED LINGUISTIC FEATURES ANALYSIS")
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
        if any(x in col for x in ['Parse', 'Constituency', 'Clause', 'Yngve', 'Arc']):
            category = 'Syntactic_Complexity'
        elif any(x in col for x in ['Verb_Argument', 'Predicate', 'Entity_Relation', 'Semantic_Mental', 'Causal_Chain', 'Temporal_Semantic']):
            category = 'Semantic_Complexity'
        elif any(x in col for x in ['Embedding', 'Coherence', 'Trajectory', 'PCA', 'Distance']):
            category = 'Embedding_Complexity'
        elif any(x in col for x in ['Flesch', 'Gunning', 'SMOG', 'Readability', 'Dale_Chall']):
            category = 'Readability_Complexity'
        else:
            category = 'Other_Features'
        
        if category not in story_categories:
            story_categories[category] = []
        story_categories[category].append(col)
    
    # Categorize question features similarly
    for col in question_features_df.columns:
        if any(x in col for x in ['Parse', 'Constituency', 'Clause', 'Yngve', 'Arc']):
            category = 'Syntactic_Complexity'
        elif any(x in col for x in ['Verb_Argument', 'Predicate', 'Entity_Relation', 'Semantic_Mental', 'Causal_Chain', 'Temporal_Semantic']):
            category = 'Semantic_Complexity'
        elif any(x in col for x in ['Embedding', 'Coherence', 'Trajectory', 'PCA', 'Distance']):
            category = 'Embedding_Complexity'
        elif any(x in col for x in ['Flesch', 'Gunning', 'SMOG', 'Readability', 'Dale_Chall']):
            category = 'Readability_Complexity'
        else:
            category = 'Other_Features'
        
        if category not in question_categories:
            question_categories[category] = []
        question_categories[category].append(col)
    
    print(f"\nStory feature categories:")
    for category, features in story_categories.items():
        print(f"  {category}: {len(features)} features")
    
    print(f"\nQuestion feature categories:")
    for category, features in question_categories.items():
        print(f"  {category}: {len(features)} features")
    
    # Sample statistics for key advanced features
    print(f"\n=== KEY ADVANCED FEATURES STATISTICS ===")
    
    # Story advanced features
    key_story_features = [
        'Story_Parse_Max_Depth',
        'Story_Constituency_Depth',
        'Story_Verb_Argument_Complexity',
        'Story_Semantic_Mental_State_Depth',
        'Story_Embedding_Variance',
        'Story_Semantic_Coherence_Avg'
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
    
    # Question advanced features
    key_question_features = [
        'Question_Parse_Max_Depth',
        'Question_Constituency_Depth',
        'Question_Verb_Argument_Complexity',
        'Question_Semantic_Mental_State_Depth'
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
    print("Starting advanced linguistic metrics application...")
    
    # Load dataset
    df = load_dataset()
    
    # Apply advanced linguistic metrics
    story_features_df, question_features_df = apply_advanced_linguistic_metrics(df)
    
    # Combine datasets
    final_df = combine_datasets(df, story_features_df, question_features_df)
    
    # Analyze new features
    analyze_new_features(story_features_df, question_features_df)
    
    # Save the new dataset
    output_file = 'dataset_v13_advanced_linguistic.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")
    
    # Show first few rows of new features
    print(f"\n=== FIRST 3 ROWS OF NEW ADVANCED FEATURES (sample) ===")
    advanced_cols = [col for col in story_features_df.columns if any(x in col for x in ['Parse', 'Semantic', 'Embedding'])][:5]
    if advanced_cols:
        print(final_df[advanced_cols].head(3))
    
    return final_df, story_features_df, question_features_df

if __name__ == "__main__":
    final_df, story_features_df, question_features_df = main()
