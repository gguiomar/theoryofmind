#!/usr/bin/env python3
"""
Run Question Complexity Analysis and create dataset_joined.csv

This script:
1. Loads one of the cleaned datasets
2. Runs question complexity analysis using the existing QuestionComplexityAnalyzer
3. Saves the results as dataset_joined.csv in the Joined_Analysis folder
"""

import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm

# Import the QuestionComplexityAnalyzer from tom_analyzers
from tom_analyzers import QuestionComplexityAnalyzer

def main():
    """Run question complexity analysis and create joined dataset."""
    
    print("=" * 60)
    print("QUESTION COMPLEXITY ANALYSIS FOR JOINED DATASET")
    print("=" * 60)
    
    # Load one of the cleaned datasets (using dataset_new.csv as the base)
    print("\n1. Loading cleaned dataset...")
    try:
        df = pd.read_csv('../dataset_new.csv')
        print(f"✓ Dataset loaded: {len(df)} samples")
        print(f"✓ Columns: {len(df.columns)}")
        print(f"✓ Sample columns: {list(df.columns[:5])}...")
    except FileNotFoundError:
        print("✗ Error: dataset_new.csv not found in parent directory")
        print("Available files:")
        for file in os.listdir('..'):
            if file.endswith('.csv'):
                print(f"  - {file}")
        return
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Check if QUESTION column exists
    if 'QUESTION' not in df.columns:
        print("✗ Error: QUESTION column not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Initialize Question Complexity Analyzer
    print("\n2. Initializing Question Complexity Analyzer...")
    try:
        question_analyzer = QuestionComplexityAnalyzer()
        print("✓ Question Complexity Analyzer initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing analyzer: {e}")
        return
    
    # Run question complexity analysis
    print("\n3. Running question complexity analysis...")
    
    # Create results storage
    all_features = []
    
    # Process each row with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing questions"):
        
        try:
            # Run question complexity analysis
            question_features = question_analyzer.analyze(row['QUESTION'])
            
            # Add index for alignment
            question_features['Index'] = idx
            all_features.append(question_features)
            
        except Exception as e:
            print(f"\n⚠ Warning: Error processing question {idx}: {e}")
            # Add empty features for this sample
            empty_features = question_analyzer._get_empty_features()
            empty_features['Index'] = idx
            all_features.append(empty_features)
            continue
    
    # Convert results to DataFrame
    print("\n4. Consolidating results...")
    
    try:
        features_df = pd.DataFrame(all_features)
        
        # Set index for proper alignment
        features_df = features_df.set_index('Index')
        
        # Merge with original dataset
        df_with_features = pd.concat([df, features_df], axis=1)
        
        print(f"✓ Analysis complete: {len(df_with_features)} samples processed")
        print(f"✓ Total features: {len(df_with_features.columns)} columns")
        
        # Show which question complexity features were added
        question_cols = [col for col in features_df.columns if col.startswith(('Q_', 'Question_'))]
        print(f"✓ Question complexity features added: {len(question_cols)}")
        
    except Exception as e:
        print(f"✗ Error consolidating results: {e}")
        return
    
    # Organize columns logically
    print("\n5. Organizing output...")
    
    try:
        # Get original columns
        original_cols = list(df.columns)
        
        # Get question complexity columns
        question_cols = [col for col in df_with_features.columns if col.startswith(('Q_', 'Question_'))]
        
        # Find the position of QUESTION column to insert complexity features after it
        question_index = original_cols.index('QUESTION')
        
        # Create new column order: original columns + question complexity features inserted after QUESTION
        new_cols = original_cols[:question_index + 1] + question_cols + original_cols[question_index + 1:]
        
        # Remove duplicates while preserving order
        final_cols = []
        for col in new_cols:
            if col not in final_cols and col in df_with_features.columns:
                final_cols.append(col)
        
        df_final = df_with_features[final_cols]
        
        print(f"✓ Columns organized: question complexity features placed after QUESTION column")
        
    except Exception as e:
        print(f"⚠ Warning: Could not organize columns: {e}")
        df_final = df_with_features
    
    # Save results
    print("\n6. Saving results...")
    
    try:
        output_file = 'dataset_joined.csv'
        df_final.to_csv(output_file, index=False)
        
        print(f"✓ Results saved to: {output_file}")
        print(f"✓ File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"✗ Error saving results: {e}")
        return
    
    # Print summary statistics
    print("\n7. Analysis Summary:")
    print("=" * 40)
    
    try:
        # Count successful analyses
        complexity_score_col = 'Question_Complexity_Score'
        if complexity_score_col in df_final.columns:
            successful_analyses = df_final[complexity_score_col].notna().sum()
            print(f"Successful question complexity analyses: {successful_analyses}/{len(df_final)}")
            
            if successful_analyses > 0:
                # Show statistics for main complexity scores
                complexity_metrics = [
                    'Question_Complexity_Score',
                    'Q_Syntactic_Complexity', 
                    'Q_Semantic_Complexity',
                    'Q_ToM_Complexity',
                    'Q_Reasoning_Complexity'
                ]
                
                print("\nQuestion Complexity Statistics:")
                print("-" * 40)
                
                for metric in complexity_metrics:
                    if metric in df_final.columns and df_final[metric].notna().sum() > 0:
                        mean_val = df_final[metric].mean()
                        std_val = df_final[metric].std()
                        min_val = df_final[metric].min()
                        max_val = df_final[metric].max()
                        
                        print(f"{metric}:")
                        print(f"  Mean: {mean_val:.3f} ± {std_val:.3f}")
                        print(f"  Range: [{min_val:.3f}, {max_val:.3f}]")
                        print()
        
        # Show sample of question complexity features
        question_feature_cols = [col for col in df_final.columns if col.startswith(('Q_', 'Question_'))]
        print(f"Question Complexity Features Added ({len(question_feature_cols)}):")
        print("-" * 50)
        for i, col in enumerate(question_feature_cols[:10]):  # Show first 10
            print(f"  {col}")
        if len(question_feature_cols) > 10:
            print(f"  ... and {len(question_feature_cols) - 10} more")
        
    except Exception as e:
        print(f"⚠ Warning: Could not generate summary statistics: {e}")
    
    print("\n" + "=" * 60)
    print("QUESTION COMPLEXITY ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nOutput file: {output_file}")
    print("This file contains the original cleaned dataset plus question complexity features.")
    print("\nQuestion Complexity Features include:")
    print("- Overall Score: Question_Complexity_Score")
    print("- Dimension Scores: Q_Syntactic_Complexity, Q_Semantic_Complexity, Q_ToM_Complexity, Q_Reasoning_Complexity")
    print("- Detailed Features: Q_Length, Q_Word_Count, Q_Mental_State_Verbs, Q_Perspective_Markers, etc.")


if __name__ == "__main__":
    main()
