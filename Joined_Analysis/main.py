"""
Comprehensive Theory of Mind (ToM) Analysis Pipeline

This script runs all ToM analyses in sequence and outputs a single comprehensive dataframe:
1. Idea Density Analysis (DEPID)
2. RST Discourse Analysis  
3. Question Complexity Analysis
4. Answer Distinctiveness Analysis

Usage:
    python main.py

Output:
    comprehensive_tom_analysis.csv - Complete dataset with all analysis features
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

# Add the parent directory to the path to import from other modules
sys.path.append('..')

# Import all analyzer classes
from tom_analyzers import (
    IdeaDensityAnalyzer,
    RSTAnalyzer, 
    QuestionComplexityAnalyzer,
    AnswerDistinctivenessAnalyzer
)

def main():
    """Run comprehensive ToM analysis pipeline."""
    
    print("=" * 60)
    print("COMPREHENSIVE THEORY OF MIND ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Load the dataset
    print("\n1. Loading dataset...")
    try:
        df = pd.read_csv('../dataset.csv')
        print(f"✓ Dataset loaded: {len(df)} samples")
        print(f"✓ Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("✗ Error: dataset.csv not found in parent directory")
        return
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Initialize all analyzers
    print("\n2. Initializing analyzers...")
    
    try:
        # Initialize analyzers
        idea_analyzer = IdeaDensityAnalyzer()
        rst_analyzer = RSTAnalyzer(cuda_device=-1)  # Use CPU
        question_analyzer = QuestionComplexityAnalyzer()
        answer_analyzer = AnswerDistinctivenessAnalyzer()
        
        print("✓ All analyzers initialized successfully")
        
    except Exception as e:
        print(f"✗ Error initializing analyzers: {e}")
        return
    
    # Run analyses
    print("\n3. Running comprehensive analysis...")
    
    # Create results storage
    all_features = []
    
    # Process each row with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing samples"):
        
        sample_features = {'Index': idx}
        
        try:
            # 1. Idea Density Analysis (on STORY)
            if 'STORY' in df.columns:
                idea_features = idea_analyzer.analyze(row['STORY'])
                sample_features.update(idea_features)
            
            # 2. RST Analysis (on STORY)
            if 'STORY' in df.columns:
                rst_features = rst_analyzer.analyze(row['STORY'])
                sample_features.update(rst_features)
            
            # 3. Question Complexity Analysis (on QUESTION)
            if 'QUESTION' in df.columns:
                question_features = question_analyzer.analyze(row['QUESTION'])
                sample_features.update(question_features)
            
            # 4. Answer Distinctiveness Analysis (on OPTIONS)
            option_cols = ['OPTION-A', 'OPTION-B', 'OPTION-C', 'OPTION-D']
            if all(col in df.columns for col in option_cols):
                answer_features = answer_analyzer.analyze(
                    row['OPTION-A'], row['OPTION-B'], 
                    row['OPTION-C'], row['OPTION-D']
                )
                sample_features.update(answer_features)
            
        except Exception as e:
            print(f"\n⚠ Warning: Error processing sample {idx}: {e}")
            # Continue with next sample
            continue
        
        all_features.append(sample_features)
    
    # Convert results to DataFrame
    print("\n4. Consolidating results...")
    
    try:
        features_df = pd.DataFrame(all_features)
        
        # Merge with original dataset
        # Use Index column to align properly
        features_df = features_df.set_index('Index')
        df_with_features = pd.concat([df, features_df], axis=1)
        
        print(f"✓ Analysis complete: {len(df_with_features)} samples processed")
        print(f"✓ Total features: {len(df_with_features.columns)} columns")
        
    except Exception as e:
        print(f"✗ Error consolidating results: {e}")
        return
    
    # Organize columns logically
    print("\n5. Organizing output...")
    
    try:
        # Define column order for better readability
        base_cols = [col for col in df.columns if col in df_with_features.columns]
        
        # Group analysis features by type
        idea_cols = [col for col in df_with_features.columns if col.startswith(('Idea_', 'Word_Count'))]
        rst_cols = [col for col in df_with_features.columns if col.startswith('RST_')]
        question_cols = [col for col in df_with_features.columns if col.startswith(('Q_', 'Question_'))]
        answer_cols = [col for col in df_with_features.columns if col.startswith(('A_', 'Answer_'))]
        
        # Reorder columns
        ordered_cols = base_cols + idea_cols + rst_cols + question_cols + answer_cols
        
        # Remove duplicates while preserving order
        final_cols = []
        for col in ordered_cols:
            if col not in final_cols and col in df_with_features.columns:
                final_cols.append(col)
        
        df_final = df_with_features[final_cols]
        
        print(f"✓ Columns organized into logical groups")
        
    except Exception as e:
        print(f"⚠ Warning: Could not organize columns: {e}")
        df_final = df_with_features
    
    # Save results
    print("\n6. Saving results...")
    
    try:
        output_file = 'comprehensive_tom_analysis.csv'
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
        # Count non-null values for each analysis type
        idea_count = df_final['Idea_Density'].notna().sum() if 'Idea_Density' in df_final.columns else 0
        rst_count = df_final['RST_EDUs'].notna().sum() if 'RST_EDUs' in df_final.columns else 0
        question_count = df_final['Question_Complexity_Score'].notna().sum() if 'Question_Complexity_Score' in df_final.columns else 0
        answer_count = df_final['Answer_Distinctiveness_Score'].notna().sum() if 'Answer_Distinctiveness_Score' in df_final.columns else 0
        
        print(f"Idea Density Analysis:     {idea_count}/{len(df_final)} samples")
        print(f"RST Analysis:              {rst_count}/{len(df_final)} samples")
        print(f"Question Complexity:       {question_count}/{len(df_final)} samples")
        print(f"Answer Distinctiveness:    {answer_count}/{len(df_final)} samples")
        
        # Show key statistics for main scores
        key_metrics = [
            'Idea_Density', 'RST_EDUs', 'Question_Complexity_Score', 'Answer_Distinctiveness_Score'
        ]
        
        print("\nKey Metric Statistics:")
        print("-" * 40)
        
        for metric in key_metrics:
            if metric in df_final.columns and df_final[metric].notna().sum() > 0:
                mean_val = df_final[metric].mean()
                std_val = df_final[metric].std()
                min_val = df_final[metric].min()
                max_val = df_final[metric].max()
                
                print(f"{metric}:")
                print(f"  Mean: {mean_val:.3f} ± {std_val:.3f}")
                print(f"  Range: [{min_val:.3f}, {max_val:.3f}]")
                print()
        
    except Exception as e:
        print(f"⚠ Warning: Could not generate summary statistics: {e}")
    
    print("=" * 60)
    print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nOutput file: {output_file}")
    print("This file contains the original dataset plus all ToM analysis features.")
    print("\nFeature groups:")
    print("- Idea Density: Idea_Density, Word_Count")
    print("- RST Features: RST_EDUs, RST_Tree_Depth, RST_attribution, RST_causal, RST_explanation")
    print("- Question Complexity: Question_Complexity_Score, Q_* features")
    print("- Answer Distinctiveness: Answer_Distinctiveness_Score, A_* features")


if __name__ == "__main__":
    main()
