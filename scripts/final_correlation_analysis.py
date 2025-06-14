#!/usr/bin/env python3
"""
Final Correlation Analysis: Universal ToM Complexity Features
Identifies features significantly correlated with performance across humans and LLMs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_data():
    """Load comprehensive features and original dataset, merge for analysis"""
    
    print("Loading datasets...")
    
    # Load comprehensive features
    df_features = pd.read_csv('../data/final/dataset_v14_comprehensive_advanced.csv')
    df_features.columns = df_features.columns.str.strip()
    
    # Load original dataset for performance calculation
    df_original = pd.read_csv('../data/original/dataset.csv')
    df_original.columns = df_original.columns.str.strip()
    
    print(f"Comprehensive features dataset: {df_features.shape}")
    print(f"Original dataset: {df_original.shape}")
    
    # Clean and prepare data
    df_clean = df_features[df_features['ABILITY'].notna()].copy()
    df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
    
    print(f"Clean dataset: {df_clean.shape}")
    print(f"Categories: {df_clean['Main_Category'].value_counts().to_dict()}")
    
    return df_clean, df_original

def extract_performance_data(df_original):
    """Extract performance data for humans and models"""
    
    print("\nExtracting performance data...")
    
    # Human performance by category (from performance_matrix_pdf.py)
    human_performance = {
        'Emotion': 86.4,
        'Desire': 90.4,
        'Intention': 82.2,
        'Knowledge': 89.3,
        'Belief': 89.0,
        'NLC': 86.1
    }
    
    # Model columns
    models = [
        'meta_llama_Llama_3.1_70B_Instruct',
        'Qwen_Qwen2.5_32B_Instruct', 
        'allenai_OLMo_2_1124_13B_Instruct',
        'mistralai_Mistral_7B_Instruct_v0.3',
        'microsoft_Phi_3_mini_4k_instruct',
        'internlm_internlm2_5_1_8b_chat'
    ]
    
    model_display_names = {
        'meta_llama_Llama_3.1_70B_Instruct': 'Llama3.1_70B',
        'Qwen_Qwen2.5_32B_Instruct': 'Qwen2.5_32B',
        'allenai_OLMo_2_1124_13B_Instruct': 'OLMo_13B',
        'mistralai_Mistral_7B_Instruct_v0.3': 'Mistral_7B',
        'microsoft_Phi_3_mini_4k_instruct': 'Phi3_Mini',
        'internlm_internlm2_5_1_8b_chat': 'InternLM_1.8B'
    }
    
    # Prepare original data
    df_orig_clean = df_original[df_original['ABILITY'].notna()].copy()
    df_orig_clean['Main_Category'] = df_orig_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_orig_clean['Main_Category'] = df_orig_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
    
    # Fix ANSWER column name
    answer_col = '\nANSWER' if '\nANSWER' in df_orig_clean.columns else 'ANSWER'
    
    # Calculate model performance by category
    def calculate_accuracy(df, model_col, category):
        subset = df[df['Main_Category'] == category]
        if len(subset) == 0:
            return 0
        correct = (subset[model_col] == subset[answer_col]).sum()
        return (correct / len(subset) * 100)
    
    categories = list(human_performance.keys())
    
    # Create performance matrix
    performance_data = {}
    performance_data['Human'] = human_performance
    
    for model in models:
        if model in df_orig_clean.columns:
            display_name = model_display_names[model]
            performance_data[display_name] = {
                cat: calculate_accuracy(df_orig_clean, model, cat) 
                for cat in categories
            }
    
    print(f"Performance data extracted for: {list(performance_data.keys())}")
    
    return performance_data

def calculate_correlations_with_significance(df_clean, performance_data):
    """Calculate correlations between features and performance with significance testing"""
    
    print("\nCalculating correlations with significance testing...")
    
    # Get comprehensive features (Story_ and Question_ prefixed)
    feature_cols = [col for col in df_clean.columns if col.startswith(('Story_', 'Question_'))]
    numeric_features = [col for col in feature_cols if df_clean[col].dtype in ['int64', 'float64']]
    
    # Filter features with sufficient data
    good_features = []
    for col in numeric_features:
        non_null_count = df_clean[col].notna().sum()
        non_zero_count = (df_clean[col] != 0).sum()
        if non_null_count > 100 and non_zero_count > 50:  # Sufficient data
            good_features.append(col)
    
    print(f"Features with sufficient data: {len(good_features)}")
    
    # Prepare results storage
    correlation_results = []
    
    # For each subject type
    subjects = list(performance_data.keys())
    categories = list(performance_data['Human'].keys())
    
    for subject in subjects:
        print(f"  Processing {subject}...")
        
        # Create performance column for this subject
        df_clean[f'{subject}_Performance'] = df_clean['Main_Category'].map(performance_data[subject])
        
        # Calculate correlations for each feature
        for feature in good_features:
            # Get non-null data
            mask = df_clean[feature].notna() & df_clean[f'{subject}_Performance'].notna()
            if mask.sum() < 50:  # Need at least 50 samples
                continue
                
            x = df_clean.loc[mask, feature]
            y = df_clean.loc[mask, f'{subject}_Performance']
            
            # Calculate Pearson correlation
            try:
                corr, p_value = pearsonr(x, y)
                
                # Store result
                correlation_results.append({
                    'Feature': feature,
                    'Subject': subject,
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Abs_Correlation': abs(corr),
                    'N_Samples': mask.sum()
                })
            except Exception as e:
                print(f"    Error with {feature}: {e}")
                continue
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(correlation_results)
    
    print(f"Total correlations calculated: {len(corr_df)}")
    
    # Apply multiple testing correction within each subject
    print("Applying multiple testing correction...")
    
    corrected_results = []
    for subject in subjects:
        subject_data = corr_df[corr_df['Subject'] == subject].copy()
        if len(subject_data) > 0:
            # Apply FDR correction
            rejected, p_corrected, _, _ = multipletests(
                subject_data['P_Value'], 
                alpha=0.05, 
                method='fdr_bh'
            )
            
            subject_data['P_Corrected'] = p_corrected
            subject_data['Significant'] = rejected
            corrected_results.append(subject_data)
    
    final_corr_df = pd.concat(corrected_results, ignore_index=True)
    
    # Filter for significant correlations
    significant_df = final_corr_df[final_corr_df['Significant']].copy()
    
    print(f"Significant correlations (p < 0.05 after correction): {len(significant_df)}")
    
    return final_corr_df, significant_df

def create_correlation_matrix(significant_df):
    """Create correlation matrix for significant features"""
    
    print("\nCreating correlation matrix...")
    
    if len(significant_df) == 0:
        print("No significant correlations found!")
        return None, None
    
    # Get features that are significant for at least one subject
    significant_features = significant_df['Feature'].unique()
    subjects = significant_df['Subject'].unique()
    
    print(f"Significant features: {len(significant_features)}")
    print(f"Subjects: {list(subjects)}")
    
    # Create matrix
    matrix_data = []
    for feature in significant_features:
        row_data = {'Feature': feature}
        for subject in subjects:
            # Get correlation for this feature-subject combination
            subset = significant_df[
                (significant_df['Feature'] == feature) & 
                (significant_df['Subject'] == subject)
            ]
            if len(subset) > 0:
                corr = subset.iloc[0]['Correlation']
                p_val = subset.iloc[0]['P_Corrected']
                
                # Add significance markers
                if p_val < 0.001:
                    marker = '***'
                elif p_val < 0.01:
                    marker = '**'
                elif p_val < 0.05:
                    marker = '*'
                else:
                    marker = ''
                
                row_data[subject] = corr
                row_data[f'{subject}_Sig'] = marker
            else:
                row_data[subject] = np.nan
                row_data[f'{subject}_Sig'] = ''
        
        matrix_data.append(row_data)
    
    matrix_df = pd.DataFrame(matrix_data)
    
    # Create correlation values matrix for heatmap
    corr_matrix = matrix_df.set_index('Feature')[subjects].astype(float)
    
    # Create significance matrix
    sig_cols = [f'{subject}_Sig' for subject in subjects]
    sig_matrix = matrix_df.set_index('Feature')[sig_cols]
    sig_matrix.columns = subjects  # Remove _Sig suffix
    
    return corr_matrix, sig_matrix

def find_universal_features(significant_df, min_subjects=3):
    """Find features that are significant across multiple subjects"""
    
    print(f"\nFinding universal features (significant for ≥{min_subjects} subjects)...")
    
    # Count how many subjects each feature is significant for
    feature_counts = significant_df.groupby('Feature')['Subject'].count().sort_values(ascending=False)
    
    universal_features = feature_counts[feature_counts >= min_subjects]
    
    print(f"Universal features (≥{min_subjects} subjects): {len(universal_features)}")
    
    if len(universal_features) > 0:
        print("\nTop universal features:")
        for feature, count in universal_features.head(10).items():
            print(f"  {feature}: {count} subjects")
            
            # Show correlations for this feature
            feature_data = significant_df[significant_df['Feature'] == feature]
            for _, row in feature_data.iterrows():
                print(f"    {row['Subject']}: r={row['Correlation']:.3f} (p={row['P_Corrected']:.3e})")
    
    return universal_features

def create_visualizations(corr_matrix, sig_matrix, output_dir='../results/visualizations/'):
    """Create correlation matrix visualizations"""
    
    print("\nCreating visualizations...")
    
    if corr_matrix is None or len(corr_matrix) == 0:
        print("No data to visualize!")
        return
    
    # Set up the plot
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(corr_matrix) * 0.3)))
    
    # Create annotations with correlation values and significance
    annotations = []
    for i, feature in enumerate(corr_matrix.index):
        row_annotations = []
        for j, subject in enumerate(corr_matrix.columns):
            corr_val = corr_matrix.iloc[i, j]
            sig_marker = sig_matrix.iloc[i, j]
            
            if pd.isna(corr_val):
                annotation = ''
            else:
                annotation = f'{corr_val:.2f}{sig_marker}'
            
            row_annotations.append(annotation)
        annotations.append(row_annotations)
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                cmap='RdBu_r',  # Red-Blue colormap (reversed)
                center=0,
                cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                square=False,
                linewidths=0.5,
                linecolor='white',
                ax=ax,
                annot=annotations,
                fmt='',
                annot_kws={'fontsize': 8, 'fontweight': 'bold'})
    
    # Customize the plot
    ax.set_title('Significant Feature Correlations with ToM Performance\n(* p<0.05, ** p<0.01, *** p<0.001)', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xlabel('Subjects (Human + LLMs)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Significant Features', fontweight='bold', fontsize=12)
    
    # Rotate labels for better readability
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save as PDF and PNG
    plt.savefig(f'{output_dir}correlation_matrix.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}correlation_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Correlation matrix saved to: {output_dir}correlation_matrix.pdf")

def save_results(final_corr_df, significant_df, universal_features, output_dir='../results/correlations/'):
    """Save all results to CSV files"""
    
    print("\nSaving results...")
    
    # Save all correlations
    final_corr_df.to_csv(f'{output_dir}all_correlations.csv', index=False)
    print(f"All correlations saved to: {output_dir}all_correlations.csv")
    
    # Save significant correlations
    significant_df.to_csv(f'{output_dir}significant_correlations.csv', index=False)
    print(f"Significant correlations saved to: {output_dir}significant_correlations.csv")
    
    # Save universal features
    if len(universal_features) > 0:
        universal_df = pd.DataFrame({
            'Feature': universal_features.index,
            'Num_Significant_Subjects': universal_features.values
        })
        universal_df.to_csv(f'{output_dir}universal_features.csv', index=False)
        print(f"Universal features saved to: {output_dir}universal_features.csv")

def main():
    """Main analysis pipeline"""
    
    print("="*80)
    print("FINAL CORRELATION ANALYSIS: UNIVERSAL ToM COMPLEXITY FEATURES")
    print("="*80)
    
    # Load and merge data
    df_clean, df_original = load_and_merge_data()
    
    # Extract performance data
    performance_data = extract_performance_data(df_original)
    
    # Calculate correlations with significance
    final_corr_df, significant_df = calculate_correlations_with_significance(df_clean, performance_data)
    
    # Create correlation matrix
    corr_matrix, sig_matrix = create_correlation_matrix(significant_df)
    
    # Find universal features
    universal_features = find_universal_features(significant_df, min_subjects=3)
    
    # Create visualizations
    if corr_matrix is not None:
        create_visualizations(corr_matrix, sig_matrix)
    
    # Save results
    save_results(final_corr_df, significant_df, universal_features)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"✓ Total correlations calculated: {len(final_corr_df)}")
    print(f"✓ Significant correlations (p<0.05): {len(significant_df)}")
    print(f"✓ Universal features (≥3 subjects): {len(universal_features)}")
    print(f"✓ Results saved to: ../results/")
    
    if len(significant_df) > 0:
        print(f"\nTop 5 strongest correlations:")
        top_corr = significant_df.nlargest(5, 'Abs_Correlation')
        for _, row in top_corr.iterrows():
            print(f"  {row['Feature'][:50]:50s} | {row['Subject']:12s} | r={row['Correlation']:6.3f} | p={row['P_Corrected']:.2e}")

if __name__ == "__main__":
    main()
