#!/usr/bin/env python3
"""
Comprehensive analysis of all metrics in dataset_v3 vs model performance
Based on the approach from idea_density_3.py but extended to all available metrics
"""

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load dataset_v3 and prepare for analysis"""
    print("Loading dataset_v3...")
    df = pd.read_csv('./dataset_v3.csv')
    df.columns = df.columns.str.strip()
    
    # Clean data - remove rows with missing ABILITY or key metrics
    df_clean = df[df['ABILITY'].notna()].copy()
    df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
    
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Categories: {df_clean['Main_Category'].unique()}")
    
    return df_clean

def get_analysis_metrics(df):
    """Identify all numeric metrics suitable for correlation analysis"""
    # Exclude non-metric columns
    exclude_cols = [
        'Unnamed: 0', 'ABILITY', 'TASK', 'INDEX', 'STORY', 'QUESTION', 
        'OPTION-A', 'OPTION-B', 'OPTION-C', 'OPTION-D', 'ANSWER',
        'Main_Category', 'Volition', 'Cognition', 'Emotion'  # Text columns, not numeric
    ]
    
    # Get model columns
    model_cols = [col for col in df.columns if any(model in col for model in 
                  ['meta_llama', 'Qwen', 'allenai', 'mistralai', 'microsoft', 'internlm'])]
    
    # Get all numeric columns that aren't models or excluded
    metric_cols = []
    for col in df.columns:
        if col not in exclude_cols and col not in model_cols:
            if df[col].dtype in ['int64', 'float64']:
                # Check if column has sufficient non-null values
                if df[col].notna().sum() > len(df) * 0.5:  # At least 50% non-null
                    metric_cols.append(col)
    
    print(f"Found {len(metric_cols)} analysis metrics:")
    for col in metric_cols:
        non_null_pct = df[col].notna().sum() / len(df) * 100
        print(f"  {col}: {non_null_pct:.1f}% non-null")
    
    return metric_cols, model_cols

def calculate_model_performance(df, model_cols):
    """Calculate model performance by category"""
    categories = df['Main_Category'].unique()
    
    # Human performance data (from original paper)
    human_performance = {
        'Emotion': 86.4,
        'Desire': 90.4,
        'Intention': 82.2,
        'Knowledge': 89.3,
        'Belief': 89.0,
        'NLC': 86.1
    }
    
    # Model name mapping for display
    model_display_names = {
        'meta_llama_Llama_3.1_70B_Instruct': 'Llama 3.1 70B',
        'Qwen_Qwen2.5_32B_Instruct': 'Qwen 2.5 32B',
        'allenai_OLMo_2_1124_13B_Instruct': 'OLMo 13B',
        'mistralai_Mistral_7B_Instruct_v0.3': 'Mistral 7B',
        'microsoft_Phi_3_mini_4k_instruct': 'Phi-3 Mini',
        'internlm_internlm2_5_1_8b_chat': 'InternLM 1.8B'
    }
    
    def calculate_accuracy(df, model_col, category):
        subset = df[df['Main_Category'] == category]
        if len(subset) == 0:
            return 0
        correct = (subset[model_col] == subset['ANSWER']).sum()
        return (correct / len(subset) * 100)
    
    # Calculate performance matrix
    performance_matrix = pd.DataFrame(index=categories)
    performance_matrix['Human'] = [human_performance.get(cat, np.nan) for cat in categories]
    
    for model in model_cols:
        if model in model_display_names:
            display_name = model_display_names[model]
            performance_matrix[display_name] = [
                calculate_accuracy(df, model, cat) for cat in categories
            ]
    
    return performance_matrix, model_display_names

def analyze_metric_correlations(df, metric_cols, performance_matrix):
    """Analyze correlations between each metric and model performance"""
    categories = performance_matrix.index.tolist()
    subjects = performance_matrix.columns.tolist()
    
    results = []
    
    print(f"\nAnalyzing correlations for {len(metric_cols)} metrics...")
    
    for metric in metric_cols:
        print(f"Processing {metric}...")
        
        # Calculate metric averages by category
        metric_by_category = df.groupby('Main_Category')[metric].mean()
        
        # Ensure we have data for all categories
        metric_values = []
        for cat in categories:
            if cat in metric_by_category.index:
                metric_values.append(metric_by_category[cat])
            else:
                metric_values.append(np.nan)
        
        # Skip if too many missing values
        if sum(pd.isna(metric_values)) > len(metric_values) * 0.5:
            continue
        
        # Calculate correlations with each subject
        for subject in subjects:
            performance_values = performance_matrix[subject].values
            
            # Skip if too many missing values
            valid_mask = ~(pd.isna(metric_values) | pd.isna(performance_values))
            if sum(valid_mask) < 3:  # Need at least 3 points for correlation
                continue
            
            x = np.array(metric_values)[valid_mask]
            y = np.array(performance_values)[valid_mask]
            
            if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0:
                corr, p_val = stats.pearsonr(x, y)
                
                results.append({
                    'Metric': metric,
                    'Subject': subject,
                    'Correlation': corr,
                    'P_Value': p_val,
                    'Significant': p_val < 0.05,
                    'N_Points': len(x)
                })
    
    return pd.DataFrame(results)

def create_comprehensive_visualization(correlation_results):
    """Create comprehensive visualization of all metric correlations"""
    
    # Prepare data for heatmap
    pivot_data = correlation_results.pivot(index='Metric', columns='Subject', values='Correlation')
    pivot_pvals = correlation_results.pivot(index='Metric', columns='Subject', values='P_Value')
    
    # Create significance mask
    significance_mask = pivot_pvals < 0.05
    
    # Sort metrics by average absolute correlation
    avg_abs_corr = pivot_data.abs().mean(axis=1).sort_values(ascending=False)
    pivot_data = pivot_data.loc[avg_abs_corr.index]
    significance_mask = significance_mask.loc[avg_abs_corr.index]
    
    # Create single plot - just the correlation matrix with square cells
    n_metrics = len(pivot_data)
    n_subjects = len(pivot_data.columns)
    
    # Calculate figure size to make cells square
    cell_size = 0.6  # Size of each cell in inches
    fig_width = n_subjects * cell_size + 3  # Extra space for labels and colorbar
    fig_height = n_metrics * cell_size + 2   # Extra space for title and labels
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # Comprehensive correlation heatmap with square cells
    sns.heatmap(pivot_data, 
                cmap='RdBu_r',  # Red-Blue colormap centered at 0
                center=0,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8},
                linewidths=0.5,
                linecolor='white',
                ax=ax,
                annot=False,
                square=True)  # Make cells square
    
    # Add significance markers
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            if significance_mask.iloc[i, j]:
                ax.text(j + 0.5, i + 0.5, '*', 
                        ha='center', va='center', 
                        color='black', fontsize=12, fontweight='bold')
    ax.set_xlabel('Subject', fontweight='bold', fontsize=12)
    ax.set_ylabel('Metrics', fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_summary_statistics(correlation_results):
    """Print comprehensive summary statistics"""
    print("\n" + "="*60)
    print("COMPREHENSIVE METRICS ANALYSIS SUMMARY")
    print("="*60)
    
    total_correlations = len(correlation_results)
    significant_correlations = correlation_results['Significant'].sum()
    
    print(f"Total correlations analyzed: {total_correlations}")
    print(f"Significant correlations (p < 0.05): {significant_correlations} ({significant_correlations/total_correlations*100:.1f}%)")
    
    print(f"\nCorrelation strength distribution:")
    print(f"Strong (|r| > 0.7): {(correlation_results['Correlation'].abs() > 0.7).sum()}")
    print(f"Moderate (0.3 < |r| ≤ 0.7): {((correlation_results['Correlation'].abs() > 0.3) & (correlation_results['Correlation'].abs() <= 0.7)).sum()}")
    print(f"Weak (|r| ≤ 0.3): {(correlation_results['Correlation'].abs() <= 0.3).sum()}")
    
    print(f"\nTop 10 strongest correlations:")
    top_correlations = correlation_results.loc[correlation_results['Correlation'].abs().nlargest(10).index]
    for _, row in top_correlations.iterrows():
        sig_marker = "*" if row['Significant'] else ""
        print(f"  {row['Metric']} vs {row['Subject']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.3f}{sig_marker}")
    
    print(f"\nMetrics with most significant correlations:")
    sig_by_metric = correlation_results[correlation_results['Significant']].groupby('Metric').size().sort_values(ascending=False)
    for metric, count in sig_by_metric.head(10).items():
        print(f"  {metric}: {count} significant correlations")
    
    print(f"\nSubjects with most significant correlations:")
    sig_by_subject = correlation_results[correlation_results['Significant']].groupby('Subject').size().sort_values(ascending=False)
    for subject, count in sig_by_subject.items():
        print(f"  {subject}: {count} significant correlations")

#%%
def main():
    """Main analysis function"""
    print("Starting comprehensive metrics analysis...")
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Get metrics and models
    metric_cols, model_cols = get_analysis_metrics(df)
    
    # Calculate model performance
    performance_matrix, model_display_names = calculate_model_performance(df, model_cols)
    
    print(f"\nPerformance matrix shape: {performance_matrix.shape}")
    print("Performance matrix:")
    print(performance_matrix.round(2))
    
    # Analyze correlations
    correlation_results = analyze_metric_correlations(df, metric_cols, performance_matrix)
    
    print(f"\nGenerated {len(correlation_results)} correlation analyses")
    
    # Create visualization
    fig = create_comprehensive_visualization(correlation_results)
    
    # Print summary
    print_summary_statistics(correlation_results)
    
    # Save results
    correlation_results.to_csv('comprehensive_correlation_results.csv', index=False)
    print(f"\nResults saved to: comprehensive_correlation_results.csv")
    
    return correlation_results, performance_matrix, fig

if __name__ == "__main__":
    correlation_results, performance_matrix, fig = main()

# %%
