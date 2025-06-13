#!/usr/bin/env python3
"""
Correlation analysis specifically for advanced ToM metrics
Based on comprehensive_metrics_analysis.py but focused on new advanced metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load dataset_v4 and prepare for analysis"""
    print("Loading dataset_v4...")
    df = pd.read_csv('./dataset_v4.csv')
    df.columns = df.columns.str.strip()
    
    # Clean data - remove rows with missing ABILITY or key metrics
    df_clean = df[df['ABILITY'].notna()].copy()
    df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
    
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Categories: {df_clean['Main_Category'].unique()}")
    
    return df_clean

def get_advanced_tom_metrics(df):
    """Identify advanced ToM metrics for correlation analysis"""
    # Get all advanced ToM metrics (those starting with Story_ or Q_)
    advanced_metrics = []
    
    for col in df.columns:
        if col.startswith(('Story_', 'Q_')) and df[col].dtype in ['int64', 'float64']:
            # Check if column has sufficient non-null values
            if df[col].notna().sum() > len(df) * 0.5:  # At least 50% non-null
                advanced_metrics.append(col)
    
    # Get model columns
    model_cols = [col for col in df.columns if any(model in col for model in 
                  ['meta_llama', 'Qwen', 'allenai', 'mistralai', 'microsoft', 'internlm'])]
    
    print(f"Found {len(advanced_metrics)} advanced ToM metrics:")
    for col in advanced_metrics:
        non_null_pct = df[col].notna().sum() / len(df) * 100
        non_zero_pct = (df[col] > 0).sum() / len(df) * 100
        print(f"  {col}: {non_null_pct:.1f}% non-null, {non_zero_pct:.1f}% non-zero")
    
    return advanced_metrics, model_cols

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

def analyze_advanced_tom_correlations(df, metric_cols, performance_matrix):
    """Analyze correlations between advanced ToM metrics and model performance"""
    categories = performance_matrix.index.tolist()
    subjects = performance_matrix.columns.tolist()
    
    results = []
    
    print(f"\nAnalyzing correlations for {len(metric_cols)} advanced ToM metrics...")
    
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

def create_advanced_tom_visualization(correlation_results):
    """Create visualization focused on advanced ToM metrics"""
    
    # Prepare data for heatmap
    pivot_data = correlation_results.pivot(index='Metric', columns='Subject', values='Correlation')
    pivot_pvals = correlation_results.pivot(index='Metric', columns='Subject', values='P_Value')
    
    # Create significance mask
    significance_mask = pivot_pvals < 0.05
    
    # Sort metrics by average absolute correlation
    avg_abs_corr = pivot_data.abs().mean(axis=1).sort_values(ascending=False)
    pivot_data = pivot_data.loc[avg_abs_corr.index]
    significance_mask = significance_mask.loc[avg_abs_corr.index]
    
    # Create visualization
    n_metrics = len(pivot_data)
    n_subjects = len(pivot_data.columns)
    
    # Calculate figure size
    cell_size = 0.6
    fig_width = n_subjects * cell_size + 4
    fig_height = n_metrics * cell_size + 3
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # Advanced ToM correlation heatmap
    sns.heatmap(pivot_data, 
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8},
                linewidths=0.5,
                linecolor='white',
                ax=ax,
                annot=False,
                square=True)
    
    # Add significance markers
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            if significance_mask.iloc[i, j]:
                ax.text(j + 0.5, i + 0.5, '*', 
                        ha='center', va='center', 
                        color='black', fontsize=12, fontweight='bold')
    
    ax.set_title('Advanced Theory of Mind Metrics - Correlation Analysis', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xlabel('Subject', fontweight='bold', fontsize=12)
    ax.set_ylabel('Advanced ToM Metrics', fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_advanced_tom_summary(correlation_results):
    """Print summary statistics for advanced ToM metrics"""
    print("\n" + "="*70)
    print("ADVANCED THEORY OF MIND METRICS CORRELATION ANALYSIS")
    print("="*70)
    
    total_correlations = len(correlation_results)
    significant_correlations = correlation_results['Significant'].sum()
    
    print(f"Total correlations analyzed: {total_correlations}")
    print(f"Significant correlations (p < 0.05): {significant_correlations} ({significant_correlations/total_correlations*100:.1f}%)")
    
    print(f"\nCorrelation strength distribution:")
    print(f"Strong (|r| > 0.7): {(correlation_results['Correlation'].abs() > 0.7).sum()}")
    print(f"Moderate (0.3 < |r| ≤ 0.7): {((correlation_results['Correlation'].abs() > 0.3) & (correlation_results['Correlation'].abs() <= 0.7)).sum()}")
    print(f"Weak (|r| ≤ 0.3): {(correlation_results['Correlation'].abs() <= 0.3).sum()}")
    
    print(f"\nTop 15 strongest correlations:")
    top_correlations = correlation_results.loc[correlation_results['Correlation'].abs().nlargest(15).index]
    for _, row in top_correlations.iterrows():
        sig_marker = "*" if row['Significant'] else ""
        print(f"  {row['Metric']} vs {row['Subject']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.3f}{sig_marker}")
    
    print(f"\nAdvanced ToM metrics with most significant correlations:")
    sig_by_metric = correlation_results[correlation_results['Significant']].groupby('Metric').size().sort_values(ascending=False)
    for metric, count in sig_by_metric.head(15).items():
        print(f"  {metric}: {count} significant correlations")
    
    print(f"\nSubjects with most significant correlations:")
    sig_by_subject = correlation_results[correlation_results['Significant']].groupby('Subject').size().sort_values(ascending=False)
    for subject, count in sig_by_subject.items():
        print(f"  {subject}: {count} significant correlations")
    
    # Category analysis
    print(f"\nMetric category analysis:")
    categories = ['MS_Embedding', 'Perspective', 'Recursive', 'Coref', 'False_Belief', 'MS_Arg', 'Temporal']
    
    for category in categories:
        category_metrics = correlation_results[correlation_results['Metric'].str.contains(category)]
        if len(category_metrics) > 0:
            sig_count = category_metrics['Significant'].sum()
            total_count = len(category_metrics)
            avg_abs_corr = category_metrics['Correlation'].abs().mean()
            print(f"  {category}: {sig_count}/{total_count} significant ({sig_count/total_count*100:.1f}%), avg |r| = {avg_abs_corr:.3f}")

def compare_with_original_metrics(advanced_results, original_results_path='comprehensive_correlation_results.csv'):
    """Compare advanced ToM metrics with original metrics"""
    try:
        original_results = pd.read_csv(original_results_path)
        
        print(f"\n" + "="*50)
        print("COMPARISON WITH ORIGINAL METRICS")
        print("="*50)
        
        # Calculate improvement metrics
        original_sig_rate = original_results['Significant'].sum() / len(original_results) * 100
        advanced_sig_rate = advanced_results['Significant'].sum() / len(advanced_results) * 100
        
        print(f"Original metrics significance rate: {original_sig_rate:.1f}%")
        print(f"Advanced ToM metrics significance rate: {advanced_sig_rate:.1f}%")
        print(f"Improvement: {advanced_sig_rate - original_sig_rate:+.1f} percentage points")
        
        # Compare average correlation strengths
        original_avg_abs_corr = original_results['Correlation'].abs().mean()
        advanced_avg_abs_corr = advanced_results['Correlation'].abs().mean()
        
        print(f"\nOriginal metrics avg |correlation|: {original_avg_abs_corr:.3f}")
        print(f"Advanced ToM metrics avg |correlation|: {advanced_avg_abs_corr:.3f}")
        print(f"Improvement: {advanced_avg_abs_corr - original_avg_abs_corr:+.3f}")
        
    except FileNotFoundError:
        print(f"\nWarning: Could not find {original_results_path} for comparison")

def main():
    """Main analysis function for advanced ToM metrics"""
    print("Starting advanced ToM metrics correlation analysis...")
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Get advanced ToM metrics and models
    metric_cols, model_cols = get_advanced_tom_metrics(df)
    
    # Calculate model performance
    performance_matrix, model_display_names = calculate_model_performance(df, model_cols)
    
    print(f"\nPerformance matrix shape: {performance_matrix.shape}")
    print("Performance matrix:")
    print(performance_matrix.round(2))
    
    # Analyze correlations
    correlation_results = analyze_advanced_tom_correlations(df, metric_cols, performance_matrix)
    
    print(f"\nGenerated {len(correlation_results)} correlation analyses")
    
    # Create visualization
    fig = create_advanced_tom_visualization(correlation_results)
    
    # Print summary
    print_advanced_tom_summary(correlation_results)
    
    # Compare with original metrics
    compare_with_original_metrics(correlation_results)
    
    # Save results
    correlation_results.to_csv('advanced_tom_correlation_results.csv', index=False)
    print(f"\nResults saved to: advanced_tom_correlation_results.csv")
    
    return correlation_results, performance_matrix, fig

if __name__ == "__main__":
    correlation_results, performance_matrix, fig = main()
