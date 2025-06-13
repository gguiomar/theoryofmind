#!/usr/bin/env python3
"""
Mixed Metric Analysis: Combining significant metrics from both comprehensive and advanced ToM analyses
to create a composite metric that shows significance across all models
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def load_correlation_results():
    """Load both correlation result files"""
    print("Loading correlation results...")
    
    # Load comprehensive results
    comprehensive_results = pd.read_csv('comprehensive_correlation_results.csv')
    print(f"Comprehensive results: {len(comprehensive_results)} correlations")
    
    # Load advanced ToM results
    advanced_results = pd.read_csv('advanced_tom_correlation_results.csv')
    print(f"Advanced ToM results: {len(advanced_results)} correlations")
    
    return comprehensive_results, advanced_results

def identify_significant_metrics(comprehensive_results, advanced_results):
    """Identify all significant metrics from both analyses"""
    
    # Get significant metrics from comprehensive analysis
    comp_significant = comprehensive_results[comprehensive_results['Significant'] == True]
    comp_metrics = comp_significant['Metric'].unique()
    
    # Get significant metrics from advanced analysis
    adv_significant = advanced_results[advanced_results['Significant'] == True]
    adv_metrics = adv_significant['Metric'].unique()
    
    # Combine all significant metrics
    all_significant_metrics = list(set(list(comp_metrics) + list(adv_metrics)))
    
    print(f"\nSignificant metrics identified:")
    print(f"From comprehensive analysis: {len(comp_metrics)} metrics")
    for metric in comp_metrics:
        correlations = comp_significant[comp_significant['Metric'] == metric]
        subjects = correlations['Subject'].tolist()
        print(f"  {metric}: {subjects}")
    
    print(f"\nFrom advanced ToM analysis: {len(adv_metrics)} metrics")
    for metric in adv_metrics:
        correlations = adv_significant[adv_significant['Metric'] == metric]
        subjects = correlations['Subject'].tolist()
        print(f"  {metric}: {subjects}")
    
    print(f"\nTotal unique significant metrics: {len(all_significant_metrics)}")
    
    return all_significant_metrics, comp_significant, adv_significant

def load_and_prepare_data():
    """Load dataset_v4 and prepare for mixed metric analysis"""
    print("\nLoading dataset_v4...")
    df = pd.read_csv('./dataset_v4.csv')
    df.columns = df.columns.str.strip()
    
    # Clean data
    df_clean = df[df['ABILITY'].notna()].copy()
    df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
    
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Categories: {df_clean['Main_Category'].unique()}")
    
    return df_clean

def calculate_model_performance(df):
    """Calculate model performance by category"""
    categories = df['Main_Category'].unique()
    
    # Human performance data
    human_performance = {
        'Emotion': 86.4,
        'Desire': 90.4,
        'Intention': 82.2,
        'Knowledge': 89.3,
        'Belief': 89.0,
        'NLC': 86.1
    }
    
    # Model columns
    model_cols = [col for col in df.columns if any(model in col for model in 
                  ['meta_llama', 'Qwen', 'allenai', 'mistralai', 'microsoft', 'internlm'])]
    
    # Model name mapping
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
    
    return performance_matrix

def create_mixed_metrics(df, significant_metrics):
    """Create various mixed metrics from significant individual metrics"""
    
    # Filter to only include metrics that exist in the dataset
    available_metrics = [metric for metric in significant_metrics if metric in df.columns]
    print(f"\nAvailable significant metrics in dataset: {len(available_metrics)}")
    
    if len(available_metrics) == 0:
        print("No significant metrics found in dataset!")
        return df
    
    # Calculate mixed metrics by category
    mixed_metrics = {}
    
    for category in df['Main_Category'].unique():
        category_data = df[df['Main_Category'] == category]
        
        # Method 1: Simple average of standardized metrics
        scaler = StandardScaler()
        if len(available_metrics) > 0:
            try:
                standardized_data = scaler.fit_transform(category_data[available_metrics].fillna(0))
                mixed_metrics[f'{category}_Mixed_Average'] = np.mean(standardized_data, axis=1)
            except:
                mixed_metrics[f'{category}_Mixed_Average'] = np.zeros(len(category_data))
    
    # Method 2: Weighted average based on correlation strength
    # Load correlation results to get weights
    comprehensive_results, advanced_results = load_correlation_results()
    all_results = pd.concat([comprehensive_results, advanced_results])
    
    # Calculate weights for each metric based on average absolute correlation
    metric_weights = {}
    for metric in available_metrics:
        metric_correlations = all_results[all_results['Metric'] == metric]['Correlation'].abs()
        if len(metric_correlations) > 0:
            metric_weights[metric] = metric_correlations.mean()
        else:
            metric_weights[metric] = 0.1  # Default small weight
    
    # Normalize weights
    total_weight = sum(metric_weights.values())
    if total_weight > 0:
        metric_weights = {k: v/total_weight for k, v in metric_weights.items()}
    
    print(f"\nMetric weights for mixed metric:")
    for metric, weight in sorted(metric_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {metric}: {weight:.3f}")
    
    # Create weighted mixed metric
    df_copy = df.copy()
    
    # Standardize all metrics first
    scaler = StandardScaler()
    if len(available_metrics) > 0:
        standardized_metrics = pd.DataFrame(
            scaler.fit_transform(df[available_metrics].fillna(0)),
            columns=available_metrics,
            index=df.index
        )
        
        # Calculate weighted sum
        weighted_sum = np.zeros(len(df))
        for metric in available_metrics:
            weighted_sum += standardized_metrics[metric] * metric_weights[metric]
        
        df_copy['Mixed_Metric_Weighted'] = weighted_sum
    
    # Method 3: PCA-based mixed metric
    if len(available_metrics) > 1:
        try:
            pca = PCA(n_components=1)
            pca_result = pca.fit_transform(standardized_metrics)
            df_copy['Mixed_Metric_PCA'] = pca_result.flatten()
            
            print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}")
            
            # Show PCA component weights
            print("PCA component weights:")
            for i, metric in enumerate(available_metrics):
                weight = pca.components_[0][i]
                print(f"  {metric}: {weight:.3f}")
                
        except:
            df_copy['Mixed_Metric_PCA'] = df_copy['Mixed_Metric_Weighted']
    else:
        df_copy['Mixed_Metric_PCA'] = df_copy['Mixed_Metric_Weighted']
    
    return df_copy, available_metrics, metric_weights

def analyze_mixed_metric_correlations(df, performance_matrix):
    """Analyze correlations for the mixed metrics"""
    
    mixed_metric_cols = [col for col in df.columns if col.startswith('Mixed_Metric')]
    categories = performance_matrix.index.tolist()
    subjects = performance_matrix.columns.tolist()
    
    results = []
    
    print(f"\nAnalyzing correlations for {len(mixed_metric_cols)} mixed metrics...")
    
    for metric in mixed_metric_cols:
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
        
        # Calculate correlations with each subject
        for subject in subjects:
            performance_values = performance_matrix[subject].values
            
            # Skip if too many missing values
            valid_mask = ~(pd.isna(metric_values) | pd.isna(performance_values))
            if sum(valid_mask) < 3:
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

def create_mixed_metric_visualization(correlation_results):
    """Create visualization for mixed metric correlations"""
    
    if len(correlation_results) == 0:
        print("No correlation results to visualize")
        return None
    
    # Prepare data for heatmap
    pivot_data = correlation_results.pivot(index='Metric', columns='Subject', values='Correlation')
    pivot_pvals = correlation_results.pivot(index='Metric', columns='Subject', values='P_Value')
    
    # Create significance mask
    significance_mask = pivot_pvals < 0.05
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Mixed metric correlation heatmap
    sns.heatmap(pivot_data, 
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8},
                linewidths=0.5,
                linecolor='white',
                ax=ax,
                annot=True,
                fmt='.3f',
                square=True)
    
    # Add significance markers
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            if significance_mask.iloc[i, j]:
                ax.text(j + 0.5, i + 0.5, '*', 
                        ha='center', va='center', 
                        color='yellow', fontsize=16, fontweight='bold')
    
    ax.set_title('Mixed Metric Correlations with Model Performance', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xlabel('Subject', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mixed Metrics', fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_mixed_metric_summary(correlation_results, available_metrics):
    """Print summary of mixed metric analysis"""
    print("\n" + "="*70)
    print("MIXED METRIC ANALYSIS SUMMARY")
    print("="*70)
    
    if len(correlation_results) == 0:
        print("No correlation results to summarize")
        return
    
    total_correlations = len(correlation_results)
    significant_correlations = correlation_results['Significant'].sum()
    
    print(f"Total correlations analyzed: {total_correlations}")
    print(f"Significant correlations (p < 0.05): {significant_correlations} ({significant_correlations/total_correlations*100:.1f}%)")
    
    print(f"\nCorrelation strength distribution:")
    print(f"Strong (|r| > 0.7): {(correlation_results['Correlation'].abs() > 0.7).sum()}")
    print(f"Moderate (0.3 < |r| ≤ 0.7): {((correlation_results['Correlation'].abs() > 0.3) & (correlation_results['Correlation'].abs() <= 0.7)).sum()}")
    print(f"Weak (|r| ≤ 0.3): {(correlation_results['Correlation'].abs() <= 0.3).sum()}")
    
    print(f"\nAll mixed metric correlations:")
    for _, row in correlation_results.iterrows():
        sig_marker = "*" if row['Significant'] else ""
        print(f"  {row['Metric']} vs {row['Subject']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.3f}{sig_marker}")
    
    print(f"\nSubjects with significant correlations:")
    sig_by_subject = correlation_results[correlation_results['Significant']].groupby('Subject').size().sort_values(ascending=False)
    for subject, count in sig_by_subject.items():
        print(f"  {subject}: {count} significant correlations")
    
    print(f"\nComponent metrics used ({len(available_metrics)}):")
    for metric in available_metrics:
        print(f"  {metric}")

def main():
    """Main mixed metric analysis function"""
    print("Starting mixed metric analysis...")
    
    # Load correlation results and identify significant metrics
    comprehensive_results, advanced_results = load_correlation_results()
    significant_metrics, comp_sig, adv_sig = identify_significant_metrics(comprehensive_results, advanced_results)
    
    # Load dataset
    df = load_and_prepare_data()
    
    # Calculate model performance
    performance_matrix = calculate_model_performance(df)
    
    print(f"\nPerformance matrix:")
    print(performance_matrix.round(2))
    
    # Create mixed metrics
    df_with_mixed, available_metrics, metric_weights = create_mixed_metrics(df, significant_metrics)
    
    # Analyze mixed metric correlations
    mixed_correlation_results = analyze_mixed_metric_correlations(df_with_mixed, performance_matrix)
    
    # Create visualization
    fig = create_mixed_metric_visualization(mixed_correlation_results)
    
    # Print summary
    print_mixed_metric_summary(mixed_correlation_results, available_metrics)
    
    # Save results
    mixed_correlation_results.to_csv('mixed_metric_correlation_results.csv', index=False)
    df_with_mixed.to_csv('dataset_v5_with_mixed_metrics.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"  mixed_metric_correlation_results.csv")
    print(f"  dataset_v5_with_mixed_metrics.csv")
    
    return mixed_correlation_results, df_with_mixed, available_metrics, metric_weights, fig

if __name__ == "__main__":
    results = main()

# %%
