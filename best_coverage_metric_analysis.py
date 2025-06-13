#!/usr/bin/env python3
"""
Best Coverage Metric Analysis: Create metric using strongest correlations for each subject
Since some subjects have no significant correlations, use strongest correlations instead
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_all_correlations():
    """Load and analyze all correlation results"""
    print("Loading all correlation results...")
    
    # Load both correlation result files
    comprehensive_results = pd.read_csv('comprehensive_correlation_results.csv')
    advanced_results = pd.read_csv('advanced_tom_correlation_results.csv')
    
    # Combine all results
    all_results = pd.concat([comprehensive_results, advanced_results], ignore_index=True)
    
    print(f"Total correlations: {len(all_results)}")
    print(f"Significant correlations: {all_results['Significant'].sum()}")
    
    return all_results

def find_strongest_correlations_per_subject(all_results):
    """Find strongest correlation for each subject, regardless of significance"""
    print("\n" + "="*70)
    print("FINDING STRONGEST CORRELATIONS PER SUBJECT")
    print("="*70)
    
    subjects = all_results['Subject'].unique()
    strongest_per_subject = {}
    
    for subject in subjects:
        subject_data = all_results[all_results['Subject'] == subject]
        
        # Find strongest absolute correlation
        strongest_idx = subject_data['Correlation'].abs().idxmax()
        strongest_row = subject_data.loc[strongest_idx]
        
        strongest_per_subject[subject] = {
            'metric': strongest_row['Metric'],
            'correlation': strongest_row['Correlation'],
            'p_value': strongest_row['P_Value'],
            'significant': strongest_row['Significant']
        }
        
        sig_marker = "*" if strongest_row['Significant'] else ""
        print(f"{subject}: {strongest_row['Metric']}")
        print(f"  r = {strongest_row['Correlation']:.3f}, p = {strongest_row['P_Value']:.3f}{sig_marker}")
    
    return strongest_per_subject

def find_best_coverage_metrics(all_results, min_subjects=5):
    """Find metrics that have strong correlations with the most subjects"""
    print(f"\n" + "="*70)
    print(f"FINDING METRICS WITH BEST SUBJECT COVERAGE (min {min_subjects} subjects)")
    print("="*70)
    
    subjects = all_results['Subject'].unique()
    metrics = all_results['Metric'].unique()
    
    metric_coverage = {}
    
    for metric in metrics:
        metric_data = all_results[all_results['Metric'] == metric]
        
        # Count subjects with strong correlations (|r| > 0.5)
        strong_correlations = metric_data[metric_data['Correlation'].abs() > 0.5]
        coverage_count = len(strong_correlations)
        
        if coverage_count >= min_subjects:
            metric_coverage[metric] = {
                'coverage_count': coverage_count,
                'subjects': strong_correlations['Subject'].tolist(),
                'correlations': strong_correlations['Correlation'].tolist(),
                'p_values': strong_correlations['P_Value'].tolist(),
                'avg_abs_correlation': strong_correlations['Correlation'].abs().mean()
            }
    
    # Sort by coverage count and average correlation strength
    sorted_metrics = sorted(metric_coverage.items(), 
                           key=lambda x: (x[1]['coverage_count'], x[1]['avg_abs_correlation']), 
                           reverse=True)
    
    print(f"Metrics with coverage of {min_subjects}+ subjects:")
    for metric, data in sorted_metrics:
        print(f"\n{metric}: covers {data['coverage_count']} subjects")
        print(f"  Average |correlation|: {data['avg_abs_correlation']:.3f}")
        print(f"  Subjects: {data['subjects']}")
        for i, subject in enumerate(data['subjects']):
            sig_marker = "*" if data['p_values'][i] < 0.05 else ""
            print(f"    {subject}: r = {data['correlations'][i]:.3f}, p = {data['p_values'][i]:.3f}{sig_marker}")
    
    return sorted_metrics

def create_best_coverage_mixed_metric(all_results):
    """Create mixed metric using metrics with best subject coverage"""
    print(f"\n" + "="*70)
    print("CREATING BEST COVERAGE MIXED METRIC")
    print("="*70)
    
    # Find metrics with good coverage
    coverage_metrics = find_best_coverage_metrics(all_results, min_subjects=4)
    
    if not coverage_metrics:
        print("No metrics found with sufficient coverage, trying with lower threshold...")
        coverage_metrics = find_best_coverage_metrics(all_results, min_subjects=3)
    
    if not coverage_metrics:
        print("Still no metrics found, using top correlations approach...")
        return create_top_correlations_mixed_metric(all_results)
    
    # Select top coverage metrics
    selected_metrics = [metric for metric, data in coverage_metrics[:5]]  # Top 5
    
    print(f"\nSelected metrics for best coverage mixed metric:")
    for metric in selected_metrics:
        print(f"  {metric}")
    
    # Load dataset
    print("\nLoading dataset_v4...")
    df = pd.read_csv('./dataset_v4.csv')
    df.columns = df.columns.str.strip()
    
    # Clean data
    df_clean = df[df['ABILITY'].notna()].copy()
    df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
    
    # Filter to only include metrics that exist in the dataset
    available_metrics = [metric for metric in selected_metrics if metric in df_clean.columns]
    print(f"Available coverage metrics in dataset: {len(available_metrics)}")
    for metric in available_metrics:
        print(f"  {metric}")
    
    if len(available_metrics) == 0:
        print("No coverage metrics found in dataset!")
        return df_clean, []
    
    # Calculate weights based on coverage and correlation strength
    metric_weights = {}
    for metric in available_metrics:
        # Find this metric in coverage results
        metric_data = None
        for m, data in coverage_metrics:
            if m == metric:
                metric_data = data
                break
        
        if metric_data:
            # Weight by coverage count and average correlation
            weight = metric_data['coverage_count'] * metric_data['avg_abs_correlation']
            metric_weights[metric] = weight
        else:
            metric_weights[metric] = 0.1
    
    # Normalize weights
    total_weight = sum(metric_weights.values())
    if total_weight > 0:
        metric_weights = {k: v/total_weight for k, v in metric_weights.items()}
    
    print(f"\nCoverage metric weights:")
    for metric, weight in sorted(metric_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {metric}: {weight:.3f}")
    
    # Create mixed metric
    df_copy = df_clean.copy()
    
    # Standardize all metrics first
    scaler = StandardScaler()
    standardized_metrics = pd.DataFrame(
        scaler.fit_transform(df_clean[available_metrics].fillna(0)),
        columns=available_metrics,
        index=df_clean.index
    )
    
    # Calculate weighted sum
    weighted_sum = np.zeros(len(df_clean))
    for metric in available_metrics:
        weighted_sum += standardized_metrics[metric] * metric_weights[metric]
    
    df_copy['Best_Coverage_Mixed_Metric'] = weighted_sum
    
    return df_copy, available_metrics

def create_top_correlations_mixed_metric(all_results):
    """Alternative: Create mixed metric using top absolute correlations for each subject"""
    print(f"\n" + "="*70)
    print("CREATING TOP CORRELATIONS MIXED METRIC")
    print("="*70)
    
    subjects = all_results['Subject'].unique()
    selected_metrics = set()
    
    # For each subject, find top 2 strongest correlations
    for subject in subjects:
        subject_data = all_results[all_results['Subject'] == subject]
        top_2 = subject_data.nlargest(2, 'Correlation', keep='all')['Metric'].tolist()
        selected_metrics.update(top_2)
        
        print(f"{subject} top correlations:")
        for _, row in subject_data.nlargest(2, 'Correlation', keep='all').iterrows():
            sig_marker = "*" if row['Significant'] else ""
            print(f"  {row['Metric']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.3f}{sig_marker}")
    
    selected_metrics = list(selected_metrics)
    print(f"\nSelected metrics from top correlations ({len(selected_metrics)} total):")
    for metric in selected_metrics:
        print(f"  {metric}")
    
    # Load dataset and create metric (similar to above)
    print("\nLoading dataset_v4...")
    df = pd.read_csv('./dataset_v4.csv')
    df.columns = df.columns.str.strip()
    
    # Clean data
    df_clean = df[df['ABILITY'].notna()].copy()
    df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
    
    # Filter to only include metrics that exist in the dataset
    available_metrics = [metric for metric in selected_metrics if metric in df_clean.columns]
    print(f"Available top correlation metrics in dataset: {len(available_metrics)}")
    
    if len(available_metrics) == 0:
        print("No top correlation metrics found in dataset!")
        return df_clean, []
    
    # Equal weights for simplicity
    metric_weights = {metric: 1.0/len(available_metrics) for metric in available_metrics}
    
    print(f"\nTop correlation metric weights (equal):")
    for metric, weight in metric_weights.items():
        print(f"  {metric}: {weight:.3f}")
    
    # Create mixed metric
    df_copy = df_clean.copy()
    
    # Standardize all metrics first
    scaler = StandardScaler()
    standardized_metrics = pd.DataFrame(
        scaler.fit_transform(df_clean[available_metrics].fillna(0)),
        columns=available_metrics,
        index=df_clean.index
    )
    
    # Calculate weighted sum
    weighted_sum = np.zeros(len(df_clean))
    for metric in available_metrics:
        weighted_sum += standardized_metrics[metric] * metric_weights[metric]
    
    df_copy['Top_Correlations_Mixed_Metric'] = weighted_sum
    
    return df_copy, available_metrics

def analyze_mixed_metric_correlations(df, metric_name):
    """Analyze correlations for the mixed metric"""
    print(f"\n" + "="*70)
    print(f"ANALYZING {metric_name.upper()} CORRELATIONS")
    print("="*70)
    
    # Calculate model performance
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
    
    print("Performance matrix:")
    print(performance_matrix.round(2))
    
    # Analyze mixed metric correlations
    subjects = performance_matrix.columns.tolist()
    
    # Calculate metric averages by category
    metric_by_category = df.groupby('Main_Category')[metric_name].mean()
    
    # Ensure we have data for all categories
    metric_values = []
    for cat in categories:
        if cat in metric_by_category.index:
            metric_values.append(metric_by_category[cat])
        else:
            metric_values.append(np.nan)
    
    results = []
    
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
                'Metric': metric_name,
                'Subject': subject,
                'Correlation': corr,
                'P_Value': p_val,
                'Significant': p_val < 0.05,
                'N_Points': len(x)
            })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{metric_name} Correlations:")
    for _, row in results_df.iterrows():
        sig_marker = "*" if row['Significant'] else ""
        print(f"  {row['Subject']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.3f}{sig_marker}")
    
    significant_count = results_df['Significant'].sum()
    total_count = len(results_df)
    
    print(f"\nSignificance Summary:")
    print(f"  Significant correlations: {significant_count}/{total_count} ({significant_count/total_count*100:.1f}%)")
    print(f"  Subjects with significant correlations: {list(results_df[results_df['Significant']]['Subject'])}")
    
    return results_df

def main():
    """Main analysis function"""
    print("Starting best coverage metric analysis...")
    
    # Load and analyze all correlations
    all_results = load_and_analyze_all_correlations()
    
    # Find strongest correlations per subject
    strongest_per_subject = find_strongest_correlations_per_subject(all_results)
    
    # Try to create best coverage mixed metric
    df_with_coverage, coverage_metrics = create_best_coverage_mixed_metric(all_results)
    
    # Analyze correlations
    if 'Best_Coverage_Mixed_Metric' in df_with_coverage.columns:
        coverage_results = analyze_mixed_metric_correlations(df_with_coverage, 'Best_Coverage_Mixed_Metric')
        coverage_results.to_csv('best_coverage_correlation_results.csv', index=False)
    
    # Also try top correlations approach
    df_with_top, top_metrics = create_top_correlations_mixed_metric(all_results)
    
    if 'Top_Correlations_Mixed_Metric' in df_with_top.columns:
        top_results = analyze_mixed_metric_correlations(df_with_top, 'Top_Correlations_Mixed_Metric')
        top_results.to_csv('top_correlations_correlation_results.csv', index=False)
    
    # Save datasets
    if 'Best_Coverage_Mixed_Metric' in df_with_coverage.columns:
        df_with_coverage.to_csv('dataset_v7_with_coverage_metric.csv', index=False)
    
    if 'Top_Correlations_Mixed_Metric' in df_with_top.columns:
        df_with_top.to_csv('dataset_v8_with_top_correlations_metric.csv', index=False)
    
    print(f"\nResults saved!")
    
    return strongest_per_subject, coverage_metrics, top_metrics

if __name__ == "__main__":
    results = main()
