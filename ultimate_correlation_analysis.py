#!/usr/bin/env python3
"""
Ultimate Correlation Analysis: Find metrics that achieve significance across ALL subjects
Using the expanded dataset_v9 with comprehensive NLP features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def load_expanded_dataset():
    """Load the expanded dataset with all features"""
    print("Loading dataset_v9_expanded_nlp...")
    df = pd.read_csv('./dataset_v9_expanded_nlp.csv')
    df.columns = df.columns.str.strip()
    
    # Clean data
    df_clean = df[df['ABILITY'].notna()].copy()
    df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
    
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Categories: {df_clean['Main_Category'].unique()}")
    
    return df_clean

def get_all_analysis_metrics(df):
    """Get all numeric metrics suitable for correlation analysis"""
    # Exclude non-metric columns
    exclude_cols = [
        'Unnamed: 0', 'ABILITY', 'TASK', 'INDEX', 'STORY', 'QUESTION', 
        'OPTION-A', 'OPTION-B', 'OPTION-C', 'OPTION-D', 'ANSWER',
        'Main_Category', 'Volition', 'Cognition', 'Emotion'
    ]
    
    # Get model columns
    model_cols = [col for col in df.columns if any(model in col for model in 
                  ['meta_llama', 'Qwen', 'allenai', 'mistralai', 'microsoft', 'internlm'])]
    
    # Get all numeric columns that aren't models or excluded
    metric_cols = []
    for col in df.columns:
        if col not in exclude_cols and col not in model_cols:
            if df[col].dtype in ['int64', 'float64']:
                # Check if column has sufficient non-null values and variance
                if df[col].notna().sum() > len(df) * 0.5 and df[col].std() > 0:
                    metric_cols.append(col)
    
    print(f"Found {len(metric_cols)} analysis metrics")
    
    return metric_cols, model_cols

def calculate_model_performance(df, model_cols):
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

def analyze_comprehensive_correlations(df, metric_cols, performance_matrix):
    """Comprehensive correlation analysis"""
    categories = performance_matrix.index.tolist()
    subjects = performance_matrix.columns.tolist()
    
    results = []
    
    print(f"\nAnalyzing correlations for {len(metric_cols)} metrics...")
    
    # Process in batches for progress tracking
    batch_size = 50
    total_batches = (len(metric_cols) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(metric_cols))
        
        print(f"Processing batch {batch_idx + 1}/{total_batches} (metrics {start_idx}-{end_idx})...")
        
        for metric_idx in range(start_idx, end_idx):
            metric = metric_cols[metric_idx]
            
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
                        'N_Points': len(x),
                        'Abs_Correlation': abs(corr)
                    })
    
    return pd.DataFrame(results)

def find_universal_metrics(correlation_results):
    """Find metrics that have strong correlations with ALL subjects"""
    print("\n" + "="*70)
    print("FINDING UNIVERSAL METRICS (STRONG CORRELATIONS WITH ALL SUBJECTS)")
    print("="*70)
    
    subjects = correlation_results['Subject'].unique()
    print(f"All subjects: {list(subjects)}")
    
    # For each metric, check coverage across subjects
    metric_coverage = {}
    
    for metric in correlation_results['Metric'].unique():
        metric_data = correlation_results[correlation_results['Metric'] == metric]
        
        # Check which subjects have strong correlations (|r| > 0.5)
        strong_correlations = metric_data[metric_data['Abs_Correlation'] > 0.5]
        covered_subjects = set(strong_correlations['Subject'].tolist())
        
        # Check which subjects have significant correlations
        significant_correlations = metric_data[metric_data['Significant'] == True]
        significant_subjects = set(significant_correlations['Subject'].tolist())
        
        metric_coverage[metric] = {
            'strong_coverage': len(covered_subjects),
            'strong_subjects': covered_subjects,
            'significant_coverage': len(significant_subjects),
            'significant_subjects': significant_subjects,
            'total_subjects': len(set(metric_data['Subject'].tolist())),
            'avg_abs_correlation': metric_data['Abs_Correlation'].mean(),
            'max_abs_correlation': metric_data['Abs_Correlation'].max(),
            'correlations': dict(zip(metric_data['Subject'], metric_data['Correlation'])),
            'p_values': dict(zip(metric_data['Subject'], metric_data['P_Value']))
        }
    
    # Sort by coverage and correlation strength
    sorted_metrics = sorted(metric_coverage.items(), 
                           key=lambda x: (x[1]['strong_coverage'], x[1]['avg_abs_correlation']), 
                           reverse=True)
    
    print(f"\nTop metrics by subject coverage:")
    for i, (metric, data) in enumerate(sorted_metrics[:20]):
        print(f"\n{i+1}. {metric}")
        print(f"   Strong correlations (|r| > 0.5): {data['strong_coverage']}/{len(subjects)} subjects")
        print(f"   Significant correlations: {data['significant_coverage']}/{len(subjects)} subjects")
        print(f"   Average |correlation|: {data['avg_abs_correlation']:.3f}")
        print(f"   Max |correlation|: {data['max_abs_correlation']:.3f}")
        
        # Show correlations for each subject
        print(f"   Subject correlations:")
        for subject in subjects:
            if subject in data['correlations']:
                corr = data['correlations'][subject]
                p_val = data['p_values'][subject]
                sig_marker = "*" if p_val < 0.05 else ""
                strong_marker = "●" if abs(corr) > 0.5 else "○"
                print(f"     {strong_marker} {subject}: r = {corr:.3f}, p = {p_val:.3f}{sig_marker}")
    
    return metric_coverage, sorted_metrics

def create_ultimate_mixed_metric(correlation_results, metric_coverage, df):
    """Create the ultimate mixed metric using top universal metrics"""
    print("\n" + "="*70)
    print("CREATING ULTIMATE MIXED METRIC")
    print("="*70)
    
    # Select top metrics with best coverage
    top_metrics = []
    for metric, data in metric_coverage.items():
        # Prioritize metrics with coverage of at least 5 subjects and high correlation
        if data['strong_coverage'] >= 5 and data['avg_abs_correlation'] > 0.4:
            top_metrics.append((metric, data))
    
    # Sort by coverage and correlation strength
    top_metrics.sort(key=lambda x: (x[1]['strong_coverage'], x[1]['avg_abs_correlation']), reverse=True)
    
    # Take top 15 metrics
    selected_metrics = [metric for metric, data in top_metrics[:15]]
    
    print(f"Selected {len(selected_metrics)} metrics for ultimate mixed metric:")
    for i, metric in enumerate(selected_metrics):
        data = metric_coverage[metric]
        print(f"  {i+1}. {metric}")
        print(f"     Coverage: {data['strong_coverage']}/7 subjects, avg |r| = {data['avg_abs_correlation']:.3f}")
    
    if len(selected_metrics) == 0:
        print("No suitable metrics found for ultimate mixed metric!")
        return df, []
    
    # Filter to only include metrics that exist in the dataset
    available_metrics = [metric for metric in selected_metrics if metric in df.columns]
    print(f"\nAvailable metrics in dataset: {len(available_metrics)}")
    
    if len(available_metrics) == 0:
        print("No selected metrics found in dataset!")
        return df, []
    
    # Calculate weights based on coverage and correlation strength
    metric_weights = {}
    for metric in available_metrics:
        data = metric_coverage[metric]
        # Weight by coverage and average correlation strength
        weight = data['strong_coverage'] * data['avg_abs_correlation']
        metric_weights[metric] = weight
    
    # Normalize weights
    total_weight = sum(metric_weights.values())
    if total_weight > 0:
        metric_weights = {k: v/total_weight for k, v in metric_weights.items()}
    
    print(f"\nUltimate metric weights:")
    for metric, weight in sorted(metric_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {metric}: {weight:.3f}")
    
    # Create ultimate mixed metric
    df_copy = df.copy()
    
    # Standardize all metrics first
    scaler = StandardScaler()
    standardized_metrics = pd.DataFrame(
        scaler.fit_transform(df[available_metrics].fillna(0)),
        columns=available_metrics,
        index=df.index
    )
    
    # Calculate weighted sum
    weighted_sum = np.zeros(len(df))
    for metric in available_metrics:
        weighted_sum += standardized_metrics[metric] * metric_weights[metric]
    
    df_copy['Ultimate_Mixed_Metric'] = weighted_sum
    
    # Also create PCA version
    if len(available_metrics) > 1:
        try:
            pca = PCA(n_components=1)
            pca_result = pca.fit_transform(standardized_metrics)
            df_copy['Ultimate_PCA_Metric'] = pca_result.flatten()
            
            print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}")
        except:
            df_copy['Ultimate_PCA_Metric'] = df_copy['Ultimate_Mixed_Metric']
    else:
        df_copy['Ultimate_PCA_Metric'] = df_copy['Ultimate_Mixed_Metric']
    
    return df_copy, available_metrics

def analyze_ultimate_metric_correlations(df, performance_matrix):
    """Analyze correlations for the ultimate mixed metrics"""
    print("\n" + "="*70)
    print("ANALYZING ULTIMATE METRIC CORRELATIONS")
    print("="*70)
    
    ultimate_metrics = ['Ultimate_Mixed_Metric', 'Ultimate_PCA_Metric']
    categories = performance_matrix.index.tolist()
    subjects = performance_matrix.columns.tolist()
    
    results = []
    
    for metric in ultimate_metrics:
        if metric not in df.columns:
            continue
            
        print(f"\nProcessing {metric}...")
        
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
    
    results_df = pd.DataFrame(results)
    
    # Print results
    for metric in ultimate_metrics:
        if metric in results_df['Metric'].values:
            metric_results = results_df[results_df['Metric'] == metric]
            print(f"\n{metric} Correlations:")
            
            significant_count = 0
            for _, row in metric_results.iterrows():
                sig_marker = "*" if row['Significant'] else ""
                if row['Significant']:
                    significant_count += 1
                print(f"  {row['Subject']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.3f}{sig_marker}")
            
            total_count = len(metric_results)
            print(f"\nSignificance Summary for {metric}:")
            print(f"  Significant correlations: {significant_count}/{total_count} ({significant_count/total_count*100:.1f}%)")
            print(f"  Subjects with significant correlations: {list(metric_results[metric_results['Significant']]['Subject'])}")
    
    return results_df

def main():
    """Main analysis function"""
    print("Starting ultimate correlation analysis...")
    
    # Load expanded dataset
    df = load_expanded_dataset()
    
    # Get all metrics
    metric_cols, model_cols = get_all_analysis_metrics(df)
    
    # Calculate model performance
    performance_matrix = calculate_model_performance(df, model_cols)
    
    print(f"\nPerformance matrix:")
    print(performance_matrix.round(2))
    
    # Analyze comprehensive correlations
    correlation_results = analyze_comprehensive_correlations(df, metric_cols, performance_matrix)
    
    print(f"\nGenerated {len(correlation_results)} correlation analyses")
    
    # Find universal metrics
    metric_coverage, sorted_metrics = find_universal_metrics(correlation_results)
    
    # Create ultimate mixed metric
    df_with_ultimate, selected_metrics = create_ultimate_mixed_metric(correlation_results, metric_coverage, df)
    
    # Analyze ultimate metric correlations
    ultimate_results = analyze_ultimate_metric_correlations(df_with_ultimate, performance_matrix)
    
    # Save results
    correlation_results.to_csv('ultimate_correlation_results.csv', index=False)
    ultimate_results.to_csv('ultimate_metric_correlation_results.csv', index=False)
    df_with_ultimate.to_csv('dataset_v10_ultimate.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"  ultimate_correlation_results.csv")
    print(f"  ultimate_metric_correlation_results.csv")
    print(f"  dataset_v10_ultimate.csv")
    
    return correlation_results, ultimate_results, df_with_ultimate, selected_metrics

if __name__ == "__main__":
    results = main()
