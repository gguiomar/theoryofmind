#!/usr/bin/env python3
"""
Comprehensive Subject Coverage Analysis: Find metrics that span all subjects
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
    """Load and analyze all correlation results to find subject coverage"""
    print("Loading all correlation results...")
    
    # Load both correlation result files
    comprehensive_results = pd.read_csv('comprehensive_correlation_results.csv')
    advanced_results = pd.read_csv('advanced_tom_correlation_results.csv')
    
    # Combine all results
    all_results = pd.concat([comprehensive_results, advanced_results], ignore_index=True)
    
    print(f"Total correlations: {len(all_results)}")
    print(f"Significant correlations: {all_results['Significant'].sum()}")
    
    return all_results

def analyze_subject_coverage(all_results):
    """Analyze which metrics cover which subjects"""
    print("\n" + "="*70)
    print("SUBJECT COVERAGE ANALYSIS")
    print("="*70)
    
    # Get all subjects
    subjects = all_results['Subject'].unique()
    print(f"All subjects: {list(subjects)}")
    
    # Get significant correlations only
    significant_results = all_results[all_results['Significant'] == True]
    
    print(f"\nSignificant correlations by subject:")
    subject_coverage = {}
    
    for subject in subjects:
        subject_sigs = significant_results[significant_results['Subject'] == subject]
        metrics = subject_sigs['Metric'].tolist()
        correlations = subject_sigs['Correlation'].tolist()
        p_values = subject_sigs['P_Value'].tolist()
        
        subject_coverage[subject] = {
            'metrics': metrics,
            'correlations': correlations,
            'p_values': p_values,
            'count': len(metrics)
        }
        
        print(f"\n{subject}: {len(metrics)} significant correlations")
        for i, metric in enumerate(metrics):
            print(f"  {metric}: r = {correlations[i]:.3f}, p = {p_values[i]:.3f}")
    
    return subject_coverage, subjects

def find_spanning_metric_combination(subject_coverage, subjects):
    """Find combination of metrics that spans all subjects"""
    print(f"\n" + "="*70)
    print("FINDING SPANNING METRIC COMBINATION")
    print("="*70)
    
    # Collect all metrics that are significant for any subject
    all_significant_metrics = set()
    for subject_data in subject_coverage.values():
        all_significant_metrics.update(subject_data['metrics'])
    
    print(f"Total unique significant metrics: {len(all_significant_metrics)}")
    
    # For each subject, find which metrics are significant
    subject_to_metrics = {}
    for subject in subjects:
        subject_to_metrics[subject] = set(subject_coverage[subject]['metrics'])
    
    print(f"\nSubject to metrics mapping:")
    for subject, metrics in subject_to_metrics.items():
        print(f"  {subject}: {len(metrics)} metrics - {list(metrics)}")
    
    # Find minimal set of metrics that covers all subjects
    print(f"\nFinding minimal spanning set...")
    
    # Greedy algorithm to find minimal covering set
    uncovered_subjects = set(subjects)
    selected_metrics = []
    
    while uncovered_subjects:
        # Find metric that covers the most uncovered subjects
        best_metric = None
        best_coverage = 0
        best_subjects_covered = set()
        
        for metric in all_significant_metrics:
            if metric in selected_metrics:
                continue
                
            # Count how many uncovered subjects this metric covers
            subjects_covered = set()
            for subject in uncovered_subjects:
                if metric in subject_to_metrics[subject]:
                    subjects_covered.add(subject)
            
            if len(subjects_covered) > best_coverage:
                best_coverage = len(subjects_covered)
                best_metric = metric
                best_subjects_covered = subjects_covered
        
        if best_metric is None:
            print(f"Cannot cover remaining subjects: {uncovered_subjects}")
            break
        
        selected_metrics.append(best_metric)
        uncovered_subjects -= best_subjects_covered
        
        print(f"Selected {best_metric} -> covers {list(best_subjects_covered)}")
        print(f"  Remaining uncovered: {list(uncovered_subjects)}")
    
    print(f"\nMinimal spanning metric set ({len(selected_metrics)} metrics):")
    for metric in selected_metrics:
        print(f"  {metric}")
    
    # Verify coverage
    print(f"\nVerifying coverage:")
    for subject in subjects:
        covered_by = [m for m in selected_metrics if m in subject_to_metrics[subject]]
        print(f"  {subject}: covered by {len(covered_by)} metrics - {covered_by}")
    
    return selected_metrics

def create_spanning_mixed_metric(selected_metrics):
    """Create mixed metric from spanning set"""
    print(f"\n" + "="*70)
    print("CREATING SPANNING MIXED METRIC")
    print("="*70)
    
    # Load dataset
    print("Loading dataset_v4...")
    df = pd.read_csv('./dataset_v4.csv')
    df.columns = df.columns.str.strip()
    
    # Clean data
    df_clean = df[df['ABILITY'].notna()].copy()
    df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
    
    # Filter to only include metrics that exist in the dataset
    available_metrics = [metric for metric in selected_metrics if metric in df_clean.columns]
    print(f"Available spanning metrics in dataset: {len(available_metrics)}")
    for metric in available_metrics:
        print(f"  {metric}")
    
    if len(available_metrics) == 0:
        print("No spanning metrics found in dataset!")
        return df_clean, []
    
    # Load correlation results to get weights
    all_results = load_and_analyze_all_correlations()
    
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
    
    print(f"\nSpanning metric weights:")
    for metric, weight in sorted(metric_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {metric}: {weight:.3f}")
    
    # Create spanning mixed metric
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
    
    df_copy['Spanning_Mixed_Metric'] = weighted_sum
    
    return df_copy, available_metrics

def analyze_spanning_metric_correlations(df, available_metrics):
    """Analyze correlations for the spanning mixed metric"""
    print(f"\n" + "="*70)
    print("ANALYZING SPANNING METRIC CORRELATIONS")
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
    
    # Analyze spanning metric correlations
    subjects = performance_matrix.columns.tolist()
    metric = 'Spanning_Mixed_Metric'
    
    # Calculate metric averages by category
    metric_by_category = df.groupby('Main_Category')[metric].mean()
    
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
                'Metric': metric,
                'Subject': subject,
                'Correlation': corr,
                'P_Value': p_val,
                'Significant': p_val < 0.05,
                'N_Points': len(x)
            })
    
    results_df = pd.DataFrame(results)
    
    print(f"\nSpanning Mixed Metric Correlations:")
    for _, row in results_df.iterrows():
        sig_marker = "*" if row['Significant'] else ""
        print(f"  {row['Subject']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.3f}{sig_marker}")
    
    significant_count = results_df['Significant'].sum()
    total_count = len(results_df)
    
    print(f"\nSignificance Summary:")
    print(f"  Significant correlations: {significant_count}/{total_count} ({significant_count/total_count*100:.1f}%)")
    print(f"  Subjects with significant correlations: {list(results_df[results_df['Significant']]['Subject'])}")
    
    return results_df, performance_matrix

def main():
    """Main analysis function"""
    print("Starting comprehensive subject coverage analysis...")
    
    # Load and analyze all correlations
    all_results = load_and_analyze_all_correlations()
    
    # Analyze subject coverage
    subject_coverage, subjects = analyze_subject_coverage(all_results)
    
    # Find spanning metric combination
    selected_metrics = find_spanning_metric_combination(subject_coverage, subjects)
    
    # Create spanning mixed metric
    df_with_spanning, available_metrics = create_spanning_mixed_metric(selected_metrics)
    
    # Analyze spanning metric correlations
    spanning_results, performance_matrix = analyze_spanning_metric_correlations(df_with_spanning, available_metrics)
    
    # Save results
    spanning_results.to_csv('spanning_metric_correlation_results.csv', index=False)
    df_with_spanning.to_csv('dataset_v6_with_spanning_metric.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"  spanning_metric_correlation_results.csv")
    print(f"  dataset_v6_with_spanning_metric.csv")
    
    return spanning_results, df_with_spanning, selected_metrics, available_metrics

if __name__ == "__main__":
    results = main()
