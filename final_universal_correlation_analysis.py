#!/usr/bin/env python3
"""
Final Universal Correlation Analysis
Testing next-generation multiplicative and dynamic complexity measures
for true universal correlation across ALL subjects including 70B
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

def load_next_gen_dataset():
    """Load the next-generation dataset"""
    print("Loading dataset_v11_next_gen...")
    df = pd.read_csv('./dataset_v11_next_gen.csv')
    df.columns = df.columns.str.strip()
    
    # Clean data
    df_clean = df[df['ABILITY'].notna()].copy()
    df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
    
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Categories: {df_clean['Main_Category'].unique()}")
    
    return df_clean

def get_next_gen_metrics(df):
    """Get next-generation metrics for analysis"""
    # Focus on the most sophisticated multiplicative and dynamic metrics
    next_gen_metrics = [
        # Multiplicative complexity indices
        'Story_Cognitive_Load_Index',
        'Story_Mental_State_Interaction', 
        'Story_Inference_Complexity',
        'Question_Cognitive_Load_Index',
        'Question_Mental_State_Interaction',
        'Question_Inference_Complexity',
        
        # Dynamic complexity measures
        'Story_Complexity_Gradient',
        'Story_Complexity_Variance',
        'Story_Working_Memory_Load',
        'Story_Attention_Switching_Cost',
        'Question_Complexity_Gradient',
        'Question_Working_Memory_Load',
        
        # Meta-cognitive complexity
        'Story_Recursive_Mental_State_Depth',
        'Story_Meta_Uncertainty_Index',
        'Story_Cognitive_Interference_Score',
        'Question_Recursive_Mental_State_Depth',
        'Question_Meta_Uncertainty_Index',
        
        # Core dimensions (enhanced)
        'Story_Entity_Density',
        'Story_Causal_Depth',
        'Story_Uncertainty_Level',
        'Story_Temporal_Complexity',
        'Question_Entity_Density',
        'Question_Causal_Depth',
        'Question_Uncertainty_Level',
        
        # Component features that showed promise
        'Story_Mental_State_Depth',
        'Story_Entity_Switches',
        'Story_Emotional_Transitions',
        'Question_Mental_State_Depth',
        'Question_Perspective_Shifts'
    ]
    
    # Filter to only include metrics that exist and have variance
    available_metrics = []
    for metric in next_gen_metrics:
        if metric in df.columns and df[metric].std() > 0:
            available_metrics.append(metric)
    
    print(f"Available next-generation metrics: {len(available_metrics)}")
    for metric in available_metrics:
        non_zero_pct = (df[metric] > 0).sum() / len(df) * 100
        print(f"  {metric}: {non_zero_pct:.1f}% non-zero, std = {df[metric].std():.6f}")
    
    return available_metrics

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

def analyze_next_gen_correlations(df, next_gen_metrics, performance_matrix):
    """Analyze correlations for next-generation metrics"""
    categories = performance_matrix.index.tolist()
    subjects = performance_matrix.columns.tolist()
    
    results = []
    
    print(f"\nAnalyzing correlations for {len(next_gen_metrics)} next-generation metrics...")
    
    for metric in next_gen_metrics:
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

def find_70b_breakthrough_metrics(correlation_results):
    """Find metrics that finally achieve significance with 70B model"""
    print("\n" + "="*70)
    print("SEARCHING FOR 70B BREAKTHROUGH METRICS")
    print("="*70)
    
    # Focus on 70B model specifically
    llama_70b_results = correlation_results[correlation_results['Subject'] == 'Llama 3.1 70B']
    
    print(f"70B correlations found: {len(llama_70b_results)}")
    
    # Sort by absolute correlation strength
    llama_70b_sorted = llama_70b_results.sort_values('Abs_Correlation', ascending=False)
    
    print(f"\nTop 10 strongest 70B correlations:")
    for i, (_, row) in enumerate(llama_70b_sorted.head(10).iterrows()):
        sig_marker = "*" if row['Significant'] else ""
        print(f"  {i+1}. {row['Metric']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.3f}{sig_marker}")
    
    # Check for significant correlations
    significant_70b = llama_70b_results[llama_70b_results['Significant']]
    
    if len(significant_70b) > 0:
        print(f"\n🎉 BREAKTHROUGH! Found {len(significant_70b)} significant 70B correlations:")
        for _, row in significant_70b.iterrows():
            print(f"  ⭐ {row['Metric']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.3f}")
    else:
        print(f"\n❌ No significant 70B correlations yet. Strongest correlation: {llama_70b_sorted.iloc[0]['Abs_Correlation']:.3f}")
    
    return llama_70b_results, significant_70b

def create_final_universal_metric(correlation_results, df):
    """Create the final universal metric using all successful approaches"""
    print("\n" + "="*70)
    print("CREATING FINAL UNIVERSAL METRIC")
    print("="*70)
    
    # Find metrics with strongest correlations across all subjects
    metric_scores = {}
    
    subjects = correlation_results['Subject'].unique()
    
    for metric in correlation_results['Metric'].unique():
        metric_data = correlation_results[correlation_results['Metric'] == metric]
        
        # Calculate comprehensive score
        avg_abs_corr = metric_data['Abs_Correlation'].mean()
        max_abs_corr = metric_data['Abs_Correlation'].max()
        significant_count = metric_data['Significant'].sum()
        subject_coverage = len(metric_data)
        
        # Weighted score prioritizing 70B performance
        llama_70b_data = metric_data[metric_data['Subject'] == 'Llama 3.1 70B']
        llama_70b_score = llama_70b_data['Abs_Correlation'].iloc[0] if len(llama_70b_data) > 0 else 0
        
        # Comprehensive score
        score = (avg_abs_corr * 0.3 + 
                max_abs_corr * 0.2 + 
                significant_count * 0.2 + 
                subject_coverage * 0.1 + 
                llama_70b_score * 0.2)  # Extra weight for 70B
        
        metric_scores[metric] = {
            'score': score,
            'avg_abs_corr': avg_abs_corr,
            'max_abs_corr': max_abs_corr,
            'significant_count': significant_count,
            'subject_coverage': subject_coverage,
            'llama_70b_score': llama_70b_score
        }
    
    # Sort by comprehensive score
    sorted_metrics = sorted(metric_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print(f"Top 15 metrics by comprehensive score:")
    for i, (metric, data) in enumerate(sorted_metrics[:15]):
        print(f"  {i+1}. {metric}")
        print(f"     Score: {data['score']:.3f} | Avg |r|: {data['avg_abs_corr']:.3f} | 70B |r|: {data['llama_70b_score']:.3f}")
        print(f"     Significant: {data['significant_count']}/{data['subject_coverage']} subjects")
    
    # Select top metrics for final combination
    top_metrics = [metric for metric, data in sorted_metrics[:10]]
    
    # Filter to only include metrics that exist in the dataset
    available_metrics = [metric for metric in top_metrics if metric in df.columns and df[metric].std() > 0]
    
    print(f"\nSelected {len(available_metrics)} metrics for final universal metric:")
    for metric in available_metrics:
        print(f"  {metric}")
    
    if len(available_metrics) == 0:
        print("No suitable metrics found!")
        return df, []
    
    # Create final metric using multiple approaches
    df_copy = df.copy()
    
    # Standardize all metrics
    scaler = StandardScaler()
    standardized_metrics = pd.DataFrame(
        scaler.fit_transform(df[available_metrics].fillna(0)),
        columns=available_metrics,
        index=df.index
    )
    
    # Approach 1: Weighted by comprehensive score
    metric_weights = {}
    for metric in available_metrics:
        if metric in metric_scores:
            metric_weights[metric] = metric_scores[metric]['score']
        else:
            metric_weights[metric] = 0.1
    
    # Normalize weights
    total_weight = sum(metric_weights.values())
    if total_weight > 0:
        metric_weights = {k: v/total_weight for k, v in metric_weights.items()}
    
    # Calculate weighted sum
    weighted_sum = np.zeros(len(df))
    for metric in available_metrics:
        weighted_sum += standardized_metrics[metric] * metric_weights[metric]
    
    df_copy['Final_Universal_Metric_Weighted'] = weighted_sum
    
    # Approach 2: PCA
    if len(available_metrics) > 1:
        try:
            pca = PCA(n_components=1)
            pca_result = pca.fit_transform(standardized_metrics)
            df_copy['Final_Universal_Metric_PCA'] = pca_result.flatten()
            
            print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}")
        except:
            df_copy['Final_Universal_Metric_PCA'] = df_copy['Final_Universal_Metric_Weighted']
    
    # Approach 3: Multiplicative (for non-zero values)
    multiplicative_product = np.ones(len(df))
    for metric in available_metrics:
        # Add 1 to avoid zeros, then subtract 1 at the end
        normalized_values = (standardized_metrics[metric] - standardized_metrics[metric].min() + 1)
        multiplicative_product *= normalized_values
    
    df_copy['Final_Universal_Metric_Multiplicative'] = multiplicative_product
    
    print(f"\nFinal metric weights:")
    for metric, weight in sorted(metric_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {metric}: {weight:.3f}")
    
    return df_copy, available_metrics

def analyze_final_metric_correlations(df, performance_matrix):
    """Analyze correlations for the final universal metrics"""
    print("\n" + "="*70)
    print("ANALYZING FINAL UNIVERSAL METRIC CORRELATIONS")
    print("="*70)
    
    final_metrics = [
        'Final_Universal_Metric_Weighted',
        'Final_Universal_Metric_PCA', 
        'Final_Universal_Metric_Multiplicative'
    ]
    
    categories = performance_matrix.index.tolist()
    subjects = performance_matrix.columns.tolist()
    
    results = []
    
    for metric in final_metrics:
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
    
    # Print results with special focus on 70B
    for metric in final_metrics:
        if metric in results_df['Metric'].values:
            metric_results = results_df[results_df['Metric'] == metric]
            print(f"\n🎯 {metric} Correlations:")
            
            significant_count = 0
            llama_70b_significant = False
            
            for _, row in metric_results.iterrows():
                sig_marker = "*" if row['Significant'] else ""
                if row['Significant']:
                    significant_count += 1
                    if row['Subject'] == 'Llama 3.1 70B':
                        llama_70b_significant = True
                
                # Highlight 70B results
                if row['Subject'] == 'Llama 3.1 70B':
                    if row['Significant']:
                        print(f"  🎉 {row['Subject']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.3f}{sig_marker} ⭐ BREAKTHROUGH!")
                    else:
                        print(f"  🔍 {row['Subject']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.3f}{sig_marker}")
                else:
                    print(f"     {row['Subject']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.3f}{sig_marker}")
            
            total_count = len(metric_results)
            print(f"\n   Summary for {metric}:")
            print(f"   Significant correlations: {significant_count}/{total_count} ({significant_count/total_count*100:.1f}%)")
            print(f"   70B significant: {'YES ⭐' if llama_70b_significant else 'NO'}")
            print(f"   Subjects with significant correlations: {list(metric_results[metric_results['Significant']]['Subject'])}")
    
    return results_df

def main():
    """Main analysis function"""
    print("Starting final universal correlation analysis...")
    
    # Load next-generation dataset
    df = load_next_gen_dataset()
    
    # Get next-generation metrics
    next_gen_metrics = get_next_gen_metrics(df)
    
    # Calculate model performance
    performance_matrix = calculate_model_performance(df)
    
    print(f"\nPerformance matrix:")
    print(performance_matrix.round(2))
    
    # Analyze next-generation correlations
    correlation_results = analyze_next_gen_correlations(df, next_gen_metrics, performance_matrix)
    
    print(f"\nGenerated {len(correlation_results)} correlation analyses")
    
    # Search for 70B breakthrough
    llama_70b_results, significant_70b = find_70b_breakthrough_metrics(correlation_results)
    
    # Create final universal metric
    df_with_final, selected_metrics = create_final_universal_metric(correlation_results, df)
    
    # Analyze final metric correlations
    final_results = analyze_final_metric_correlations(df_with_final, performance_matrix)
    
    # Save results
    correlation_results.to_csv('final_correlation_results.csv', index=False)
    final_results.to_csv('final_universal_metric_results.csv', index=False)
    df_with_final.to_csv('dataset_v12_final_universal.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"  final_correlation_results.csv")
    print(f"  final_universal_metric_results.csv") 
    print(f"  dataset_v12_final_universal.csv")
    
    return correlation_results, final_results, df_with_final, selected_metrics

if __name__ == "__main__":
    results = main()
