#!/usr/bin/env python3
"""
Create a focused plot showing only significant correlations from the comprehensive analysis
"""
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_significant_correlations_plot():
    """Create a focused plot showing only significant correlations"""
    
    # Load the correlation results
    print("Loading correlation results...")
    correlation_results = pd.read_csv('comprehensive_correlation_results.csv')
    
    # Filter for significant correlations only
    significant_results = correlation_results[correlation_results['Significant'] == True].copy()
    
    print(f"Found {len(significant_results)} significant correlations out of {len(correlation_results)} total")
    
    if len(significant_results) == 0:
        print("No significant correlations found!")
        return None
    
    # Create pivot table for heatmap
    pivot_data = significant_results.pivot(index='Metric', columns='Subject', values='Correlation')
    pivot_pvals = significant_results.pivot(index='Metric', columns='Subject', values='P_Value')
    
    # Fill NaN values with 0 for visualization (non-significant correlations)
    pivot_data = pivot_data.fillna(0)
    
    # Sort metrics by average absolute correlation (excluding zeros)
    def avg_abs_corr_nonzero(row):
        nonzero_vals = row[row != 0]
        if len(nonzero_vals) == 0:
            return 0
        return nonzero_vals.abs().mean()
    
    avg_abs_corr = pivot_data.apply(avg_abs_corr_nonzero, axis=1).sort_values(ascending=False)
    pivot_data = pivot_data.loc[avg_abs_corr.index]
    
    # Create the plot
    n_metrics = len(pivot_data)
    n_subjects = len(pivot_data.columns)
    
    # Calculate figure size for square cells
    cell_size = 0.8
    fig_width = n_subjects * cell_size + 4
    fig_height = n_metrics * cell_size + 3
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # Create custom colormap that shows zero as white
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 'white', '#fdbf6f', '#ff7f00', '#d94701', '#8c2d04']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom_rdbu', colors, N=n_bins)
    
    # Create heatmap
    sns.heatmap(pivot_data, 
                cmap=cmap,
                center=0,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8},
                linewidths=1.0,
                linecolor='lightgray',
                ax=ax,
                annot=True,  # Show correlation values
                fmt='.3f',
                annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                square=True)
    
    # Customize the plot
    ax.set_xlabel('Subject', fontweight='bold', fontsize=14)
    ax.set_ylabel('Metrics', fontweight='bold', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', rotation=0, labelsize=12)
    
    plt.tight_layout()
    plt.show()
    
    
    print("\nAll significant correlations (sorted by absolute correlation):")
    significant_sorted = significant_results.sort_values('Correlation', key=abs, ascending=False)
    for _, row in significant_sorted.iterrows():
        direction = "positive" if row['Correlation'] > 0 else "negative"
        print(f"  {row['Metric']} vs {row['Subject']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.3f} ({direction})")
    
    print(f"\nMetrics with significant correlations:")
    metric_counts = significant_results.groupby('Metric').size().sort_values(ascending=False)
    for metric, count in metric_counts.items():
        print(f"  {metric}: {count} significant correlation(s)")
    
    print(f"\nSubjects with significant correlations:")
    subject_counts = significant_results.groupby('Subject').size().sort_values(ascending=False)
    for subject, count in subject_counts.items():
        print(f"  {subject}: {count} significant correlation(s)")
    
    # Save the plot
    plt.savefig('significant_correlations_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: significant_correlations_heatmap.png")
    
    return fig, significant_results

if __name__ == "__main__":
    fig, significant_results = create_significant_correlations_plot()

# %%
