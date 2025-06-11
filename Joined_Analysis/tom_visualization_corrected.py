#!/usr/bin/env python3
"""
Corrected Theory of Mind Analysis Visualization Class

This class shows the relationship between:
1. Model performance (accuracy) and question characteristics (analysis metrics)
2. How different analysis metrics correlate with model performance
3. Model performance across ability groups and their question characteristics

Uses seaborn rocket color palette throughout.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style and rocket color palette
sns.set_style("whitegrid")
sns.set_palette("rocket")

class ToMVisualizationCorrected:
    """
    Corrected visualization class showing model performance vs question characteristics.
    """
    
    def __init__(self, dataset_path='dataset_joined_corrected.csv'):
        """Initialize the visualization class with the corrected dataset."""
        print("Loading dataset for corrected visualization...")
        self.df = pd.read_csv(dataset_path)
        
        # Define column mappings with all 6 models
        self.model_columns = [
            'meta_llama_Llama_3.1_70B_Instruct',
            'Qwen_Qwen2.5_32B_Instruct',
            'allenai_OLMo_2_1124_13B_Instruct', 
            'mistralai_Mistral_7B_Instruct_v0.3',
            'microsoft_Phi_3_mini_4k_instruct',
            'internlm_internlm2_5_1_8b_chat'
        ]
        
        # Simplified model names for plotting
        self.model_names = [
            'Meta Llama 3.1 70B',
            'Qwen 2.5 32B',
            'OLMo 13B',
            'Mistral 7B',
            'Phi 3 Mini',
            'InternLM 1.8B'
        ]
        
        # Analysis metrics (question characteristics)
        self.analysis_metrics = {
            'Idea Density': 'Idea_Density',
            'Question Complexity': 'Question_Complexity_Score',
            'Syntactic Complexity': 'Q_Syntactic_Complexity',
            'Semantic Complexity': 'Q_Semantic_Complexity',
            'ToM Complexity': 'Q_ToM_Complexity',
            'Reasoning Complexity': 'Q_Reasoning_Complexity',
            'RST EDUs': 'num_edus',
            'RST Tree Depth': 'tree_depth',
            'RST Attribution': 'rel_attribution',
            'RST Causal': 'rel_causal',
            'RST Explanation': 'rel_explanation'
        }
        
        self.ability_column = '\nABILITY'
        
        # Group abilities into main categories
        self.ability_groups = self._group_abilities()
        
        # Calculate model performance for each sample
        self._calculate_model_performance()
        
        print(f"✓ Dataset loaded: {len(self.df)} samples")
        print(f"✓ Models: {len(self.model_columns)}")
        print(f"✓ Analysis metrics: {len(self.analysis_metrics)}")
        print(f"✓ Ability groups: {len(self.ability_groups)}")
    
    def _group_abilities(self):
        """Group abilities into main categories for better visualization."""
        abilities = self.df[self.ability_column].unique()
        
        groups = {
            'Emotion': [],
            'Belief': [],
            'Desire': [],
            'Intention': [],
            'Knowledge': [],
            'Non-Literal Communication': []
        }
        
        for ability in abilities:
            if 'Emotion:' in ability:
                groups['Emotion'].append(ability)
            elif 'Belief:' in ability:
                groups['Belief'].append(ability)
            elif 'Desire:' in ability:
                groups['Desire'].append(ability)
            elif 'Intention:' in ability:
                groups['Intention'].append(ability)
            elif 'Knowledge:' in ability:
                groups['Knowledge'].append(ability)
            elif 'Non-Literal Communication:' in ability:
                groups['Non-Literal Communication'].append(ability)
        
        return groups
    
    def _calculate_model_performance(self):
        """Calculate binary performance (correct/incorrect) for each model on each sample."""
        print("Calculating model performance for each sample...")
        
        for i, model_col in enumerate(self.model_columns):
            model_name = self.model_names[i]
            perf_col = f'{model_name}_Performance'
            
            if model_col in self.df.columns and '\nANSWER' in self.df.columns:
                # Binary performance: 1 if correct, 0 if incorrect
                self.df[perf_col] = (self.df[model_col] == self.df['\nANSWER']).astype(int)
            else:
                self.df[perf_col] = 0
        
        print("✓ Model performance calculated for each sample")
    
    def plot_performance_vs_complexity_bins(self, figsize=(16, 12), save_path=None):
        """Show model performance across complexity bins for key metrics."""
        # Select key metrics for binning
        key_metrics = {
            'Question Complexity': 'Question_Complexity_Score',
            'Idea Density': 'Idea_Density',
            'RST EDUs': 'num_edus',
            'ToM Complexity': 'Q_ToM_Complexity'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (metric_name, metric_col) in enumerate(key_metrics.items()):
            if idx >= len(axes) or metric_col not in self.df.columns:
                continue
                
            ax = axes[idx]
            
            # Create quartile bins, handling duplicate values
            try:
                bins = pd.qcut(self.df[metric_col], q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'], duplicates='drop')
            except ValueError:
                # If still fails, use regular cut instead
                bins = pd.cut(self.df[metric_col], bins=4, labels=['Low', 'Med-Low', 'Med-High', 'High'])
            
            # Calculate performance for each model in each bin
            performance_data = []
            
            for bin_label in ['Low', 'Med-Low', 'Med-High', 'High']:
                mask = bins == bin_label
                subset = self.df[mask]
                
                if len(subset) == 0:
                    continue
                
                for model_name in self.model_names:
                    perf_col = f'{model_name}_Performance'
                    if perf_col in subset.columns:
                        accuracy = subset[perf_col].mean()
                        performance_data.append({
                            'Complexity_Bin': bin_label,
                            'Model': model_name,
                            'Accuracy': accuracy,
                            'Count': len(subset)
                        })
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                
                # Create grouped bar plot
                sns.barplot(data=perf_df, x='Complexity_Bin', y='Accuracy', 
                           hue='Model', ax=ax, palette='rocket')
                
                ax.set_title(f'Model Performance by {metric_name}', fontweight='bold')
                ax.set_xlabel(f'{metric_name} Level', fontweight='bold')
                ax.set_ylabel('Accuracy', fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Model Performance Across Question Complexity Levels', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_metric_correlations(self, figsize=(14, 10), save_path=None):
        """Show correlation between model performance and analysis metrics."""
        # Calculate correlations for each model
        correlation_data = []
        
        for model_name in self.model_names:
            perf_col = f'{model_name}_Performance'
            
            if perf_col not in self.df.columns:
                continue
            
            for metric_name, metric_col in self.analysis_metrics.items():
                if metric_col in self.df.columns:
                    # Calculate correlation
                    corr, p_value = pearsonr(self.df[perf_col], self.df[metric_col])
                    
                    correlation_data.append({
                        'Model': model_name,
                        'Metric': metric_name,
                        'Correlation': corr,
                        'P_Value': p_value,
                        'Significant': p_value < 0.05
                    })
        
        if not correlation_data:
            print("No correlation data available")
            return
        
        corr_df = pd.DataFrame(correlation_data)
        
        # Create pivot table for heatmap
        pivot_corr = corr_df.pivot(index='Model', columns='Metric', values='Correlation')
        
        # Create heatmap
        plt.figure(figsize=figsize)
        
        # Create mask for non-significant correlations
        pivot_pval = corr_df.pivot(index='Model', columns='Metric', values='P_Value')
        mask = pivot_pval > 0.05
        
        sns.heatmap(pivot_corr, annot=True, cmap='rocket', center=0,
                   square=True, cbar_kws={'shrink': 0.8}, fmt='.3f',
                   mask=mask, linewidths=0.5)
        
        plt.title('Model Performance Correlation with Question Characteristics\n(Only significant correlations shown, p<0.05)', 
                 fontweight='bold', fontsize=14)
        plt.xlabel('Analysis Metrics', fontweight='bold')
        plt.ylabel('Models', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return corr_df
    
    def plot_ability_group_performance_analysis(self, figsize=(16, 12), save_path=None):
        """Show model performance and question characteristics by ability group."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (group_name, abilities) in enumerate(self.ability_groups.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Filter data for this ability group
            group_data = self.df[self.df[self.ability_column].isin(abilities)]
            
            if len(group_data) == 0:
                ax.set_title(f'{group_name}\n(No data)', fontsize=12)
                continue
            
            # Calculate model performance for this group
            performance_data = []
            
            for model_name in self.model_names:
                perf_col = f'{model_name}_Performance'
                if perf_col in group_data.columns:
                    accuracy = group_data[perf_col].mean()
                    performance_data.append({
                        'Model': model_name,
                        'Accuracy': accuracy
                    })
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                
                # Create bar plot
                sns.barplot(data=perf_df, x='Model', y='Accuracy', ax=ax, palette='rocket')
                
                ax.set_title(f'{group_name}\n({len(group_data)} samples)', 
                           fontweight='bold', fontsize=10)
                ax.set_xlabel('')
                ax.set_ylabel('Accuracy', fontweight='bold')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                ax.grid(axis='y', alpha=0.3)
                
                # Add mean complexity as text
                if 'Question_Complexity_Score' in group_data.columns:
                    mean_complexity = group_data['Question_Complexity_Score'].mean()
                    ax.text(0.02, 0.98, f'Avg Q Complexity: {mean_complexity:.2f}', 
                           transform=ax.transAxes, fontsize=8, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for idx in range(len(self.ability_groups), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Model Performance by Theory of Mind Ability Groups', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_scatter_performance_vs_metrics(self, figsize=(16, 12), save_path=None):
        """Create scatter plots showing performance vs key metrics for each model."""
        # Select key metrics
        key_metrics = ['Question_Complexity_Score', 'Idea_Density', 'num_edus', 'Q_ToM_Complexity']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, metric_col in enumerate(key_metrics):
            if idx >= len(axes) or metric_col not in self.df.columns:
                continue
                
            ax = axes[idx]
            
            # Create bins for the metric to calculate average performance
            bins = pd.cut(self.df[metric_col], bins=10)
            bin_centers = []
            
            for model_name in self.model_names:
                perf_col = f'{model_name}_Performance'
                if perf_col not in self.df.columns:
                    continue
                
                # Calculate mean performance for each bin
                bin_performance = []
                bin_centers = []
                
                for bin_interval in bins.cat.categories:
                    mask = bins == bin_interval
                    subset = self.df[mask]
                    
                    if len(subset) > 0:
                        bin_centers.append(bin_interval.mid)
                        bin_performance.append(subset[perf_col].mean())
                
                if bin_centers and bin_performance:
                    ax.plot(bin_centers, bin_performance, 'o-', 
                           label=model_name, linewidth=2, markersize=6, alpha=0.8)
            
            metric_name = [k for k, v in self.analysis_metrics.items() if v == metric_col][0]
            ax.set_title(f'Performance vs {metric_name}', fontweight='bold')
            ax.set_xlabel(metric_name, fontweight='bold')
            ax.set_ylabel('Accuracy', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Model Performance vs Question Characteristics', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_circular_ability_performance(self, figsize=(12, 12), save_path=None):
        """Create circular plot with model performance across ToM ability groups."""
        # Calculate performance for each model in each ability group
        ability_performance = {}
        ability_names = list(self.ability_groups.keys())
        
        for group_name, abilities in self.ability_groups.items():
            group_data = self.df[self.df[self.ability_column].isin(abilities)]
            
            if len(group_data) == 0:
                continue
            
            for model_name in self.model_names:
                perf_col = f'{model_name}_Performance'
                if perf_col in group_data.columns:
                    accuracy = group_data[perf_col].mean()
                    
                    if model_name not in ability_performance:
                        ability_performance[model_name] = []
                    ability_performance[model_name].append(accuracy)
        
        # Create circular plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Set up angles for each ability group
        angles = np.linspace(0, 2 * np.pi, len(ability_names), endpoint=False)
        
        # Colors for different models using rocket palette
        colors = sns.color_palette("rocket", len(self.model_names))
        
        # Plot each model
        for idx, (model_name, performances) in enumerate(ability_performance.items()):
            if len(performances) == len(ability_names):
                # Close the circle
                performances_closed = performances + [performances[0]]
                angles_closed = np.concatenate([angles, [angles[0]]])
                
                # Plot line and fill
                ax.plot(angles_closed, performances_closed, 'o-', 
                       linewidth=3, markersize=8, label=model_name, 
                       color=colors[idx], alpha=0.8)
                ax.fill(angles_closed, performances_closed, 
                       color=colors[idx], alpha=0.1)
        
        # Customize the plot
        ax.set_xticks(angles)
        ax.set_xticklabels(ability_names, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add radial labels
        ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_rlabel_position(0)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        # Add title
        plt.title('Model Performance Across Theory of Mind Ability Groups', 
                 fontsize=16, fontweight='bold', pad=30)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return ability_performance
    
    def generate_all_plots(self, output_dir='plots_corrected'):
        """Generate all corrected visualization plots and save them."""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating corrected ToM analysis visualizations...")
        
        # 1. Performance vs complexity bins
        print("1. Creating performance vs complexity bins...")
        self.plot_performance_vs_complexity_bins(
            save_path=f'{output_dir}/performance_vs_complexity_bins.png'
        )
        
        # 2. Performance-metric correlations
        print("2. Creating performance-metric correlations...")
        corr_df = self.plot_performance_metric_correlations(
            save_path=f'{output_dir}/performance_metric_correlations.png'
        )
        
        # Save correlation data
        if corr_df is not None:
            corr_df.to_csv(f'{output_dir}/performance_metric_correlations.csv', index=False)
        
        # 3. Ability group performance analysis
        print("3. Creating ability group performance analysis...")
        self.plot_ability_group_performance_analysis(
            save_path=f'{output_dir}/ability_group_performance.png'
        )
        
        # 4. Scatter plots
        print("4. Creating scatter plots...")
        self.plot_scatter_performance_vs_metrics(
            save_path=f'{output_dir}/scatter_performance_vs_metrics.png'
        )
        
        # 5. Circular ability performance plot
        print("5. Creating circular ability performance plot...")
        ability_perf = self.plot_circular_ability_performance(
            save_path=f'{output_dir}/circular_ability_performance.png'
        )
        
        print(f"\n✓ All corrected plots saved to '{output_dir}' directory")
        print("Generated files:")
        print("- performance_vs_complexity_bins.png")
        print("- performance_metric_correlations.png")
        print("- performance_metric_correlations.csv")
        print("- ability_group_performance.png")
        print("- scatter_performance_vs_metrics.png")
        print("- circular_ability_performance.png")
        
        return corr_df


def main():
    """Main function to demonstrate the corrected visualization class."""
    print("=" * 60)
    print("CORRECTED THEORY OF MIND ANALYSIS VISUALIZATION")
    print("=" * 60)
    
    # Initialize visualization class
    viz = ToMVisualizationCorrected()
    
    # Generate all plots
    corr_df = viz.generate_all_plots()
    
    print("\n" + "=" * 60)
    print("CORRECTED VISUALIZATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
