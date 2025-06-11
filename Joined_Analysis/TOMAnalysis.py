#!/usr/bin/env python3
"""
Simplified Theory of Mind Analysis Class

Contains only the essential plots:
1. Circular ability performance (polar bar chart style)
2. Comprehensive correlation matrix (with significance shading and model ranking)
3. Single correlation matrix for submeasures
4. Scatter performance vs metrics (clean styling)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set clean styling
sns.set_style("white")
sns.set_palette("crest")

class TOMAnalysis:
    """
    Simplified Theory of Mind analysis with essential visualizations.
    """
    
    def __init__(self, dataset_path='dataset_joined_corrected.csv'):
        """Initialize with the corrected dataset."""
        print("Loading dataset for TOM analysis...")
        self.df = pd.read_csv(dataset_path)
        
        # Model definitions
        self.model_columns = [
            'meta_llama_Llama_3.1_70B_Instruct',
            'Qwen_Qwen2.5_32B_Instruct',
            'allenai_OLMo_2_1124_13B_Instruct', 
            'mistralai_Mistral_7B_Instruct_v0.3',
            'microsoft_Phi_3_mini_4k_instruct',
            'internlm_internlm2_5_1_8b_chat'
        ]
        
        self.model_names = [
            'Meta Llama 3.1 70B',
            'Qwen 2.5 32B',
            'OLMo 13B',
            'Mistral 7B',
            'Phi 3 Mini',
            'InternLM 1.8B'
        ]
        
        # Analysis metrics
        self.all_metrics = {
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
        self.ability_groups = self._group_abilities()
        self._calculate_model_performance()
        
        print(f"✓ Dataset loaded: {len(self.df)} samples")
        print(f"✓ Models: {len(self.model_columns)}")
        print(f"✓ Metrics: {len(self.all_metrics)}")
        print(f"✓ Ability groups: {len(self.ability_groups)}")
    
    def _group_abilities(self):
        """Group abilities into main categories."""
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
        """Calculate binary performance for each model."""
        for i, model_col in enumerate(self.model_columns):
            model_name = self.model_names[i]
            perf_col = f'{model_name}_Performance'
            
            if model_col in self.df.columns and '\nANSWER' in self.df.columns:
                self.df[perf_col] = (self.df[model_col] == self.df['\nANSWER']).astype(int)
            else:
                self.df[perf_col] = 0
    
    def circular_ability_performance(self, figsize=(20, 10), save_path=None):
        """Create polar bar chart for ability group performance."""
        # Calculate performance for each model in each ability group
        ability_names = list(self.ability_groups.keys())
        model_performances = {}
        
        for model_name in self.model_names:
            perf_col = f'{model_name}_Performance'
            performances = []
            
            for group_name, abilities in self.ability_groups.items():
                group_data = self.df[self.df[self.ability_column].isin(abilities)]
                if len(group_data) > 0 and perf_col in group_data.columns:
                    accuracy = group_data[perf_col].mean()
                    performances.append(accuracy * 100)  # Convert to percentage
                else:
                    performances.append(0)
            
            model_performances[model_name] = performances
        
        # Create polar bar chart
        plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar=True)
        plt.axis('off')
        
        # Set limits
        upperLimit = 100
        lowerLimit = 30
        
        # Calculate positions
        n_groups = len(ability_names)
        n_models = len(self.model_names)
        width = 2 * np.pi / (n_groups * n_models)
        
        colors = sns.color_palette("crest", n_models)
        
        for model_idx, (model_name, performances) in enumerate(model_performances.items()):
            # Compute heights
            max_perf = max(max(model_performances.values(), key=max))
            slope = (upperLimit - lowerLimit) / max_perf
            heights = [slope * perf + lowerLimit for perf in performances]
            
            # Compute angles for this model
            base_angles = [i * (2 * np.pi / n_groups) for i in range(n_groups)]
            angles = [angle + model_idx * width for angle in base_angles]
            
            # Draw bars
            bars = ax.bar(
                x=angles,
                height=heights,
                width=width * 0.8,
                bottom=lowerLimit,
                linewidth=2,
                edgecolor="white",
                color=colors[model_idx],
                alpha=0.8,
                label=model_name
            )
        
        # Add labels
        label_angles = [i * (2 * np.pi / n_groups) for i in range(n_groups)]
        ax.set_xticks(label_angles)
        ax.set_xticklabels(ability_names, fontsize=14, fontweight='bold')
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        plt.title('Model Performance Across ToM Ability Groups', 
                 fontsize=18, fontweight='bold', pad=30)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def comprehensive_correlation_matrix(self, figsize=(16, 12), save_path=None):
        """Correlation matrix with significance shading and model ranking."""
        # Calculate correlations
        correlation_data = []
        
        for model_name in self.model_names:
            perf_col = f'{model_name}_Performance'
            if perf_col not in self.df.columns:
                continue
            
            for metric_name, metric_col in self.all_metrics.items():
                if metric_col in self.df.columns:
                    corr, p_value = pearsonr(self.df[perf_col], self.df[metric_col])
                    correlation_data.append({
                        'Model': model_name,
                        'Metric': metric_name,
                        'Correlation': corr,
                        'P_Value': p_value,
                        'Significant': p_value < 0.05
                    })
        
        corr_df = pd.DataFrame(correlation_data)
        
        # Calculate overall performance for ranking
        model_overall_perf = {}
        for model_name in self.model_names:
            perf_col = f'{model_name}_Performance'
            if perf_col in self.df.columns:
                model_overall_perf[model_name] = self.df[perf_col].mean()
        
        # Sort models by performance
        sorted_models = sorted(model_overall_perf.keys(), 
                             key=lambda x: model_overall_perf[x], reverse=True)
        
        # Create pivot tables
        pivot_corr = corr_df.pivot(index='Model', columns='Metric', values='Correlation')
        pivot_pval = corr_df.pivot(index='Model', columns='Metric', values='P_Value')
        
        # Reorder by performance
        pivot_corr = pivot_corr.reindex(sorted_models)
        pivot_pval = pivot_pval.reindex(sorted_models)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create significance mask (non-significant correlations will be lighter)
        significance_mask = pivot_pval > 0.05
        
        # Plot heatmap
        sns.heatmap(pivot_corr, annot=True, cmap='crest', center=0,
                   square=True, cbar_kws={'shrink': 0.8}, fmt='.3f',
                   linewidths=0.5, ax=ax, annot_kws={'size': 10})
        
        # Add significance overlay
        for i in range(len(pivot_corr.index)):
            for j in range(len(pivot_corr.columns)):
                if significance_mask.iloc[i, j]:
                    # Add white overlay for non-significant correlations
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, 
                                             fill=True, color='white', alpha=0.5))
        
        # Clean styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        ax.set_title('Model Performance Correlations (Ranked by Overall Performance)\nShaded areas indicate non-significant correlations (p>0.05)', 
                    fontweight='bold', fontsize=16, pad=20)
        ax.set_xlabel('Analysis Metrics', fontweight='bold', fontsize=14)
        ax.set_ylabel('Models (Ranked by Performance)', fontweight='bold', fontsize=14)
        
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_df
    
    def single_correlation_matrix(self, figsize=(12, 10), save_path=None):
        """Single correlation matrix for all submeasures."""
        # Get all analysis metric columns
        all_analysis_cols = [col for col in self.all_metrics.values() if col in self.df.columns]
        
        if len(all_analysis_cols) < 2:
            print("Not enough analysis metrics for correlation")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[all_analysis_cols].corr()
        
        # Create readable labels
        readable_labels = []
        for col in all_analysis_cols:
            for readable_name, col_name in self.all_metrics.items():
                if col_name == col:
                    readable_labels.append(readable_name)
                    break
            else:
                readable_labels.append(col)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(corr_matrix, annot=True, cmap='crest', center=0,
                   square=True, cbar_kws={'shrink': 0.8},
                   fmt='.2f', annot_kws={'size': 10}, linewidths=0.5, ax=ax)
        
        # Clean styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        ax.set_title('Analysis Submeasure Correlations', 
                    fontweight='bold', fontsize=16, pad=20)
        
        ax.set_xticklabels(readable_labels, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(readable_labels, rotation=0, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_matrix
    
    def scatter_performance_vs_metrics(self, figsize=(16, 12), save_path=None):
        """Scatter plots with clean styling."""
        key_metrics = ['Question_Complexity_Score', 'Idea_Density', 'num_edus', 'Q_ToM_Complexity']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        colors = sns.color_palette("crest", len(self.model_names))
        
        for idx, metric_col in enumerate(key_metrics):
            if idx >= len(axes) or metric_col not in self.df.columns:
                continue
                
            ax = axes[idx]
            
            # Create bins for the metric
            bins = pd.cut(self.df[metric_col], bins=10)
            
            for model_idx, model_name in enumerate(self.model_names):
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
                           label=model_name, linewidth=3, markersize=8, 
                           color=colors[model_idx], alpha=0.8)
            
            # Find metric name
            metric_name = [k for k, v in self.all_metrics.items() if v == metric_col][0]
            
            # Clean styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)
            ax.grid(False)
            
            ax.set_title(f'Performance vs {metric_name}', fontweight='bold', fontsize=14)
            ax.set_xlabel(metric_name, fontweight='bold', fontsize=12)
            ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
            ax.tick_params(labelsize=12)
            
            if idx == 0:
                ax.legend(fontsize=10)
        
        plt.suptitle('Model Performance vs Question Characteristics', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_plots(self, output_dir='plots_final'):
        """Generate all essential plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating essential TOM analysis plots...")
        
        print("1. Circular ability performance...")
        self.circular_ability_performance(
            save_path=f'{output_dir}/circular_ability_performance.png'
        )
        
        print("2. Comprehensive correlation matrix...")
        corr_df = self.comprehensive_correlation_matrix(
            save_path=f'{output_dir}/comprehensive_correlation_matrix.png'
        )
        
        print("3. Single correlation matrix...")
        single_corr = self.single_correlation_matrix(
            save_path=f'{output_dir}/single_correlation_matrix.png'
        )
        
        print("4. Scatter performance vs metrics...")
        self.scatter_performance_vs_metrics(
            save_path=f'{output_dir}/scatter_performance_vs_metrics.png'
        )
        
        # Save correlation data
        if corr_df is not None:
            corr_df.to_csv(f'{output_dir}/correlations.csv', index=False)
        
        print(f"\n✓ All plots saved to '{output_dir}' directory")
        print("Generated files:")
        print("- circular_ability_performance.png")
        print("- comprehensive_correlation_matrix.png")
        print("- single_correlation_matrix.png")
        print("- scatter_performance_vs_metrics.png")
        print("- correlations.csv")


def main():
    """Main function."""
    print("=" * 50)
    print("TOM ANALYSIS - ESSENTIAL PLOTS")
    print("=" * 50)
    
    analysis = TOMAnalysis()
    analysis.generate_all_plots()
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE!")
    print("=" * 50)


if __name__ == "__main__":
    main()
