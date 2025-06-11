#!/usr/bin/env python3
"""
Enhanced Theory of Mind Analysis Visualization Class

Enhanced features:
1. All correlations shown with crest color scheme
2. Circular bar plots for performance across complexity levels
3. Correlation matrices per model for all submeasures
4. Clean styling with no grids/borders and larger fonts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style and crest color palette
sns.set_style("white")
sns.set_palette("crest")

class ToMVisualizationEnhanced:
    """
    Enhanced visualization class with improved styling and comprehensive analysis.
    """
    
    def __init__(self, dataset_path='dataset_joined_corrected.csv'):
        """Initialize the visualization class with the corrected dataset."""
        print("Loading dataset for enhanced visualization...")
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
        
        # Analysis metrics grouped by category with all submeasures
        self.analysis_groups = {
            'Idea Density': {
                'Idea Density': 'Idea_Density'
            },
            'Question Complexity': {
                'Overall Complexity': 'Question_Complexity_Score',
                'Syntactic': 'Q_Syntactic_Complexity',
                'Semantic': 'Q_Semantic_Complexity',
                'ToM': 'Q_ToM_Complexity',
                'Reasoning': 'Q_Reasoning_Complexity'
            },
            'RST Analysis': {
                'EDUs': 'num_edus',
                'Tree Depth': 'tree_depth',
                'Attribution': 'rel_attribution',
                'Causal': 'rel_causal',
                'Explanation': 'rel_explanation'
            }
        }
        
        # Flatten all metrics for comprehensive analysis
        self.all_metrics = {}
        for group_name, metrics in self.analysis_groups.items():
            for metric_name, metric_col in metrics.items():
                full_name = f"{group_name}: {metric_name}"
                self.all_metrics[full_name] = metric_col
        
        self.ability_column = '\nABILITY'
        
        # Group abilities into main categories
        self.ability_groups = self._group_abilities()
        
        # Calculate model performance for each sample
        self._calculate_model_performance()
        
        print(f"✓ Dataset loaded: {len(self.df)} samples")
        print(f"✓ Models: {len(self.model_columns)}")
        print(f"✓ Analysis groups: {len(self.analysis_groups)}")
        print(f"✓ All metrics: {len(self.all_metrics)}")
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
    
    def _clean_axes(self, ax):
        """Apply clean styling to axes - remove grids and borders."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.grid(False)
        ax.tick_params(labelsize=12)
    
    def plot_circular_performance_by_complexity(self, figsize=(20, 5), save_path=None):
        """Create circular bar plots for performance across complexity levels per metric."""
        # Select key metrics for circular plots
        key_metrics = {
            'Question Complexity': 'Question_Complexity_Score',
            'Idea Density': 'Idea_Density',
            'RST EDUs': 'num_edus',
            'ToM Complexity': 'Q_ToM_Complexity'
        }
        
        fig, axes = plt.subplots(1, 4, figsize=figsize, subplot_kw=dict(projection='polar'))
        
        colors = sns.color_palette("crest", len(self.model_names))
        
        for idx, (metric_name, metric_col) in enumerate(key_metrics.items()):
            if idx >= len(axes) or metric_col not in self.df.columns:
                continue
                
            ax = axes[idx]
            
            # Create quartile bins
            try:
                bins = pd.qcut(self.df[metric_col], q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'], duplicates='drop')
            except ValueError:
                bins = pd.cut(self.df[metric_col], bins=4, labels=['Low', 'Med-Low', 'Med-High', 'High'])
            
            # Calculate performance for each model in each bin
            bin_labels = ['Low', 'Med-Low', 'Med-High', 'High']
            angles = np.linspace(0, 2 * np.pi, len(bin_labels), endpoint=False)
            
            # Plot each model
            for model_idx, model_name in enumerate(self.model_names):
                perf_col = f'{model_name}_Performance'
                if perf_col not in self.df.columns:
                    continue
                
                performances = []
                for bin_label in bin_labels:
                    mask = bins == bin_label
                    subset = self.df[mask]
                    if len(subset) > 0:
                        accuracy = subset[perf_col].mean()
                        performances.append(accuracy)
                    else:
                        performances.append(0)
                
                # Create circular bar plot
                bars = ax.bar(angles, performances, width=0.4, alpha=0.8, 
                             color=colors[model_idx], label=model_name)
            
            # Customize the plot
            ax.set_xticks(angles)
            ax.set_xticklabels(bin_labels, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            
            # Add legend only to first subplot
            if idx == 0:
                ax.legend(bbox_to_anchor=(1.3, 1.1), loc='upper left', fontsize=10)
        
        plt.suptitle('Model Performance Across Complexity Levels (Circular Bar Plots)', 
                    fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_comprehensive_correlation_matrix(self, figsize=(16, 12), save_path=None):
        """Show all correlations between model performance and analysis metrics using crest colormap."""
        # Calculate correlations for each model
        correlation_data = []
        
        for model_name in self.model_names:
            perf_col = f'{model_name}_Performance'
            
            if perf_col not in self.df.columns:
                continue
            
            for metric_name, metric_col in self.all_metrics.items():
                if metric_col in self.df.columns:
                    # Calculate correlation
                    corr, p_value = pearsonr(self.df[perf_col], self.df[metric_col])
                    
                    correlation_data.append({
                        'Model': model_name,
                        'Metric': metric_name,
                        'Correlation': corr,
                        'P_Value': p_value
                    })
        
        if not correlation_data:
            print("No correlation data available")
            return
        
        corr_df = pd.DataFrame(correlation_data)
        
        # Create pivot table for heatmap
        pivot_corr = corr_df.pivot(index='Model', columns='Metric', values='Correlation')
        
        # Create heatmap with all correlations shown
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(pivot_corr, annot=True, cmap='crest', center=0,
                   square=True, cbar_kws={'shrink': 0.8}, fmt='.3f',
                   linewidths=0.5, ax=ax, annot_kws={'size': 10})
        
        # Clean styling
        self._clean_axes(ax)
        
        ax.set_title('Model Performance Correlation with All Question Characteristics', 
                    fontweight='bold', fontsize=16, pad=20)
        ax.set_xlabel('Analysis Metrics', fontweight='bold', fontsize=14)
        ax.set_ylabel('Models', fontweight='bold', fontsize=14)
        
        # Rotate labels
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return corr_df
    
    def plot_correlation_matrices_per_model(self, figsize=(20, 15), save_path=None):
        """Create correlation matrices for all submeasures, one per model."""
        # Get all analysis metric columns
        all_analysis_cols = [col for col in self.all_metrics.values() if col in self.df.columns]
        
        if len(all_analysis_cols) < 2:
            print("Not enough analysis metrics for correlation")
            return
        
        # Create subplots for each model
        n_models = len(self.model_names)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for i, model_name in enumerate(self.model_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Calculate correlation matrix for analysis metrics
            corr_matrix = self.df[all_analysis_cols].corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='crest', center=0,
                       square=True, ax=ax, cbar_kws={'shrink': 0.8},
                       fmt='.2f', annot_kws={'size': 8}, linewidths=0.5)
            
            # Clean styling
            self._clean_axes(ax)
            
            ax.set_title(f'{model_name}\nSubmeasure Correlations', 
                        fontweight='bold', fontsize=12)
            
            # Create readable labels
            readable_labels = []
            for col in all_analysis_cols:
                # Find the readable name
                for readable_name, col_name in self.all_metrics.items():
                    if col_name == col:
                        # Shorten the label
                        short_name = readable_name.split(': ')[-1]
                        readable_labels.append(short_name)
                        break
                else:
                    readable_labels.append(col)
            
            ax.set_xticklabels(readable_labels, rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(readable_labels, rotation=0, fontsize=10)
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Analysis Submeasure Correlation Matrices by Model', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return corr_matrix
    
    def plot_ability_group_performance_enhanced(self, figsize=(16, 12), save_path=None):
        """Enhanced ability group performance with clean styling."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        colors = sns.color_palette("crest", len(self.model_names))
        
        for idx, (group_name, abilities) in enumerate(self.ability_groups.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Filter data for this ability group
            group_data = self.df[self.df[self.ability_column].isin(abilities)]
            
            if len(group_data) == 0:
                ax.set_title(f'{group_name}\n(No data)', fontsize=14, fontweight='bold')
                self._clean_axes(ax)
                continue
            
            # Calculate model performance for this group
            performance_data = []
            
            for model_idx, model_name in enumerate(self.model_names):
                perf_col = f'{model_name}_Performance'
                if perf_col in group_data.columns:
                    accuracy = group_data[perf_col].mean()
                    performance_data.append({
                        'Model': model_name,
                        'Accuracy': accuracy
                    })
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                
                # Create bar plot with custom colors
                bars = ax.bar(range(len(perf_df)), perf_df['Accuracy'], 
                             color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                
                ax.set_title(f'{group_name}\n({len(group_data)} samples)', 
                           fontweight='bold', fontsize=14)
                ax.set_xlabel('')
                ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
                ax.set_xticks(range(len(perf_df)))
                ax.set_xticklabels([name.split()[0] for name in perf_df['Model']], 
                                  rotation=45, ha='right', fontsize=11)
                
                # Clean styling
                self._clean_axes(ax)
                
                # Add mean complexity as text
                if 'Question_Complexity_Score' in group_data.columns:
                    mean_complexity = group_data['Question_Complexity_Score'].mean()
                    ax.text(0.02, 0.98, f'Avg Q Complexity: {mean_complexity:.2f}', 
                           transform=ax.transAxes, fontsize=10, verticalalignment='top',
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
    
    def plot_circular_ability_performance_enhanced(self, figsize=(12, 12), save_path=None):
        """Enhanced circular plot with model performance across ToM ability groups."""
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
        
        # Colors for different models using crest palette
        colors = sns.color_palette("crest", len(self.model_names))
        
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
        ax.set_xticklabels(ability_names, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add radial labels
        ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_rlabel_position(0)
        ax.tick_params(labelsize=12)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        
        # Add title
        plt.title('Model Performance Across Theory of Mind Ability Groups', 
                 fontsize=16, fontweight='bold', pad=30)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return ability_performance
    
    def generate_all_plots(self, output_dir='plots_enhanced'):
        """Generate all enhanced visualization plots and save them."""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating enhanced ToM analysis visualizations...")
        
        # 1. Circular performance by complexity
        print("1. Creating circular performance by complexity plots...")
        self.plot_circular_performance_by_complexity(
            save_path=f'{output_dir}/circular_performance_by_complexity.png'
        )
        
        # 2. Comprehensive correlation matrix
        print("2. Creating comprehensive correlation matrix...")
        corr_df = self.plot_comprehensive_correlation_matrix(
            save_path=f'{output_dir}/comprehensive_correlation_matrix.png'
        )
        
        # Save correlation data
        if corr_df is not None:
            corr_df.to_csv(f'{output_dir}/comprehensive_correlations.csv', index=False)
        
        # 3. Correlation matrices per model
        print("3. Creating correlation matrices per model...")
        model_corr = self.plot_correlation_matrices_per_model(
            save_path=f'{output_dir}/correlation_matrices_per_model.png'
        )
        
        # 4. Enhanced ability group performance
        print("4. Creating enhanced ability group performance...")
        self.plot_ability_group_performance_enhanced(
            save_path=f'{output_dir}/ability_group_performance_enhanced.png'
        )
        
        # 5. Enhanced circular ability performance
        print("5. Creating enhanced circular ability performance...")
        ability_perf = self.plot_circular_ability_performance_enhanced(
            save_path=f'{output_dir}/circular_ability_performance_enhanced.png'
        )
        
        print(f"\n✓ All enhanced plots saved to '{output_dir}' directory")
        print("Generated files:")
        print("- circular_performance_by_complexity.png")
        print("- comprehensive_correlation_matrix.png")
        print("- comprehensive_correlations.csv")
        print("- correlation_matrices_per_model.png")
        print("- ability_group_performance_enhanced.png")
        print("- circular_ability_performance_enhanced.png")
        
        return corr_df, model_corr


def main():
    """Main function to demonstrate the enhanced visualization class."""
    print("=" * 60)
    print("ENHANCED THEORY OF MIND ANALYSIS VISUALIZATION")
    print("=" * 60)
    
    # Initialize visualization class
    viz = ToMVisualizationEnhanced()
    
    # Generate all plots
    corr_df, model_corr = viz.generate_all_plots()
    
    print("\n" + "=" * 60)
    print("ENHANCED VISUALIZATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
