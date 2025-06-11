#!/usr/bin/env python3
"""
Final Theory of Mind Analysis Visualization Class

This class creates corrected visualizations addressing the issues:
1. Model performance comparison based on accuracy across different complexity levels
2. Analysis metric correlation matrices per model
3. Single circular plot with all models overlayed for each ability group
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class ToMVisualizationFinal:
    """
    Final corrected visualization class for Theory of Mind analysis results.
    """
    
    def __init__(self, dataset_path='dataset_joined_corrected.csv'):
        """Initialize the visualization class with the corrected dataset."""
        print("Loading corrected dataset for final visualization...")
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
        
        # Analysis metrics grouped by type for proper normalization
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
        
        self.ability_column = '\nABILITY'
        
        # Group abilities into main categories
        self.ability_groups = self._group_abilities()
        
        # Normalize analysis metrics within their groups
        self._normalize_analysis_metrics()
        
        print(f"✓ Dataset loaded: {len(self.df)} samples")
        print(f"✓ Models: {len(self.model_columns)}")
        print(f"✓ Analysis groups: {len(self.analysis_groups)}")
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
    
    def _normalize_analysis_metrics(self):
        """Normalize analysis metrics within their respective groups."""
        print("Normalizing analysis metrics within groups...")
        
        scaler = MinMaxScaler()
        
        for group_name, metrics in self.analysis_groups.items():
            # Get the columns for this group
            group_cols = [col for col in metrics.values() if col in self.df.columns]
            
            if group_cols:
                # Normalize within this group
                normalized_data = scaler.fit_transform(self.df[group_cols])
                
                # Create normalized column names
                for i, col in enumerate(group_cols):
                    normalized_col_name = f"{col}_normalized"
                    self.df[normalized_col_name] = normalized_data[:, i]
                
                print(f"✓ Normalized {len(group_cols)} metrics in {group_name} group")
        
        # Update analysis groups to use normalized columns
        self.normalized_analysis_groups = {}
        for group_name, metrics in self.analysis_groups.items():
            self.normalized_analysis_groups[group_name] = {}
            for metric_name, col_name in metrics.items():
                normalized_col = f"{col_name}_normalized"
                if normalized_col in self.df.columns:
                    self.normalized_analysis_groups[group_name][metric_name] = normalized_col
                else:
                    self.normalized_analysis_groups[group_name][metric_name] = col_name
    
    def calculate_model_performance_by_complexity(self):
        """Calculate model performance across different complexity levels."""
        performance_data = []
        
        # Create complexity bins for each analysis type
        complexity_metrics = {
            'Question Complexity': 'Question_Complexity_Score',
            'Idea Density': 'Idea_Density',
            'RST Complexity': 'num_edus'
        }
        
        for complexity_type, complexity_col in complexity_metrics.items():
            if complexity_col not in self.df.columns:
                continue
                
            # Create quartiles for complexity
            quartiles = pd.qcut(self.df[complexity_col], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
            
            for quartile in ['Low', 'Medium-Low', 'Medium-High', 'High']:
                mask = quartiles == quartile
                subset = self.df[mask]
                
                if len(subset) == 0:
                    continue
                
                for i, model_col in enumerate(self.model_columns):
                    if model_col in subset.columns and '\nANSWER' in subset.columns:
                        accuracy = (subset[model_col] == subset['\nANSWER']).mean()
                        
                        performance_data.append({
                            'Model': self.model_names[i],
                            'Complexity_Type': complexity_type,
                            'Complexity_Level': quartile,
                            'Accuracy': accuracy,
                            'Sample_Count': len(subset)
                        })
        
        return pd.DataFrame(performance_data)
    
    def plot_model_performance_by_complexity(self, figsize=(16, 10), save_path=None):
        """Create plots showing model performance across complexity levels."""
        perf_df = self.calculate_model_performance_by_complexity()
        
        if perf_df.empty:
            print("No performance data available")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        complexity_types = perf_df['Complexity_Type'].unique()
        
        for i, complexity_type in enumerate(complexity_types):
            if i >= len(axes):
                break
                
            ax = axes[i]
            subset = perf_df[perf_df['Complexity_Type'] == complexity_type]
            
            # Pivot for easier plotting
            pivot_data = subset.pivot(index='Complexity_Level', columns='Model', values='Accuracy')
            
            # Plot grouped bar chart
            pivot_data.plot(kind='bar', ax=ax, alpha=0.8, width=0.8)
            ax.set_title(f'Model Performance by {complexity_type}', fontweight='bold')
            ax.set_xlabel('Complexity Level', fontweight='bold')
            ax.set_ylabel('Accuracy', fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        plt.suptitle('Model Performance Across Different Complexity Levels', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return perf_df
    
    def plot_analysis_correlation_per_model(self, figsize=(20, 12), save_path=None):
        """Create correlation matrices for analysis metrics, one per model."""
        # Get all normalized analysis metrics
        all_analysis_cols = []
        for group_name, metrics in self.normalized_analysis_groups.items():
            for metric_name, col_name in metrics.items():
                if col_name in self.df.columns:
                    all_analysis_cols.append(col_name)
        
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
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=ax, cbar_kws={'shrink': 0.8},
                       fmt='.2f', annot_kws={'size': 8})
            
            ax.set_title(f'{model_name}\nAnalysis Metrics Correlation', 
                        fontweight='bold', fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            
            # Rotate labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Analysis Metrics Correlation Matrices by Model', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return corr_matrix
    
    def plot_overlayed_circular_by_ability(self, figsize=(18, 12), save_path=None):
        """Create single circular plot with all models overlayed for each ability group."""
        fig, axes = plt.subplots(2, 3, figsize=figsize, subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        # Colors for different models
        model_colors = plt.cm.Set1(np.linspace(0, 1, len(self.model_names)))
        
        for i, (group_name, abilities) in enumerate(self.ability_groups.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Filter data for this ability group
            group_data = self.df[self.df[self.ability_column].isin(abilities)]
            
            if len(group_data) == 0:
                ax.set_title(f'{group_name}\n(No data)', fontsize=12)
                continue
            
            # Collect all normalized metrics
            all_metrics = []
            all_labels = []
            
            for analysis_group, metrics in self.normalized_analysis_groups.items():
                for metric_name, metric_col in metrics.items():
                    if metric_col in group_data.columns:
                        all_metrics.append(metric_col)
                        all_labels.append(f"{analysis_group[:3]}: {metric_name}")
            
            if not all_metrics:
                ax.set_title(f'{group_name}\n(No metrics)', fontsize=12)
                continue
            
            # Create angles for metrics
            angles = np.linspace(0, 2 * np.pi, len(all_metrics), endpoint=False)
            
            # Plot each model
            for model_idx, model_name in enumerate(self.model_names):
                model_col = self.model_columns[model_idx]
                
                # Calculate model performance for this ability group
                if model_col in group_data.columns and '\nANSWER' in group_data.columns:
                    model_accuracy = (group_data[model_col] == group_data['\nANSWER']).mean()
                else:
                    model_accuracy = 0
                
                # Get analysis metric scores for this group
                scores = []
                for metric_col in all_metrics:
                    mean_score = group_data[metric_col].mean()
                    scores.append(mean_score)
                
                # Close the circle
                scores_closed = np.concatenate((scores, [scores[0]]))
                angles_closed = np.concatenate((angles, [angles[0]]))
                
                # Plot with model-specific color
                ax.plot(angles_closed, scores_closed, 'o-', 
                       linewidth=2, markersize=4, 
                       color=model_colors[model_idx], 
                       label=f'{model_name} (Acc: {model_accuracy:.2f})',
                       alpha=0.8)
                
                # Fill with transparency
                ax.fill(angles_closed, scores_closed, 
                       color=model_colors[model_idx], alpha=0.1)
            
            ax.set_xticks(angles)
            ax.set_xticklabels(all_labels, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(f'{group_name}\n({len(group_data)} samples)', 
                        fontsize=12, fontweight='bold')
            ax.grid(True)
            
            # Add legend for first subplot only
            if i == 0:
                ax.legend(bbox_to_anchor=(1.3, 1.1), loc='upper left', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(self.ability_groups), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Analysis Metrics by Ability Group - All Models Overlayed', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_all_plots(self, output_dir='plots_final'):
        """Generate all corrected visualization plots and save them."""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating all corrected ToM analysis visualizations...")
        
        # 1. Model performance by complexity levels
        print("1. Creating model performance by complexity plots...")
        perf_df = self.plot_model_performance_by_complexity(
            save_path=f'{output_dir}/model_performance_by_complexity.png'
        )
        
        # Save performance data
        if not perf_df.empty:
            perf_df.to_csv(f'{output_dir}/model_performance_data.csv', index=False)
        
        # 2. Analysis correlation matrices per model
        print("2. Creating analysis correlation matrices per model...")
        corr_matrix = self.plot_analysis_correlation_per_model(
            save_path=f'{output_dir}/analysis_correlations_per_model.png'
        )
        
        # Save correlation matrix
        if corr_matrix is not None:
            corr_matrix.to_csv(f'{output_dir}/analysis_correlation_matrix.csv')
        
        # 3. Overlayed circular plots
        print("3. Creating overlayed circular plots...")
        self.plot_overlayed_circular_by_ability(
            save_path=f'{output_dir}/circular_overlayed_all_models.png'
        )
        
        print(f"\n✓ All corrected plots saved to '{output_dir}' directory")
        print("Generated files:")
        print("- model_performance_by_complexity.png")
        print("- model_performance_data.csv")
        print("- analysis_correlations_per_model.png")
        print("- analysis_correlation_matrix.csv")
        print("- circular_overlayed_all_models.png")
        
        return perf_df, corr_matrix


def main():
    """Main function to demonstrate the final corrected visualization class."""
    print("=" * 60)
    print("FINAL CORRECTED THEORY OF MIND ANALYSIS VISUALIZATION")
    print("=" * 60)
    
    # Initialize visualization class
    viz = ToMVisualizationFinal()
    
    # Generate all plots
    perf_df, corr_matrix = viz.generate_all_plots()
    
    print("\n" + "=" * 60)
    print("FINAL CORRECTED VISUALIZATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
