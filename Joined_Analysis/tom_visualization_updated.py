#!/usr/bin/env python3
"""
Updated Theory of Mind Analysis Visualization Class

This class creates comprehensive visualizations for the ToM analysis results with:
1. All 6 language models including Meta Llama 70B
2. Proper normalization within analysis groups
3. Overall score bar plots for each language model
4. Circular plots of analysis scores split by ability groups for each model
5. Correlation matrices between model scores and abilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class ToMVisualizationUpdated:
    """
    Updated comprehensive visualization class for Theory of Mind analysis results.
    """
    
    def __init__(self, dataset_path='dataset_joined_corrected.csv'):
        """Initialize the visualization class with the corrected dataset."""
        print("Loading corrected dataset for visualization...")
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
    
    def calculate_model_scores(self):
        """Calculate overall performance scores for each model."""
        model_scores = {}
        
        for i, model_col in enumerate(self.model_columns):
            model_name = self.model_names[i]
            
            if '\nANSWER' in self.df.columns and model_col in self.df.columns:
                correct_answers = self.df['\nANSWER']
                model_predictions = self.df[model_col]
                
                # Calculate accuracy
                accuracy = (model_predictions == correct_answers).mean()
                model_scores[model_name] = accuracy
            else:
                model_scores[model_name] = 0
        
        return model_scores
    
    def plot_overall_model_performance(self, figsize=(14, 8), save_path=None):
        """Create bar plot of overall model performance."""
        model_scores = self.calculate_model_scores()
        
        plt.figure(figsize=figsize)
        
        models = list(model_scores.keys())
        scores = list(model_scores.values())
        
        # Create color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        bars = plt.bar(models, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Overall Model Performance Comparison (All 6 Models)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Language Models', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy Score', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, max(scores) * 1.1)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return model_scores
    
    def plot_circular_analysis_by_ability(self, model_idx=0, figsize=(16, 12), save_path=None):
        """Create circular plot of normalized analysis scores split by ability groups."""
        model_name = self.model_names[model_idx]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize, subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        # Colors for different analysis groups
        group_colors = {'Idea Density': 'blue', 'Question Complexity': 'red', 'RST Analysis': 'green'}
        
        for i, (group_name, abilities) in enumerate(self.ability_groups.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Filter data for this ability group
            group_data = self.df[self.df[self.ability_column].isin(abilities)]
            
            if len(group_data) == 0:
                ax.set_title(f'{group_name}\n(No data)', fontsize=10)
                continue
            
            # Collect all normalized metrics
            all_metrics = []
            all_labels = []
            all_colors = []
            
            for analysis_group, metrics in self.normalized_analysis_groups.items():
                for metric_name, metric_col in metrics.items():
                    if metric_col in group_data.columns:
                        mean_score = group_data[metric_col].mean()
                        all_metrics.append(mean_score)
                        all_labels.append(f"{analysis_group[:3]}: {metric_name}")
                        all_colors.append(group_colors.get(analysis_group, 'gray'))
            
            if not all_metrics:
                ax.set_title(f'{group_name}\n(No metrics)', fontsize=10)
                continue
            
            # Create circular plot
            angles = np.linspace(0, 2 * np.pi, len(all_metrics), endpoint=False)
            scores = np.array(all_metrics)
            
            # Close the circle
            scores = np.concatenate((scores, [scores[0]]))
            angles = np.concatenate((angles, [angles[0]]))
            
            ax.plot(angles, scores, 'o-', linewidth=2, markersize=6)
            ax.fill(angles, scores, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(all_labels, fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_title(f'{group_name}\n({len(group_data)} samples)', 
                        fontsize=10, fontweight='bold')
            ax.grid(True)
        
        # Hide unused subplots
        for i in range(len(self.ability_groups), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Normalized Analysis Scores by Ability Group - {model_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_analysis_group_comparison(self, figsize=(15, 10), save_path=None):
        """Create comparison plots for each analysis group across all models."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot each analysis group
        for group_name, metrics in self.normalized_analysis_groups.items():
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            # Calculate mean scores for each model and metric
            model_scores = {model: [] for model in self.model_names}
            metric_names = list(metrics.keys())
            
            for model_idx, model_name in enumerate(self.model_names):
                for metric_name, metric_col in metrics.items():
                    if metric_col in self.df.columns:
                        mean_score = self.df[metric_col].mean()
                        model_scores[model_name].append(mean_score)
                    else:
                        model_scores[model_name].append(0)
            
            # Create grouped bar plot
            x = np.arange(len(metric_names))
            width = 0.12
            
            for i, (model_name, scores) in enumerate(model_scores.items()):
                ax.bar(x + i * width, scores, width, label=model_name, alpha=0.8)
            
            ax.set_xlabel('Metrics', fontweight='bold')
            ax.set_ylabel('Normalized Score', fontweight='bold')
            ax.set_title(f'{group_name} - Model Comparison', fontweight='bold')
            ax.set_xticks(x + width * 2.5)
            ax.set_xticklabels(metric_names, rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            
            plot_idx += 1
        
        # Hide unused subplot
        if plot_idx < len(axes):
            axes[-1].set_visible(False)
        
        plt.suptitle('Analysis Group Comparisons Across All Models', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrices(self, figsize=(16, 12), save_path=None):
        """Create correlation matrices between model scores and normalized analysis metrics."""
        # Prepare data for correlation analysis
        correlation_data = []
        
        # For each ability group, calculate mean performance
        for group_name, abilities in self.ability_groups.items():
            group_data = self.df[self.df[self.ability_column].isin(abilities)]
            
            if len(group_data) == 0:
                continue
            
            row_data = {'Ability_Group': group_name}
            
            # Add model performance scores
            for i, model_col in enumerate(self.model_columns):
                model_name = self.model_names[i]
                if '\nANSWER' in self.df.columns and model_col in group_data.columns:
                    correct = group_data['\nANSWER']
                    predictions = group_data[model_col]
                    accuracy = (predictions == correct).mean()
                    row_data[model_name] = accuracy
                else:
                    row_data[model_name] = 0
            
            # Add normalized analysis metric scores
            for analysis_group, metrics in self.normalized_analysis_groups.items():
                for metric_name, metric_col in metrics.items():
                    if metric_col in group_data.columns:
                        full_name = f"{analysis_group}: {metric_name}"
                        row_data[full_name] = group_data[metric_col].mean()
            
            correlation_data.append(row_data)
        
        # Convert to DataFrame
        corr_df = pd.DataFrame(correlation_data)
        corr_df = corr_df.set_index('Ability_Group')
        
        # Create correlation matrix
        correlation_matrix = corr_df.corr()
        
        # Split into different correlation types
        model_cols = self.model_names
        analysis_cols = [col for col in correlation_matrix.columns if ':' in col]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Model-Model correlations
        if len(model_cols) > 1:
            model_corr = correlation_matrix.loc[model_cols, model_cols]
            sns.heatmap(model_corr, annot=True, cmap='RdBu_r', center=0, 
                       square=True, ax=axes[0,0], cbar_kws={'shrink': 0.8})
            axes[0,0].set_title('Model-Model Correlations', fontweight='bold')
        
        # 2. Analysis-Analysis correlations (subset)
        if len(analysis_cols) > 1:
            # Select a subset for readability
            analysis_subset = analysis_cols[:10] if len(analysis_cols) > 10 else analysis_cols
            analysis_corr = correlation_matrix.loc[analysis_subset, analysis_subset]
            sns.heatmap(analysis_corr, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=axes[0,1], cbar_kws={'shrink': 0.8})
            axes[0,1].set_title('Analysis Metric Correlations', fontweight='bold')
            axes[0,1].tick_params(axis='both', labelsize=8)
        
        # 3. Model-Analysis correlations (subset)
        if len(analysis_cols) > 0:
            analysis_subset = analysis_cols[:8] if len(analysis_cols) > 8 else analysis_cols
            model_analysis_corr = correlation_matrix.loc[model_cols, analysis_subset]
            sns.heatmap(model_analysis_corr, annot=True, cmap='RdBu_r', center=0,
                       ax=axes[1,0], cbar_kws={'shrink': 0.8})
            axes[1,0].set_title('Model-Analysis Correlations', fontweight='bold')
            axes[1,0].tick_params(axis='x', labelsize=8, rotation=45)
        
        # 4. Overall heatmap (key metrics only)
        key_analysis = [col for col in analysis_cols if any(key in col for key in ['Overall', 'Idea Density', 'EDUs'])]
        if key_analysis:
            key_cols = model_cols + key_analysis[:5]
            subset_corr = correlation_matrix.loc[key_cols, key_cols]
            sns.heatmap(subset_corr, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=axes[1,1], cbar_kws={'shrink': 0.8})
            axes[1,1].set_title('Key Correlations Overview', fontweight='bold')
            axes[1,1].tick_params(axis='both', labelsize=8)
        else:
            axes[1,1].set_visible(False)
        
        plt.suptitle('Correlation Analysis: Models vs Normalized Analysis Metrics', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return correlation_matrix
    
    def generate_all_plots(self, output_dir='plots_updated'):
        """Generate all visualization plots and save them."""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating all updated ToM analysis visualizations...")
        
        # 1. Overall model performance
        print("1. Creating overall model performance plot...")
        self.plot_overall_model_performance(
            save_path=f'{output_dir}/model_performance_all_6_models.png'
        )
        
        # 2. Analysis group comparison
        print("2. Creating analysis group comparison...")
        self.plot_analysis_group_comparison(
            save_path=f'{output_dir}/analysis_group_comparison.png'
        )
        
        # 3. Circular plots for each model
        print("3. Creating circular plots for individual models...")
        for i in range(len(self.model_names)):
            model_name_clean = self.model_names[i].replace(" ", "_").replace(".", "")
            self.plot_circular_analysis_by_ability(
                model_idx=i,
                save_path=f'{output_dir}/circular_normalized_{model_name_clean}.png'
            )
        
        # 4. Correlation matrices
        print("4. Creating correlation matrices...")
        correlation_matrix = self.plot_correlation_matrices(
            save_path=f'{output_dir}/correlation_matrices_normalized.png'
        )
        
        # Save correlation matrix as CSV
        correlation_matrix.to_csv(f'{output_dir}/correlation_matrix_normalized.csv')
        
        print(f"\n✓ All plots saved to '{output_dir}' directory")
        print("Generated files:")
        print("- model_performance_all_6_models.png")
        print("- analysis_group_comparison.png")
        print("- circular_normalized_[model_name].png (6 files)")
        print("- correlation_matrices_normalized.png")
        print("- correlation_matrix_normalized.csv")
        
        return correlation_matrix


def main():
    """Main function to demonstrate the updated visualization class."""
    print("=" * 60)
    print("UPDATED THEORY OF MIND ANALYSIS VISUALIZATION")
    print("=" * 60)
    
    # Initialize visualization class
    viz = ToMVisualizationUpdated()
    
    # Generate all plots
    correlation_matrix = viz.generate_all_plots()
    
    print("\n" + "=" * 60)
    print("UPDATED VISUALIZATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
