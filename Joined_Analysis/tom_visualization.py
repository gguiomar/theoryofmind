#!/usr/bin/env python3
"""
Theory of Mind Analysis Visualization Class

This class creates comprehensive visualizations for the ToM analysis results:
1. Overall score bar plots for each language model
2. Circular plots of analysis scores split by ability groups for each model
3. Correlation matrices between model scores and abilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

class ToMVisualization:
    """
    Comprehensive visualization class for Theory of Mind analysis results.
    """
    
    def __init__(self, dataset_path='dataset_joined.csv'):
        """Initialize the visualization class with the dataset."""
        print("Loading dataset for visualization...")
        self.df = pd.read_csv(dataset_path)
        
        # Define column mappings
        self.model_columns = [
            'Qwen_Qwen2.5_32B_Instruct',
            'allenai_OLMo_2_1124_13B_Instruct', 
            'mistralai_Mistral_7B_Instruct_v0.3',
            'microsoft_Phi_3_mini_4k_instruct',
            'internlm_internlm2_5_1_8b_chat'
        ]
        
        # Simplified model names for plotting
        self.model_names = [
            'Qwen 2.5 32B',
            'OLMo 13B',
            'Mistral 7B',
            'Phi 3 Mini',
            'InternLM 1.8B'
        ]
        
        # Analysis metrics
        self.analysis_metrics = {
            'Idea Density': 'Idea_Density',
            'Question Complexity': 'Question_Complexity_Score',
            'RST EDUs': 'num_edus',
            'RST Tree Depth': 'tree_depth',
            'RST Attribution': 'rel_attribution',
            'RST Causal': 'rel_causal',
            'RST Explanation': 'rel_explanation'
        }
        
        self.ability_column = '\nABILITY'
        
        # Group abilities into main categories
        self.ability_groups = self._group_abilities()
        
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
    
    def calculate_model_scores(self):
        """Calculate overall performance scores for each model."""
        model_scores = {}
        
        for i, model_col in enumerate(self.model_columns):
            model_name = self.model_names[i]
            
            # Convert model predictions to binary (assuming they're categorical)
            # We'll calculate accuracy by comparing with the correct answer
            if '\nANSWER' in self.df.columns:
                correct_answers = self.df['\nANSWER']
                model_predictions = self.df[model_col]
                
                # Calculate accuracy
                accuracy = (model_predictions == correct_answers).mean()
                model_scores[model_name] = accuracy
            else:
                # If no answer column, use mean of predictions (assuming numeric)
                model_scores[model_name] = self.df[model_col].mean()
        
        return model_scores
    
    def plot_overall_model_performance(self, figsize=(12, 6), save_path=None):
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
        
        plt.title('Overall Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Language Models', fontsize=12, fontweight='bold')
        plt.ylabel('Performance Score', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return model_scores
    
    def plot_circular_analysis_by_ability(self, model_idx=0, figsize=(15, 10), save_path=None):
        """Create circular plot of analysis scores split by ability groups for a specific model."""
        model_name = self.model_names[model_idx]
        model_col = self.model_columns[model_idx]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize, subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        # Calculate scores for each ability group
        group_names = list(self.ability_groups.keys())
        
        for i, (group_name, abilities) in enumerate(self.ability_groups.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Filter data for this ability group
            group_data = self.df[self.df[self.ability_column].isin(abilities)]
            
            if len(group_data) == 0:
                ax.set_title(f'{group_name}\n(No data)', fontsize=10)
                continue
            
            # Calculate mean scores for each analysis metric
            analysis_names = list(self.analysis_metrics.keys())
            scores = []
            
            for metric_name, metric_col in self.analysis_metrics.items():
                if metric_col in group_data.columns:
                    mean_score = group_data[metric_col].mean()
                    scores.append(mean_score)
                else:
                    scores.append(0)
            
            # Normalize scores to 0-1 range for better visualization
            if max(scores) > 0:
                scores = np.array(scores) / max(scores)
            
            # Create circular plot
            angles = np.linspace(0, 2 * np.pi, len(analysis_names), endpoint=False)
            scores = np.concatenate((scores, [scores[0]]))  # Close the circle
            angles = np.concatenate((angles, [angles[0]]))
            
            ax.plot(angles, scores, 'o-', linewidth=2, markersize=6)
            ax.fill(angles, scores, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(analysis_names, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(f'{group_name}\n({len(group_data)} samples)', fontsize=10, fontweight='bold')
            ax.grid(True)
        
        # Hide unused subplots
        for i in range(len(self.ability_groups), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Analysis Scores by Ability Group - {model_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_all_models_circular(self, figsize=(20, 25), save_path=None):
        """Create circular plots for all models."""
        fig = plt.figure(figsize=figsize)
        
        for model_idx in range(len(self.model_names)):
            print(f"Creating circular plot for {self.model_names[model_idx]}...")
            
            # Create subplot for this model
            for group_idx, (group_name, abilities) in enumerate(self.ability_groups.items()):
                subplot_idx = model_idx * len(self.ability_groups) + group_idx + 1
                ax = fig.add_subplot(len(self.model_names), len(self.ability_groups), 
                                   subplot_idx, projection='polar')
                
                # Filter data for this ability group
                group_data = self.df[self.df[self.ability_column].isin(abilities)]
                
                if len(group_data) == 0:
                    ax.set_title(f'{group_name}\n(No data)', fontsize=8)
                    continue
                
                # Calculate mean scores for each analysis metric
                analysis_names = list(self.analysis_metrics.keys())
                scores = []
                
                for metric_name, metric_col in self.analysis_metrics.items():
                    if metric_col in group_data.columns:
                        mean_score = group_data[metric_col].mean()
                        scores.append(mean_score)
                    else:
                        scores.append(0)
                
                # Normalize scores
                if max(scores) > 0:
                    scores = np.array(scores) / max(scores)
                
                # Create circular plot
                angles = np.linspace(0, 2 * np.pi, len(analysis_names), endpoint=False)
                scores = np.concatenate((scores, [scores[0]]))
                angles = np.concatenate((angles, [angles[0]]))
                
                ax.plot(angles, scores, 'o-', linewidth=1.5, markersize=4)
                ax.fill(angles, scores, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(analysis_names, fontsize=6)
                ax.set_ylim(0, 1)
                
                # Add model name to first subplot of each row
                if group_idx == 0:
                    ax.set_ylabel(self.model_names[model_idx], fontsize=10, fontweight='bold')
                
                # Add group name to top row
                if model_idx == 0:
                    ax.set_title(f'{group_name}', fontsize=8, fontweight='bold')
                
                ax.grid(True)
        
        plt.suptitle('Analysis Scores by Ability Group - All Models', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrices(self, figsize=(15, 12), save_path=None):
        """Create correlation matrices between model scores and abilities."""
        # Prepare data for correlation analysis
        correlation_data = []
        
        # For each ability group, calculate mean performance for each model
        for group_name, abilities in self.ability_groups.items():
            group_data = self.df[self.df[self.ability_column].isin(abilities)]
            
            if len(group_data) == 0:
                continue
            
            row_data = {'Ability_Group': group_name}
            
            # Add model performance scores
            for i, model_col in enumerate(self.model_columns):
                model_name = self.model_names[i]
                if '\nANSWER' in self.df.columns:
                    # Calculate accuracy for this ability group
                    correct = group_data['\nANSWER']
                    predictions = group_data[model_col]
                    accuracy = (predictions == correct).mean()
                    row_data[model_name] = accuracy
                else:
                    row_data[model_name] = group_data[model_col].mean()
            
            # Add analysis metric scores
            for metric_name, metric_col in self.analysis_metrics.items():
                if metric_col in group_data.columns:
                    row_data[metric_name] = group_data[metric_col].mean()
                else:
                    row_data[metric_name] = 0
            
            correlation_data.append(row_data)
        
        # Convert to DataFrame
        corr_df = pd.DataFrame(correlation_data)
        corr_df = corr_df.set_index('Ability_Group')
        
        # Create correlation matrix
        correlation_matrix = corr_df.corr()
        
        # Split into model-model, model-analysis, and analysis-analysis correlations
        model_cols = self.model_names
        analysis_cols = list(self.analysis_metrics.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Model-Model correlations
        model_corr = correlation_matrix.loc[model_cols, model_cols]
        sns.heatmap(model_corr, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=axes[0,0], cbar_kws={'shrink': 0.8})
        axes[0,0].set_title('Model-Model Correlations', fontweight='bold')
        
        # 2. Analysis-Analysis correlations
        analysis_corr = correlation_matrix.loc[analysis_cols, analysis_cols]
        sns.heatmap(analysis_corr, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=axes[0,1], cbar_kws={'shrink': 0.8})
        axes[0,1].set_title('Analysis Metric Correlations', fontweight='bold')
        
        # 3. Model-Analysis correlations
        model_analysis_corr = correlation_matrix.loc[model_cols, analysis_cols]
        sns.heatmap(model_analysis_corr, annot=True, cmap='RdBu_r', center=0,
                   ax=axes[1,0], cbar_kws={'shrink': 0.8})
        axes[1,0].set_title('Model-Analysis Correlations', fontweight='bold')
        
        # 4. Overall correlation matrix (subset)
        # Select most interesting correlations
        interesting_cols = model_cols + ['Idea Density', 'Question Complexity', 'RST EDUs']
        if all(col in correlation_matrix.columns for col in interesting_cols):
            subset_corr = correlation_matrix.loc[interesting_cols, interesting_cols]
            sns.heatmap(subset_corr, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=axes[1,1], cbar_kws={'shrink': 0.8})
            axes[1,1].set_title('Key Correlations Overview', fontweight='bold')
        else:
            axes[1,1].set_visible(False)
        
        plt.suptitle('Correlation Analysis: Models vs Analysis Metrics', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return correlation_matrix
    
    def generate_all_plots(self, output_dir='plots'):
        """Generate all visualization plots and save them."""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating all ToM analysis visualizations...")
        
        # 1. Overall model performance
        print("1. Creating overall model performance plot...")
        self.plot_overall_model_performance(
            save_path=f'{output_dir}/model_performance_overall.png'
        )
        
        # 2. Circular plots for each model
        print("2. Creating circular plots for individual models...")
        for i in range(len(self.model_names)):
            self.plot_circular_analysis_by_ability(
                model_idx=i,
                save_path=f'{output_dir}/circular_analysis_{self.model_names[i].replace(" ", "_")}.png'
            )
        
        # 3. All models circular plot
        print("3. Creating comprehensive circular plot for all models...")
        self.plot_all_models_circular(
            save_path=f'{output_dir}/circular_analysis_all_models.png'
        )
        
        # 4. Correlation matrices
        print("4. Creating correlation matrices...")
        correlation_matrix = self.plot_correlation_matrices(
            save_path=f'{output_dir}/correlation_matrices.png'
        )
        
        # Save correlation matrix as CSV
        correlation_matrix.to_csv(f'{output_dir}/correlation_matrix.csv')
        
        print(f"\n✓ All plots saved to '{output_dir}' directory")
        print("Generated files:")
        print("- model_performance_overall.png")
        print("- circular_analysis_[model_name].png (5 files)")
        print("- circular_analysis_all_models.png")
        print("- correlation_matrices.png")
        print("- correlation_matrix.csv")
        
        return correlation_matrix


def main():
    """Main function to demonstrate the visualization class."""
    print("=" * 60)
    print("THEORY OF MIND ANALYSIS VISUALIZATION")
    print("=" * 60)
    
    # Initialize visualization class
    viz = ToMVisualization()
    
    # Generate all plots
    correlation_matrix = viz.generate_all_plots()
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
