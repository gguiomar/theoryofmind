#!/usr/bin/env python3
"""
Generate clean performance matrix heatmap as PDF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

def create_performance_matrix():
    """Create the performance matrix from idea_density_3.py"""
    
    # Load and prepare data
    df = pd.read_csv('./dataset.csv')
    df.columns = df.columns.str.strip()

    # Fix the ANSWER column name issue
    answer_col = '\nANSWER' if '\nANSWER' in df.columns else 'ANSWER'
    
    df_clean = df[df['ABILITY'].notna() & df['Idea_Density'].notna()].copy()
    df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')

    # Human performance data
    human_performance = {
        'Emotion': 86.4,
        'Desire': 90.4,
        'Intention': 82.2,
        'Knowledge': 89.3,
        'Belief': 89.0,
        'NLC': 86.1
    }

    # Model list and display names
    models = [
        'meta_llama_Llama_3.1_70B_Instruct',
        'Qwen_Qwen2.5_32B_Instruct',
        'allenai_OLMo_2_1124_13B_Instruct',
        'mistralai_Mistral_7B_Instruct_v0.3',
        'microsoft_Phi_3_mini_4k_instruct',
        'internlm_internlm2_5_1_8b_chat'
    ]

    model_display_names = {
        'meta_llama_Llama_3.1_70B_Instruct': 'Llama 3.1 70B',
        'Qwen_Qwen2.5_32B_Instruct': 'Qwen 2.5 32B',
        'allenai_OLMo_2_1124_13B_Instruct': 'OLMo 13B',
        'mistralai_Mistral_7B_Instruct_v0.3': 'Mistral 7B',
        'microsoft_Phi_3_mini_4k_instruct': 'Phi-3 Mini',
        'internlm_internlm2_5_1_8b_chat': 'InternLM 1.8B'
    }

    # Calculate model performance
    def calculate_accuracy(df, model_col, category):
        subset = df[df['Main_Category'] == category]
        if len(subset) == 0:
            return 0
        correct = (subset[model_col] == subset[answer_col]).sum()
        return (correct / len(subset) * 100)

    # Define categories
    categories = list(human_performance.keys())

    # Create performance matrix
    performance_matrix = pd.DataFrame(index=categories)
    performance_matrix['Human'] = [human_performance[cat] for cat in categories]

    # Calculate model performances
    for model in models:
        display_name = model_display_names[model]
        performance_matrix[display_name] = [
            calculate_accuracy(df_clean, model, cat) for cat in categories
        ]

    # Sort by human performance (descending)
    performance_matrix = performance_matrix.sort_values(by='Human', ascending=False)
    
    return performance_matrix

def create_heatmap_pdf(performance_matrix, filename='performance_matrix.pdf'):
    """Create a clean heatmap and save as PDF"""
    
    # Set up the plot with high DPI for PDF
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    sns.heatmap(performance_matrix.T, 
                cmap='rocket_r',  # Seaborn rocket colormap (reversed)
                cbar_kws={'label': 'Performance (%)', 'shrink': 0.8},
                square=True,
                linewidths=2.0,
                linecolor='white',
                ax=ax,
                annot=True,  # Show values
                fmt='.1f',   # Format to 1 decimal place
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})

    # Customize the plot
    ax.set_title('Performance by Theory of Mind Category', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xlabel('ToM Categories', fontweight='bold', fontsize=12)
    ax.set_ylabel('Models/Subjects', fontweight='bold', fontsize=12)
    
    # Rotate labels for better readability
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', rotation=0, labelsize=11)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save as PDF
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
    
    plt.close()
    print(f"Performance matrix heatmap saved as: {filename}")

def main():
    """Main function"""
    print("Creating performance matrix...")
    
    # Create performance matrix
    performance_matrix = create_performance_matrix()
    
    print("Performance Matrix:")
    print(performance_matrix.round(1))
    
    # Create PDF
    create_heatmap_pdf(performance_matrix, 'ToM_Performance_Matrix.pdf')
    
    print("\nPDF created successfully!")

if __name__ == "__main__":
    main()
