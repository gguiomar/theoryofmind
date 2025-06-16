#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style and remove grid
sns.set_style("white")
plt.rcParams['axes.grid'] = False

# Read and prepare data
df = pd.read_csv('./main_with_perspective_annotations.csv')
df.columns = df.columns.str.strip()
df_clean = df[df['ABILITY'].notna() & df['Idea_Density'].notna()].copy()
df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()

# Calculate average idea density per category
idea_density_by_category = df_clean.groupby('Main_Category')['Idea_Density'].mean()

# Calculate average MSVD overall per category
msvd_overall_by_category = df_clean.groupby('Main_Category')['Overall_MSVD'].mean()

# Human performance data (from the original paper)
human_performance = {
    'Emotion': 86.4,
    'Desire': 90.4,
    'Intention': 82.2,
    'Knowledge': 89.3,
    'Belief': 89.0,
    'Non-Literal Communication': 86.1
}

# Model list
models = [
    'meta_llama_Llama_3.1_70B_Instruct',
    'Qwen_Qwen2.5_32B_Instruct',
    'allenai_OLMo_2_1124_13B_Instruct',
    'mistralai_Mistral_7B_Instruct_v0.3',
    'microsoft_Phi_3_mini_4k_instruct',
    'internlm_internlm2_5_1_8b_chat'
]

# Model display names (shortened)
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
    correct = (subset[model_col] == subset['ANSWER']).sum()
    return (correct / len(subset) * 100)

# Reorder categories to match the reference (by human performance)
categories_ordered = ['Desire', 'Knowledge', 'Belief', 'Emotion', 'Non-Literal Communication', 'Intention']

# Create gradient colors based on correlation values
def get_bar_color(corr):
    if corr < -0.4:
        return '#4B0082'  # Dark purple
    elif corr < -0.2:
        return '#663399'  # Medium purple
    elif corr < 0:
        return '#8B4789'  # Light purple
    elif corr < 0.2:
        return '#CD5C5C'  # Light red
    elif corr < 0.4:
        return '#FF6347'  # Tomato
    else:
        return '#FFA07A'  # Light salmon

#%%
# MSVD Correlation Plot
fig, ax = plt.subplots(figsize=(10,9))
fig.patch.set_facecolor('white')

correlations_msvd = []
subjects = ['Human'] + [model_display_names[model] for model in models]

# Calculate correlations with MSVD Overall
for i, subject in enumerate(subjects):
    if subject == 'Human':
        x = [msvd_overall_by_category[cat] for cat in categories_ordered]
        y = [human_performance[cat] for cat in categories_ordered]
    else:
        # Find the full model name
        full_model = next(m for m, display in model_display_names.items() if display == subject)
        x = []
        y = []
        for cat in categories_ordered:
            x.append(msvd_overall_by_category[cat])
            y.append(calculate_accuracy(df_clean, full_model, cat))
    
    corr = np.corrcoef(x, y)[0, 1]
    correlations_msvd.append(corr)

bar_colors_msvd = [get_bar_color(corr) for corr in correlations_msvd]

# Create bar plot
bars = ax.bar(subjects, correlations_msvd, color=bar_colors_msvd, 
               alpha=0.85, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, corr in zip(bars, correlations_msvd):
    if corr >= 0:
        y_pos = bar.get_height() + 0.015
        va = 'bottom'
    else:
        y_pos = bar.get_height() - 0.015
        va = 'top'
    
    color = '#2E8B57' if corr > 0 else '#8B0000'
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
            f'{corr:.3f}', ha='center', va=va, 
            fontsize=18, fontweight='bold', color=color)

# Customize the plot
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax.set_ylabel('Pearson Correlation Coefficient', fontsize=20, fontweight='bold')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right', fontsize=20)
plt.yticks(fontsize=25)

# Remove grid
ax.grid(False)
# Set y-axis limits tighter to the data
max_corr = max(correlations_msvd)
min_corr = min(correlations_msvd)
y_margin = 0.05  # Small margin for readability
ax.set_ylim(min_corr - y_margin, max_corr + y_margin)

# Style the plot
ax.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('msvd_correlation_combined.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Idea Density Correlation Plot
fig, ax = plt.subplots(figsize=(10,9))
fig.patch.set_facecolor('white')

correlations_idea = []

# Calculate correlations with Idea Density
for i, subject in enumerate(subjects):
    if subject == 'Human':
        x = [idea_density_by_category[cat] for cat in categories_ordered]
        y = [human_performance[cat] for cat in categories_ordered]
    else:
        # Find the full model name
        full_model = next(m for m, display in model_display_names.items() if display == subject)
        x = []
        y = []
        for cat in categories_ordered:
            x.append(idea_density_by_category[cat])
            y.append(calculate_accuracy(df_clean, full_model, cat))
    
    corr = np.corrcoef(x, y)[0, 1]
    correlations_idea.append(corr)

bar_colors_idea = [get_bar_color(corr) for corr in correlations_idea]

# Create bar plot
bars = ax.bar(subjects, correlations_idea, color=bar_colors_idea, 
               alpha=0.85, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, corr in zip(bars, correlations_idea):
    if corr >= 0:
        y_pos = bar.get_height() + 0.015
        va = 'bottom'
    else:
        y_pos = bar.get_height() - 0.015
        va = 'top'
    
    color = '#2E8B57' if corr > 0 else '#8B0000'
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
            f'{corr:.3f}', ha='center', va=va, 
            fontsize=18, fontweight='bold', color=color)

# Customize the plot
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax.set_ylabel('Pearson Correlation Coefficient', fontsize=20, fontweight='bold')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right', fontsize=20)
plt.yticks(fontsize=25)

# Remove grid
ax.grid(False)
# Set y-axis limits tighter to the data
max_corr = max(correlations_idea)
min_corr = min(correlations_idea)
y_margin = 0.05  # Small margin for readability
ax.set_ylim(min_corr - y_margin, max_corr + y_margin)

# Style the plot
ax.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('idea_density_correlation_combined.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Complexity Plot
fig, ax = plt.subplots(figsize=(10,9))
fig.patch.set_facecolor('white')

# Define complexity levels in order
complexity_levels = ['SIMPLE_STORY', 'LOW', 'MEDIUM', 'HIGH']

# Order models by size (largest to smallest) and assign rocket palette colors
models_by_size = [
    ('Llama 3.1 70B', 70),
    ('Qwen 2.5 32B', 32), 
    ('OLMo 13B', 13),
    ('Mistral 7B', 7),
    ('Phi-3 Mini', 4),  # Phi-3 mini 4k
    ('InternLM 1.8B', 1.8)
]

# Get rocket palette colors
rocket_colors = sns.color_palette("rocket", len(models_by_size))

# Create color mapping ordered by model size (largest gets darkest rocket color)
complexity_colors = {}
for i, (model_name, size) in enumerate(models_by_size):
    complexity_colors[model_name] = rocket_colors[i]

# Calculate and plot for each LLM
for model_col, model_name in model_display_names.items():
    accuracies = []
    
    for level in complexity_levels:
        # Filter data for this complexity level
        level_data = df[df['Complexity_Level'] == level]
        
        # Calculate accuracy (excluding undefined/NaN responses)
        valid_responses = level_data[pd.notna(level_data[model_col])]
        
        if len(valid_responses) > 0:
            correct = sum(valid_responses[model_col] == valid_responses['ANSWER'])
            accuracy = (correct / len(valid_responses)) * 100
        else:
            accuracy = 0
        
        accuracies.append(accuracy)
    
    # Plot line for this model
    ax.plot(complexity_levels, accuracies, 
            marker='o', linewidth=3, markersize=8,
            label=model_name, color=complexity_colors[model_name])

# Customize the plot
ax.set_ylabel('Accuracy (%)', fontsize=20, fontweight='bold')
ax.set_xlabel('Complexity Level', fontsize=20, fontweight='bold')

# Set tick label sizes
plt.xticks(fontsize=20)
plt.yticks(fontsize=25)

# Set y-axis range
ax.set_ylim(45, 100)

# Add legend
ax.legend(fontsize=16, loc='upper left')

# Remove grid
ax.grid(False)

# Style the plot
ax.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('complexity_combined.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
