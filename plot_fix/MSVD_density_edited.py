import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Read and prepare data
df = pd.read_csv('./dataset_2.csv')
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

# Create performance matrix with proper ordering
performance_data = []
row_labels = ['Human']

# Add human data
human_row = [human_performance[cat] for cat in categories_ordered]
performance_data.append(human_row)

# Add model data
for model in models:
    model_row = [calculate_accuracy(df_clean, model, cat) for cat in categories_ordered]
    performance_data.append(model_row)
    row_labels.append(model_display_names[model])

# Convert to DataFrame
performance_df = pd.DataFrame(performance_data, 
                            index=row_labels,
                            columns=categories_ordered)

# Create figure with more space at the top for MSVD overall
fig = plt.figure(figsize=(12, 9))
ax = plt.subplot(111)

# Adjust subplot to leave room at the top for MSVD overall
plt.subplots_adjust(top=0.88, bottom=0.12, left=0.22, right=0.92)

# Create custom colormap similar to the reference
colors = ['#8B0000', '#CD5C5C', '#FFA07A', '#FFE4B5', '#9370DB', '#6B4C7A', '#483D8B', '#191970']
n_bins = 256
cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Create the heatmap
sns.heatmap(performance_df, 
            annot=True, 
            fmt='.1f', 
            cmap=cmap,
            vmin=30, 
            vmax=95,
            cbar_kws={'label': 'Performance (%)', 'shrink': 0.8, 'pad': 0.02},
            linewidths=2,
            linecolor='white',
            square=False,
            annot_kws={'fontsize': 11, 'fontweight': 'bold', 'color': 'white'},
            ax=ax)

# Set title (positioned lower to make room for MSVD overall)
# ax.set_title('Performance by Theory of Mind Category', 
#              fontsize=18, fontweight='bold', pad=40)

# Set axis labels
ax.set_xlabel('ToM Categories', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('Models', fontsize=14, fontweight='bold', labelpad=10)

# Customize x-axis labels (abbreviate Non-Literal Communication)
x_labels = ['Desire', 'Knowledge', 'Belief', 'Emotion', 'NLC', 'Intention']
ax.set_xticklabels(x_labels, fontsize=12, rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, rotation=0)

# Add "MSVD Overall" label at the top
ax.text(-1.2, len(row_labels) + 0.5, 'MSVD\nOverall', 
        transform=ax.transData,
        ha='center', va='center', 
        fontsize=11, fontweight='bold')

# Add MSVD overall values at the very top
for i, cat in enumerate(categories_ordered):
    msvd_val = msvd_overall_by_category[cat]
    # Position at the top of the figure
    ax.text(i + 0.5, len(row_labels) + 0.5, f'{msvd_val:.3f}', 
            transform=ax.transData,
            ha='center', va='center', 
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
    
# Add "Idea Density" label at the top
ax.text(-1.2, len(row_labels) + 1.1, 'Idea\nDensity', 
        transform=ax.transData,
        ha='center', va='center', 
        fontsize=11, fontweight='bold')

# Add idea density values at the very top
for i, cat in enumerate(categories_ordered):
    idea_dens = idea_density_by_category[cat]
    # Position at the top of the figure
    ax.text(i + 0.5, len(row_labels) + 1.1, f'{idea_dens:.3f}', 
            transform=ax.transData,
            ha='center', va='center', 
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

# Customize colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Performance (%)', fontsize=12, fontweight='bold')

# Extend y-axis to show MSVD overall values at the top
ax.set_ylim(-0.5, len(row_labels) + 1.2)

plt.show()

# Create the correlation plot with professional styling
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')

correlations = []
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
    correlations.append(corr)

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

bar_colors = [get_bar_color(corr) for corr in correlations]

# Create bar plot
bars = ax.bar(subjects, correlations, color=bar_colors, 
               alpha=0.85, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, corr in zip(bars, correlations):
    if corr >= 0:
        y_pos = bar.get_height() + 0.015
        va = 'bottom'
    else:
        y_pos = bar.get_height() - 0.015
        va = 'top'
    
    color = '#2E8B57' if corr > 0 else '#8B0000'
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
            f'{corr:.3f}', ha='center', va=va, 
            fontsize=11, fontweight='bold', color=color)

# Customize the plot
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
# ax.set_title('Correlation between MSVD Overall and Performance', 
#            fontsize=18, fontweight='bold', pad=20)
ax.set_ylabel('Pearson Correlation Coefficient', fontsize=14, fontweight='bold')
ax.set_xlabel('Subject', fontsize=14, fontweight='bold')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)

# Add grid
ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
ax.set_ylim(-0.6, 0.6)

# Style the plot
ax.set_facecolor('#FAFAFA')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.show()

# Create an alternative version with MSVD overall as a separate row
print("\nCreating alternative layout with MSVD overall as a row...\n")

# Create extended dataframe with MSVD overall as first row
extended_data = []
extended_labels = ['MSVD Overall'] + row_labels

# Add MSVD overall row
msvd_overall_row = [msvd_overall_by_category[cat] for cat in categories_ordered]
extended_data.append(msvd_overall_row)

# Add all performance data
extended_data.extend(performance_data)

# Create extended DataFrame
extended_df = pd.DataFrame(extended_data, 
                          index=extended_labels,
                          columns=categories_ordered)

# Create figure
fig, ax = plt.subplots(figsize=(12, 9))

# Create mask to use different colormap for MSVD overall row
mask = np.zeros_like(extended_df, dtype=bool)
mask[0, :] = True  # Mask the first row (MSVD overall)

# Plot the main heatmap (without MSVD overall row)
sns.heatmap(extended_df.iloc[1:, :], 
            annot=True, 
            fmt='.1f', 
            cmap=cmap,
            vmin=30, 
            vmax=95,
            cbar_kws={'label': 'Performance (%)', 'shrink': 0.7},
            linewidths=2,
            linecolor='white',
            square=False,
            annot_kws={'fontsize': 11, 'fontweight': 'bold', 'color': 'white'},
            ax=ax)

# Manually add MSVD overall row with different styling
for i, cat in enumerate(categories_ordered):
    rect = Rectangle((i, len(extended_labels)-1.5), 1, 0.5, 
                     facecolor='lightgray', edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(i + 0.5, len(extended_labels)-1.25, f'{msvd_overall_row[i]:.3f}',
            ha='center', va='center', fontsize=11, fontweight='bold')

# Add horizontal line to separate MSVD overall from performance data
ax.axhline(y=len(row_labels), color='black', linewidth=2)

# Set labels
ax.set_title('Performance by Theory of Mind Category', 
              fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('ToM Categories', fontsize=14, fontweight='bold')
ax.set_ylabel('', fontsize=14)

# Fix y-axis labels to include "MSVD Overall"
y_labels = ['MSVD Overall'] + row_labels
ax.set_yticklabels(y_labels, fontsize=12, rotation=0)

# Fix x-axis labels
x_labels = ['Desire', 'Knowledge', 'Belief', 'Emotion', 'NLC', 'Intention']
ax.set_xticklabels(x_labels, fontsize=12, rotation=0)

plt.tight_layout()
plt.show()

plt.savefig('MSVD_Idea_combined.png', dpi=100, bbox_inches='tight')