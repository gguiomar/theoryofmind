import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TO DO:
# - If both plots do not contradict each other
# - If it is calculated correctly
# - Make plots more pretty and more readable 

df = pd.read_csv('./dataset.csv')
df.columns = df.columns.str.strip()
df_clean = df[df['ABILITY'].notna() & df['Idea_Density'].notna()].copy()
df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()

# Calculate average idea density per category
idea_density_by_category = df_clean.groupby('Main_Category')['Idea_Density'].mean()

# Human performance data (from the original paper)
human_performance = {
    'Emotion': 86.4,
    'Desire': 90.4,
    'Intention': 82.2,
    'Knowledge': 89.3,
    'Belief': 89.0,
    'Non-Literal Communication': 86.1
}

# Model list that we usu
models = [
    'meta_llama_Llama_3.1_70B_Instruct',
    'Qwen_Qwen2.5_32B_Instruct',
    'allenai_OLMo_2_1124_13B_Instruct',
    'mistralai_Mistral_7B_Instruct_v0.3',
    'microsoft_Phi_3_mini_4k_instruct',
    'internlm_internlm2_5_1_8b_chat'
]

# Calculate model performance
def calculate_accuracy(df, model_col, category):
    subset = df[df['Main_Category'] == category]
    if len(subset) == 0:
        return 0
    correct = (subset[model_col] == subset['ANSWER']).sum()
    return (correct / len(subset) * 100)

# Define categories
categories = list(human_performance.keys())

# Create a heatmap showing performance by category
plt.figure(figsize=(12, 8))

# Prepare data for heatmap
performance_matrix = pd.DataFrame(index=categories)
performance_matrix['Human'] = [human_performance[cat] for cat in categories]

for model in models:
    model_name = model.split('_')[0]
    performance_matrix[model_name] = [
        calculate_accuracy(df_clean, model, cat) for cat in categories
    ]

# Add idea density as the first column for reference
performance_matrix.insert(0, 'Idea_Density', 
                         [idea_density_by_category[cat] for cat in categories])

# Create heatmap
sns.heatmap(performance_matrix.iloc[:, 1:], annot=True, fmt='.1f', 
            cmap='RdYlGn', center=70, vmin=40, vmax=95,
            cbar_kws={'label': 'Performance (%)'})

plt.title('Performance Heatmap by ToM Category', fontsize=14, fontweight='bold')
plt.xlabel('')
plt.ylabel('ToM Category', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Add idea density annotations on the left
for i, (idx, row) in enumerate(performance_matrix.iterrows()):
    plt.text(-0.5, i + 0.5, f'{row["Idea_Density"]:.3f}', 
             ha='right', va='center', fontsize=9, fontweight='bold')

plt.text(-0.5, -0.7, 'Idea\nDensity', ha='right', va='top', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Create correlation plot
plt.figure(figsize=(10, 6))

correlations = []
subjects = ['Human'] + [model.split('_')[0] for model in models]

# Calculate correlations
for subject in subjects:
    if subject == 'Human':
        x = [idea_density_by_category[cat] for cat in categories]
        y = [human_performance[cat] for cat in categories]
    else:
        # Find the full model name
        full_model = next(m for m in models if m.startswith(subject))
        x = []
        y = []
        for cat in categories:
            x.append(idea_density_by_category[cat])
            y.append(calculate_accuracy(df_clean, full_model, cat))
    
    corr = np.corrcoef(x, y)[0, 1]
    correlations.append(corr)

# Define colors for bars
subject_colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))

# Create bar plot of correlations
bars = plt.bar(subjects, correlations, color=subject_colors, 
                alpha=0.7, edgecolor='black')

# Add value labels
for bar, corr in zip(bars, correlations):
    color = 'green' if corr > 0 else 'red'
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{corr:.3f}', ha='center', va='bottom', fontsize=10, color=color)

plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.title('Correlation between Idea Density and Performance', fontsize=14, fontweight='bold')
plt.ylabel('Pearson Correlation Coefficient', fontsize=12)
plt.xlabel('Subject', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(-0.5, 0.5)

plt.tight_layout()
plt.show()