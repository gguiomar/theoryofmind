#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style and remove grid
sns.set_style("white")
plt.rcParams['axes.grid'] = False

# Read the CSV file
df = pd.read_csv('main_with_perspective_annotations.csv')

# Define the LLM columns and their display names
llm_columns = {
    'meta_llama_Llama_3.1_70B_Instruct': 'Llama 3.1 70B',
    'Qwen_Qwen2.5_32B_Instruct': 'Qwen 2.5 32B',
    'allenai_OLMo_2_1124_13B_Instruct': 'OLMo 2 13B',
    'mistralai_Mistral_7B_Instruct_v0.3': 'Mistral 7B v0.3',
    'microsoft_Phi_3_mini_4k_instruct': 'Phi-3 mini 4k',
    'internlm_internlm2_5_1_8b_chat': 'InternLM 2.5 1.8B'
}

# Define complexity levels in order
complexity_levels = ['SIMPLE_STORY', 'LOW', 'MEDIUM', 'HIGH']

# Use seaborn crest color palette
colors = sns.color_palette("crest", len(llm_columns))

#%%
# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate and plot for each LLM
for i, (llm_col, llm_name) in enumerate(llm_columns.items()):
    accuracies = []
    
    for level in complexity_levels:
        # Filter data for this complexity level
        level_data = df[df['Complexity_Level'] == level]
        
        # Calculate accuracy (excluding undefined/NaN responses)
        valid_responses = level_data[pd.notna(level_data[llm_col])]
        
        if len(valid_responses) > 0:
            correct = sum(valid_responses[llm_col] == valid_responses['ANSWER'])
            accuracy = (correct / len(valid_responses)) * 100
        else:
            accuracy = 0
        
        accuracies.append(accuracy)
    
    # Add line plot without markers
    ax.plot(complexity_levels, accuracies, 
            color=colors[i], 
            linewidth=3, 
            label=llm_name)

# Customize the plot
ax.set_xlabel('Complexity Level', fontsize=20, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=20, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_ylim(45, 100)

# Remove top and right spines
sns.despine()

# Add legend
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

# Tight layout
plt.tight_layout()

# Save the plot
plt.savefig("complexity_seaborn.png", dpi=300, bbox_inches='tight')
print("Plot saved as complexity_seaborn.png")

plt.show()

# %%
