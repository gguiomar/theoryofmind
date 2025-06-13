#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('./dataset.csv')
df.columns = df.columns.str.strip()

# Fix the ANSWER column name issue (likely has newline prefix)
answer_col = '\nANSWER' if '\nANSWER' in df.columns else 'ANSWER'
print(f"Using answer column: {answer_col}")

df_clean = df[df['ABILITY'].notna() & df['Idea_Density'].notna()].copy()
df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
# Map "Non-Literal Communication" to "NLC" for consistency
df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')

# Calculate average idea density per category
idea_density_by_category = df_clean.groupby('Main_Category')['Idea_Density'].mean()

# Human performance data (from the original paper)
human_performance = {
    'Emotion': 86.4,
    'Desire': 90.4,
    'Intention': 82.2,
    'Knowledge': 89.3,
    'Belief': 89.0,
    'NLC': 86.1  # Non-Literal Communication shortened
}

# Model list with proper name mapping
models = [
    'meta_llama_Llama_3.1_70B_Instruct',
    'Qwen_Qwen2.5_32B_Instruct',
    'allenai_OLMo_2_1124_13B_Instruct',
    'mistralai_Mistral_7B_Instruct_v0.3',
    'microsoft_Phi_3_mini_4k_instruct',
    'internlm_internlm2_5_1_8b_chat'
]

# Model name mapping for display
model_display_names = {
    'meta_llama_Llama_3.1_70B_Instruct': 'Llama 3.1 70B',
    'Qwen_Qwen2.5_32B_Instruct': 'Qwen 2.5 32B',
    'allenai_OLMo_2_1124_13B_Instruct': 'OLMo 13B',
    'mistralai_Mistral_7B_Instruct_v0.3': 'Mistral 7B',
    'microsoft_Phi_3_mini_4k_instruct': 'Phi-3 Mini',
    'internlm_internlm2_5_1_8b_chat': 'InternLM 1.8B'
}

# Calculate model performance with proper answer column
def calculate_accuracy(df, model_col, category):
    subset = df[df['Main_Category'] == category]
    if len(subset) == 0:
        return 0
    correct = (subset[model_col] == subset[answer_col]).sum()
    return (correct / len(subset) * 100)

# Define categories
categories = list(human_performance.keys())

# Prepare data for analysis
performance_matrix = pd.DataFrame(index=categories)
performance_matrix['Human'] = [human_performance[cat] for cat in categories]

# Calculate model performances
for model in models:
    display_name = model_display_names[model]
    performance_matrix[display_name] = [
        calculate_accuracy(df_clean, model, cat) for cat in categories
    ]

# Add idea density for reference
performance_matrix.insert(0, 'Idea_Density', 
                         [idea_density_by_category[cat] for cat in categories])

# UNIT TESTS AND VALIDATION
print("=== UNIT TESTS AND VALIDATION ===")

# Test 1: Check if correlations are calculated correctly
print("\n1. Testing correlation calculations:")
test_x = [1, 2, 3, 4, 5]
test_y = [2, 4, 6, 8, 10]
expected_corr = 1.0
calculated_corr = np.corrcoef(test_x, test_y)[0, 1]
print(f"   Perfect correlation test: Expected {expected_corr}, Got {calculated_corr:.3f}")
assert abs(calculated_corr - expected_corr) < 0.001, "Correlation calculation failed"

# Test 2: Check data consistency
print("\n2. Checking data consistency:")
print(f"   Categories in idea density: {set(idea_density_by_category.index)}")
print(f"   Categories in human performance: {set(human_performance.keys())}")
print(f"   Categories match: {set(idea_density_by_category.index) == set(human_performance.keys())}")

# Test 3: Check if model performances are reasonable (not all zeros)
print("\n3. Checking model performance calculations:")
for model in models:
    display_name = model_display_names[model]
    avg_perf = performance_matrix[display_name].mean()
    print(f"   {display_name}: Average performance = {avg_perf:.1f}%")
    if avg_perf == 0:
        print(f"   WARNING: {display_name} has zero performance - check data!")

# Calculate correlations with proper error handling
correlations = []
p_values = []
subjects = ['Human'] + list(model_display_names.values())

print("\n4. Calculating correlations:")
for subject in subjects:
    if subject == 'Human':
        x = [idea_density_by_category[cat] for cat in categories]
        y = [human_performance[cat] for cat in categories]
    else:
        x = [idea_density_by_category[cat] for cat in categories]
        y = [performance_matrix.loc[cat, subject] for cat in categories]
    
    # Calculate correlation and p-value
    corr, p_val = stats.pearsonr(x, y)
    correlations.append(corr)
    p_values.append(p_val)
    
    print(f"   {subject}: r = {corr:.3f}, p = {p_val:.3f}")

# Test 4: Check for contradictions between plots
print("\n5. Checking for contradictions:")
idea_density_order = idea_density_by_category.sort_values(ascending=False).index.tolist()
print(f"   Categories by idea density (high to low): {idea_density_order}")

for subject in subjects:
    if subject == 'Human':
        perf_data = [human_performance[cat] for cat in idea_density_order]
    else:
        perf_data = [performance_matrix.loc[cat, subject] for cat in idea_density_order]
    
    corr_val = correlations[subjects.index(subject)]
    if corr_val < 0:
        expected_trend = "decreasing"
    else:
        expected_trend = "increasing"
    
    print(f"   {subject}: Correlation = {corr_val:.3f} (should show {expected_trend} trend)")

print("\n=== CREATING IMPROVED VISUALIZATIONS ===")

#%%
# Create improved grid plot with seaborn rocket colormap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 0.7]})

# Plot 1: Compact heatmap without numbers
performance_data = performance_matrix.iloc[:, 1:]  # Exclude idea density column
sns.heatmap(performance_data.T, 
            cmap='rocket_r',  # Seaborn rocket colormap (reversed for better readability)
            cbar_kws={'label': 'Performance (%)', 'shrink': 0.8},
            square=True,
            linewidths=2.0,  # Make lines thicker as requested
            linecolor='white',
            ax=ax1,
            annot=False)  # Remove numbers as requested

ax1.set_title('Performance by ToM Category', fontweight='bold', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
ax1.tick_params(axis='y', rotation=0)

# Plot 2: Clean correlation bar plot
rocket_colors = sns.color_palette('rocket', len(subjects))
bars = ax2.bar(range(len(subjects)), correlations, 
               color=rocket_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

# Remove correlation values above bars as requested

# Clean up bar plot
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
ax2.set_title('Idea Density vs Performance Correlation', fontweight='bold', fontsize=12)
ax2.set_ylabel('Pearson Correlation', fontweight='bold')
ax2.set_xticks(range(len(subjects)))
ax2.set_xticklabels(subjects, rotation=45, ha='right')
ax2.set_ylim(-0.7, 0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Remove background grid as requested
ax2.grid(False)

plt.tight_layout()
plt.show()

#%%
# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Idea density range: {idea_density_by_category.min():.3f} - {idea_density_by_category.max():.3f}")
print(f"Average correlation: {np.mean(correlations):.3f}")
print(f"Significant correlations (p < 0.05): {sum(p < 0.05 for p in p_values)}/{len(p_values)}")

# Check consistency between plots
print("\n=== CONSISTENCY CHECK ===")
print("Both plots should tell the same story:")
print("- Heatmap shows performance patterns across categories")
print("- Bar plot shows correlation between idea density and performance")
print("- If correlation is negative, categories with higher idea density should show lower performance")
print("- If correlation is positive, categories with higher idea density should show higher performance")

# Display the performance matrix for reference
print("\n=== PERFORMANCE MATRIX ===")
print(performance_matrix.round(2))

print("\n=== ANALYSIS COMPLETE ===")

# %%

# sort performance matrix by performance in humans 

performance_matrix = performance_matrix.sort_values(by='Human', ascending=False)