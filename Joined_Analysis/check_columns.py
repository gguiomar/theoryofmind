import pandas as pd

df = pd.read_csv('dataset_joined.csv')

print('RST columns check:')
rst_cols = ['num_edus', 'tree_depth', 'rel_attribution', 'rel_causal', 'rel_explanation']
for col in rst_cols:
    print(f'{col}: {col in df.columns}')

print('\nAll analysis columns:')
analysis_cols = ['Idea_Density', 'Question_Complexity_Score'] + rst_cols
print(analysis_cols)

print('\nModel columns:')
model_cols = [col for col in df.columns if any(model in col.lower() for model in ['qwen', 'olmo', 'mistral', 'phi', 'internlm'])]
print(model_cols)

print('\nAbility column values (first 10):')
print(df['\nABILITY'].unique()[:10])
