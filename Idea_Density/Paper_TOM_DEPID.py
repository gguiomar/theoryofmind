#%%
import pandas as pd
from ideadensity import depid

# Load your dataset
df = pd.read_csv('../dataset.csv')

# Function to calculate idea density and word count
def calculate_idea_metrics(text):
        if pd.isna(text) or text.strip() == '':
            return pd.Series({'Idea_Density': None, 'Word_Count': None})
        density, word_count, dependencies = depid(text)
        return pd.Series({
            'Idea_Density': round(density, 3),
            'Word_Count': int(word_count)
        })

# Apply and create multiple columns
df[['Idea_Density', 'Word_Count']] = df['STORY'].apply(calculate_idea_metrics)

# Convert to integer type 
df['Word_Count'] = df['Word_Count'].astype('Int64')

# to place after STORY
story_index = df.columns.get_loc('STORY')
cols = df.columns.tolist()
cols.remove('Idea_Density')
cols.remove('Word_Count')
cols.insert(story_index + 1, 'Idea_Density')
cols.insert(story_index + 2, 'Word_Count')
df = df[cols]

# Save the updated dataset
df.to_csv('dataset_with_idea_density.csv', index=False)

# %%
