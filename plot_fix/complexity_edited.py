#%% 

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

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

# Define colors for each model
colors = {
    'Llama 3.1 70B': '#1f77b4',
    'Qwen 2.5 32B': '#ff7f0e',
    'OLMo 2 13B': '#2ca02c',
    'Mistral 7B v0.3': '#d62728',
    'Phi-3 mini 4k': '#9467bd',
    'InternLM 2.5 1.8B': '#8c564b'
}

# Create the figure
fig = go.Figure()

# Calculate and plot for each LLM
for llm_col, llm_name in llm_columns.items():
    accuracies = []
    hover_texts = []
    
    for level in complexity_levels:
        # Filter data for this complexity level
        level_data = df[df['Complexity_Level'] == level]
        
        # Calculate accuracy (excluding undefined/NaN responses)
        valid_responses = level_data[pd.notna(level_data[llm_col])]
        
        if len(valid_responses) > 0:
            correct = sum(valid_responses[llm_col] == valid_responses['ANSWER'])
            accuracy = (correct / len(valid_responses)) * 100
            hover_text = f"{llm_name}<br>{level}<br>Accuracy: {accuracy:.1f}%<br>Correct: {correct}/{len(valid_responses)}"
        else:
            accuracy = 0
            hover_text = f"{llm_name}<br>{level}<br>No valid responses"
        
        accuracies.append(accuracy)
        hover_texts.append(hover_text)
    
    # Add trace for this LLM
    fig.add_trace(go.Scatter(
        x=complexity_levels,
        y=accuracies,
        mode='lines+markers',
        name=llm_name,
        line=dict(color=colors[llm_name], width=3),
        marker=dict(size=8),
        hovertext=hover_texts,
        hoverinfo='text'
    ))

# Update layout with corrected title structure
fig.update_layout(
    # title=dict(
    #     text='Complexity Categories and LLM Performance',
    #     font=dict(size=20, family='Arial Black'),
    #     x=0.5,
    #     xanchor='center'
    # ),
    xaxis=dict(
        title=dict(
            text='Complexity Level',
            font=dict(size=14)
        ),
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title=dict(
            text='Accuracy (%)',
            font=dict(size=14)
        ),
        tickfont=dict(size=12),
        range=[45, 100],
        dtick=10
    ),
    legend=dict(
        x=1.02,
        y=0.5,
        yanchor='middle',
        font=dict(size=11)
    ),
    hovermode='x unified',
    template='plotly_white',
    margin=dict(t=10, l=60, r=150, b=60),
    width=700,
    height=400
)

# Add grid
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

# Save as PNG image
fig.write_image("complexity.png", width=700, height=400, scale=2)
print("Plot saved as complexity.png")

# Also save as HTML for interactive viewing (optional)
# fig.write_html("complexity_paradox.html")

# If you want to display it immediately (requires PIL/Pillow):
# from PIL import Image
# img = Image.open("complexity_paradox.png")
# img.show()
# %%
