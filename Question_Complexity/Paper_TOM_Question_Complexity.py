#%%
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from transformers import pipeline
import re

class QuestionComplexityAnalyzer:
    """
    Analyze question complexity for Theory of Mind (ToM) tasks.
    
    Measures multiple dimensions of complexity:
    - Syntactic complexity (dependency depth, clause count)
    - Semantic complexity (word embeddings, concept diversity)
    - Pragmatic complexity (mental state references, perspective shifts)
    - Reasoning complexity (logical operators, temporal references)
    """
    
    def __init__(self):
        """Initialize the analyzer with required models."""
        print("Loading models...")
        
        # Load spaCy model for linguistic analysis
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load sentence transformer for semantic analysis
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Mental state verbs and concepts for ToM analysis
        self.mental_state_verbs = {
            'think', 'believe', 'know', 'understand', 'realize', 'assume',
            'suppose', 'imagine', 'feel', 'want', 'hope', 'expect', 'doubt',
            'wonder', 'guess', 'suspect', 'remember', 'forget', 'notice'
        }
        
        self.perspective_markers = {
            'he thinks', 'she believes', 'they know', 'he feels', 'she wants',
            'his opinion', 'her view', 'their perspective', 'from his point',
            'in her mind', 'he assumes', 'she expects', 'they suppose'
        }
        
        print("✓ Models loaded successfully")
    
    def analyze_question(self, question):
        """Analyze a single question for complexity metrics."""
        if pd.isna(question) or question.strip() == '':
            return self._get_empty_features()
        
        # Clean the question
        question = question.strip()
        
        # Parse with spaCy
        doc = self.nlp(question)
        
        features = {}
        
        # 1. Basic metrics
        features.update(self._basic_metrics(question, doc))
        
        # 2. Syntactic complexity
        features.update(self._syntactic_complexity(doc))
        
        # 3. Semantic complexity
        features.update(self._semantic_complexity(question, doc))
        
        # 4. ToM-specific complexity
        features.update(self._tom_complexity(question, doc))
        
        # 5. Reasoning complexity
        features.update(self._reasoning_complexity(question, doc))
        
        return features
    
    def _basic_metrics(self, question, doc):
        """Calculate basic question metrics."""
        return {
            'question_length': len(question),
            'word_count': len([token for token in doc if not token.is_space]),
            'sentence_count': len(list(doc.sents)),
            'avg_word_length': np.mean([len(token.text) for token in doc if token.is_alpha])
        }
    
    def _syntactic_complexity(self, doc):
        """Calculate syntactic complexity metrics."""
        # Dependency tree depth
        def get_depth(token, depth=0):
            if not list(token.children):
                return depth
            return max(get_depth(child, depth + 1) for child in token.children)
        
        depths = [get_depth(sent.root) for sent in doc.sents]
        max_depth = max(depths) if depths else 0
        avg_depth = np.mean(depths) if depths else 0
        
        # Count different dependency types
        dep_types = set(token.dep_ for token in doc)
        
        # Count clauses (approximated by subordinating conjunctions and relative pronouns)
        clause_markers = ['SCONJ', 'PRON']  # subordinating conjunctions, relative pronouns
        clause_count = sum(1 for token in doc if token.pos_ in clause_markers)
        
        return {
            'max_dependency_depth': max_depth,
            'avg_dependency_depth': avg_depth,
            'unique_dependency_types': len(dep_types),
            'clause_count': clause_count
        }
    
    def _semantic_complexity(self, question, doc):
        """Calculate semantic complexity metrics."""
        # Lexical diversity (Type-Token Ratio)
        words = [token.lemma_.lower() for token in doc if token.is_alpha]
        ttr = len(set(words)) / len(words) if words else 0
        
        # Named entities
        entity_count = len(doc.ents)
        entity_types = len(set(ent.label_ for ent in doc.ents))
        
        # Content word ratio
        content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
        content_words = sum(1 for token in doc if token.pos_ in content_pos)
        content_ratio = content_words / len(doc) if len(doc) > 0 else 0
        
        return {
            'lexical_diversity': ttr,
            'entity_count': entity_count,
            'entity_type_count': entity_types,
            'content_word_ratio': content_ratio
        }
    
    def _tom_complexity(self, question, doc):
        """Calculate Theory of Mind specific complexity."""
        question_lower = question.lower()
        
        # Mental state verb count
        mental_state_count = sum(1 for token in doc 
                                if token.lemma_.lower() in self.mental_state_verbs)
        
        # Perspective markers
        perspective_count = sum(1 for marker in self.perspective_markers 
                              if marker in question_lower)
        
        # Person references (1st, 2nd, 3rd person)
        person_pronouns = {
            'first': {'i', 'me', 'my', 'mine', 'myself'},
            'second': {'you', 'your', 'yours', 'yourself'},
            'third': {'he', 'she', 'him', 'her', 'his', 'hers', 'they', 'them', 'their'}
        }
        
        person_counts = {}
        for person, pronouns in person_pronouns.items():
            count = sum(1 for token in doc if token.lemma_.lower() in pronouns)
            person_counts[f'{person}_person_count'] = count
        
        # Question type analysis
        question_words = {'what', 'who', 'when', 'where', 'why', 'how', 'which'}
        question_word_count = sum(1 for token in doc 
                                if token.lemma_.lower() in question_words)
        
        return {
            'mental_state_verbs': mental_state_count,
            'perspective_markers': perspective_count,
            'question_word_count': question_word_count,
            **person_counts
        }
    
    def _reasoning_complexity(self, question, doc):
        """Calculate reasoning complexity metrics."""
        question_lower = question.lower()
        
        # Logical operators
        logical_ops = ['and', 'or', 'but', 'if', 'then', 'because', 'since', 'although']
        logical_count = sum(1 for op in logical_ops if op in question_lower)
        
        # Temporal references
        temporal_words = ['before', 'after', 'when', 'while', 'during', 'then', 'now', 'later']
        temporal_count = sum(1 for word in temporal_words if word in question_lower)
        
        # Negation
        negation_count = sum(1 for token in doc if token.dep_ == 'neg')
        
        # Conditional structures
        conditional_count = question_lower.count('if') + question_lower.count('would')
        
        return {
            'logical_operators': logical_count,
            'temporal_references': temporal_count,
            'negation_count': negation_count,
            'conditional_count': conditional_count
        }
    
    def _get_empty_features(self):
        """Return empty feature dict when analysis fails."""
        return {
            # Basic metrics
            'question_length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            
            # Syntactic complexity
            'max_dependency_depth': 0,
            'avg_dependency_depth': 0,
            'unique_dependency_types': 0,
            'clause_count': 0,
            
            # Semantic complexity
            'lexical_diversity': 0,
            'entity_count': 0,
            'entity_type_count': 0,
            'content_word_ratio': 0,
            
            # ToM complexity
            'mental_state_verbs': 0,
            'perspective_markers': 0,
            'question_word_count': 0,
            'first_person_count': 0,
            'second_person_count': 0,
            'third_person_count': 0,
            
            # Reasoning complexity
            'logical_operators': 0,
            'temporal_references': 0,
            'negation_count': 0,
            'conditional_count': 0
        }

def calculate_question_complexity(text):
    """Calculate question complexity metrics for a single question."""
    if pd.isna(text) or text.strip() == '':
        return pd.Series({
            'Question_Complexity_Score': None,
            'Syntactic_Complexity': None,
            'Semantic_Complexity': None,
            'ToM_Complexity': None,
            'Reasoning_Complexity': None
        })
    
    # Initialize analyzer (this should be done once globally for efficiency)
    if not hasattr(calculate_question_complexity, 'analyzer'):
        calculate_question_complexity.analyzer = QuestionComplexityAnalyzer()
    
    analyzer = calculate_question_complexity.analyzer
    features = analyzer.analyze_question(text)
    
    # Calculate composite scores
    syntactic_score = (
        features['max_dependency_depth'] * 0.3 +
        features['unique_dependency_types'] * 0.3 +
        features['clause_count'] * 0.4
    )
    
    semantic_score = (
        features['lexical_diversity'] * 0.4 +
        features['entity_count'] * 0.3 +
        features['content_word_ratio'] * 0.3
    )
    
    tom_score = (
        features['mental_state_verbs'] * 0.4 +
        features['perspective_markers'] * 0.3 +
        features['third_person_count'] * 0.3
    )
    
    reasoning_score = (
        features['logical_operators'] * 0.3 +
        features['temporal_references'] * 0.3 +
        features['conditional_count'] * 0.4
    )
    
    # Overall complexity score (normalized)
    overall_score = (syntactic_score + semantic_score + tom_score + reasoning_score) / 4
    
    return pd.Series({
        'Question_Complexity_Score': round(overall_score, 3),
        'Syntactic_Complexity': round(syntactic_score, 3),
        'Semantic_Complexity': round(semantic_score, 3),
        'ToM_Complexity': round(tom_score, 3),
        'Reasoning_Complexity': round(reasoning_score, 3)
    })

# Load your dataset
df = pd.read_csv('../dataset.csv')

print("Analyzing question complexity...")
print(f"Processing {len(df)} questions...")

# Apply complexity analysis
complexity_results = df['QUESTION'].apply(calculate_question_complexity)

# Add results to dataframe
df = pd.concat([df, complexity_results], axis=1)

# Convert to appropriate types
for col in ['Question_Complexity_Score', 'Syntactic_Complexity', 'Semantic_Complexity', 
           'ToM_Complexity', 'Reasoning_Complexity']:
    df[col] = df[col].astype('float64')

# Reorder columns to place complexity metrics after QUESTION
question_index = df.columns.get_loc('QUESTION')
cols = df.columns.tolist()

# Remove complexity columns from their current positions
complexity_cols = ['Question_Complexity_Score', 'Syntactic_Complexity', 'Semantic_Complexity', 
                  'ToM_Complexity', 'Reasoning_Complexity']
for col in complexity_cols:
    cols.remove(col)

# Insert after QUESTION
for i, col in enumerate(complexity_cols):
    cols.insert(question_index + 1 + i, col)

df = df[cols]

# Save the updated dataset
df.to_csv('dataset_with_question_complexity.csv', index=False)

print("\nQuestion Complexity Analysis Complete!")
print(f"Results saved to: dataset_with_question_complexity.csv")

# Print summary statistics
print("\nComplexity Score Statistics:")
for col in complexity_cols:
    if df[col].notna().sum() > 0:
        print(f"{col}:")
        print(f"  Mean: {df[col].mean():.3f}")
        print(f"  Std:  {df[col].std():.3f}")
        print(f"  Min:  {df[col].min():.3f}")
        print(f"  Max:  {df[col].max():.3f}")
        print()

# %%
