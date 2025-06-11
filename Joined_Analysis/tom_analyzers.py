"""
Consolidated Theory of Mind (ToM) Analysis Module

This module contains all the analyzer classes for comprehensive ToM analysis:
- Idea Density Analysis (DEPID)
- RST Discourse Analysis
- Question Complexity Analysis
- Answer Distinctiveness Analysis
"""

import pandas as pd
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

# Import the ideadensity package
from ideadensity import depid

# Import RST parser
from isanlp_rst.parser import Parser


class IdeaDensityAnalyzer:
    """
    Analyzer for Dependency-based Propositional Idea Density (DEPID).
    
    Measures the density of propositions in text based on syntactic dependencies.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        print("✓ Idea Density Analyzer initialized")
    
    def analyze(self, text):
        """Calculate idea density metrics for a single text."""
        if pd.isna(text) or text.strip() == '':
            return {
                'Idea_Density': None,
                'Word_Count': None
            }
        
        density, word_count, dependencies = depid(text)
        return {
            'Idea_Density': round(density, 3),
            'Word_Count': int(word_count)
        }


class RSTAnalyzer:
    """
    Analyzer for Rhetorical Structure Theory (RST) features.
    
    Extracts discourse structure features relevant for Theory of Mind analysis.
    """
    
    def __init__(self, version='gumrrg', cuda_device=-1):
        """Initialize RST parser."""
        print("Loading RST parser...")
        self.parser = Parser(
            hf_model_name='tchewik/isanlp_rst_v3', 
            hf_model_version=version, 
            cuda_device=cuda_device
        )
        print("✓ RST Analyzer initialized")
    
    def analyze(self, text):
        """Extract RST features from a single text."""
        try:
            result = self.parser(text)
            
            if not result or 'rst' not in result or not result['rst']:
                return self._get_empty_features()
            
            rst_tree = result['rst'][0]
            
            features = {
                'RST_EDUs': self._count_edus(rst_tree),
                'RST_Tree_Depth': self._calculate_depth(rst_tree),
            }
            
            # Extract relation type counts
            relation_counts = self._count_relation_types(rst_tree)
            features.update(relation_counts)
            
            return features
            
        except:
            return self._get_empty_features()
    
    def _count_edus(self, rst_tree):
        """Count Elementary Discourse Units."""
        edu_count_ref = [0]
        self._count_elementary_nodes(rst_tree, edu_count_ref)
        return edu_count_ref[0] if edu_count_ref[0] > 0 else 1
    
    def _count_elementary_nodes(self, node, count_ref):
        """Count nodes with 'elementary' relation (EDUs)."""
        if node is None:
            return
        
        if hasattr(node, 'relation') and node.relation.lower() == 'elementary':
            count_ref[0] += 1
        
        if hasattr(node, 'left') and node.left:
            self._count_elementary_nodes(node.left, count_ref)
        
        if hasattr(node, 'right') and node.right:
            self._count_elementary_nodes(node.right, count_ref)
    
    def _calculate_depth(self, node, current_depth=0):
        """Calculate maximum depth of the RST tree."""
        if not hasattr(node, 'left') and not hasattr(node, 'right'):
            return current_depth
        
        max_depth = current_depth
        
        if hasattr(node, 'left') and node.left:
            max_depth = max(max_depth, self._calculate_depth(node.left, current_depth + 1))
        
        if hasattr(node, 'right') and node.right:
            max_depth = max(max_depth, self._calculate_depth(node.right, current_depth + 1))
        
        return max_depth
    
    def _count_relation_types(self, node):
        """Count occurrences of ToM-relevant relation types."""
        relation_types = ['attribution', 'causal', 'explanation']
        counts = {f'RST_{rel}': 0 for rel in relation_types}
        self._traverse_relations(node, counts)
        return counts
    
    def _traverse_relations(self, node, counts):
        """Traverse tree and count relations."""
        if hasattr(node, 'relation'):
            rel_type = node.relation.lower().replace('-', '_').replace(' ', '_')
            
            if rel_type != 'elementary':
                key = f'RST_{rel_type}'
                if key in counts:
                    counts[key] += 1
        
        if hasattr(node, 'left') and node.left:
            self._traverse_relations(node.left, counts)
        
        if hasattr(node, 'right') and node.right:
            self._traverse_relations(node.right, counts)
    
    def _get_empty_features(self):
        """Return empty feature dict when parsing fails."""
        return {
            'RST_EDUs': 0,
            'RST_Tree_Depth': 0,
            'RST_attribution': 0,
            'RST_causal': 0,
            'RST_explanation': 0
        }


class QuestionComplexityAnalyzer:
    """
    Analyzer for question complexity across multiple dimensions.
    
    Measures syntactic, semantic, ToM-specific, and reasoning complexity.
    """
    
    def __init__(self):
        """Initialize the analyzer with required models."""
        print("Loading Question Complexity models...")
        
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
        
        print("✓ Question Complexity Analyzer initialized")
    
    def analyze(self, question):
        """Analyze a single question for complexity metrics."""
        if pd.isna(question) or question.strip() == '':
            return self._get_empty_features()
        
        question = question.strip()
        doc = self.nlp(question)
        
        features = {}
        features.update(self._basic_metrics(question, doc))
        features.update(self._syntactic_complexity(doc))
        features.update(self._semantic_complexity(question, doc))
        features.update(self._tom_complexity(question, doc))
        features.update(self._reasoning_complexity(question, doc))
        
        # Calculate composite scores
        features.update(self._calculate_composite_scores(features))
        
        return features
    
    def _basic_metrics(self, question, doc):
        """Calculate basic question metrics."""
        return {
            'Q_Length': len(question),
            'Q_Word_Count': len([token for token in doc if not token.is_space]),
            'Q_Sentence_Count': len(list(doc.sents)),
            'Q_Avg_Word_Length': np.mean([len(token.text) for token in doc if token.is_alpha])
        }
    
    def _syntactic_complexity(self, doc):
        """Calculate syntactic complexity metrics."""
        def get_depth(token, depth=0):
            if not list(token.children):
                return depth
            return max(get_depth(child, depth + 1) for child in token.children)
        
        depths = [get_depth(sent.root) for sent in doc.sents]
        max_depth = max(depths) if depths else 0
        avg_depth = np.mean(depths) if depths else 0
        
        dep_types = set(token.dep_ for token in doc)
        clause_markers = ['SCONJ', 'PRON']
        clause_count = sum(1 for token in doc if token.pos_ in clause_markers)
        
        return {
            'Q_Max_Dependency_Depth': max_depth,
            'Q_Avg_Dependency_Depth': avg_depth,
            'Q_Unique_Dependency_Types': len(dep_types),
            'Q_Clause_Count': clause_count
        }
    
    def _semantic_complexity(self, question, doc):
        """Calculate semantic complexity metrics."""
        words = [token.lemma_.lower() for token in doc if token.is_alpha]
        ttr = len(set(words)) / len(words) if words else 0
        
        entity_count = len(doc.ents)
        entity_types = len(set(ent.label_ for ent in doc.ents))
        
        content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
        content_words = sum(1 for token in doc if token.pos_ in content_pos)
        content_ratio = content_words / len(doc) if len(doc) > 0 else 0
        
        return {
            'Q_Lexical_Diversity': ttr,
            'Q_Entity_Count': entity_count,
            'Q_Entity_Type_Count': entity_types,
            'Q_Content_Word_Ratio': content_ratio
        }
    
    def _tom_complexity(self, question, doc):
        """Calculate Theory of Mind specific complexity."""
        question_lower = question.lower()
        
        mental_state_count = sum(1 for token in doc 
                                if token.lemma_.lower() in self.mental_state_verbs)
        
        perspective_count = sum(1 for marker in self.perspective_markers 
                              if marker in question_lower)
        
        person_pronouns = {
            'first': {'i', 'me', 'my', 'mine', 'myself'},
            'second': {'you', 'your', 'yours', 'yourself'},
            'third': {'he', 'she', 'him', 'her', 'his', 'hers', 'they', 'them', 'their'}
        }
        
        person_counts = {}
        for person, pronouns in person_pronouns.items():
            count = sum(1 for token in doc if token.lemma_.lower() in pronouns)
            person_counts[f'Q_{person.title()}_Person_Count'] = count
        
        question_words = {'what', 'who', 'when', 'where', 'why', 'how', 'which'}
        question_word_count = sum(1 for token in doc 
                                if token.lemma_.lower() in question_words)
        
        return {
            'Q_Mental_State_Verbs': mental_state_count,
            'Q_Perspective_Markers': perspective_count,
            'Q_Question_Word_Count': question_word_count,
            **person_counts
        }
    
    def _reasoning_complexity(self, question, doc):
        """Calculate reasoning complexity metrics."""
        question_lower = question.lower()
        
        logical_ops = ['and', 'or', 'but', 'if', 'then', 'because', 'since', 'although']
        logical_count = sum(1 for op in logical_ops if op in question_lower)
        
        temporal_words = ['before', 'after', 'when', 'while', 'during', 'then', 'now', 'later']
        temporal_count = sum(1 for word in temporal_words if word in question_lower)
        
        negation_count = sum(1 for token in doc if token.dep_ == 'neg')
        conditional_count = question_lower.count('if') + question_lower.count('would')
        
        return {
            'Q_Logical_Operators': logical_count,
            'Q_Temporal_References': temporal_count,
            'Q_Negation_Count': negation_count,
            'Q_Conditional_Count': conditional_count
        }
    
    def _calculate_composite_scores(self, features):
        """Calculate composite complexity scores."""
        syntactic_score = (
            features['Q_Max_Dependency_Depth'] * 0.3 +
            features['Q_Unique_Dependency_Types'] * 0.3 +
            features['Q_Clause_Count'] * 0.4
        )
        
        semantic_score = (
            features['Q_Lexical_Diversity'] * 0.4 +
            features['Q_Entity_Count'] * 0.3 +
            features['Q_Content_Word_Ratio'] * 0.3
        )
        
        tom_score = (
            features['Q_Mental_State_Verbs'] * 0.4 +
            features['Q_Perspective_Markers'] * 0.3 +
            features['Q_Third_Person_Count'] * 0.3
        )
        
        reasoning_score = (
            features['Q_Logical_Operators'] * 0.3 +
            features['Q_Temporal_References'] * 0.3 +
            features['Q_Conditional_Count'] * 0.4
        )
        
        overall_score = (syntactic_score + semantic_score + tom_score + reasoning_score) / 4
        
        return {
            'Question_Complexity_Score': round(overall_score, 3),
            'Q_Syntactic_Complexity': round(syntactic_score, 3),
            'Q_Semantic_Complexity': round(semantic_score, 3),
            'Q_ToM_Complexity': round(tom_score, 3),
            'Q_Reasoning_Complexity': round(reasoning_score, 3)
        }
    
    def _get_empty_features(self):
        """Return empty feature dict when analysis fails."""
        return {
            # Basic metrics
            'Q_Length': 0,
            'Q_Word_Count': 0,
            'Q_Sentence_Count': 0,
            'Q_Avg_Word_Length': 0,
            
            # Syntactic complexity
            'Q_Max_Dependency_Depth': 0,
            'Q_Avg_Dependency_Depth': 0,
            'Q_Unique_Dependency_Types': 0,
            'Q_Clause_Count': 0,
            
            # Semantic complexity
            'Q_Lexical_Diversity': 0,
            'Q_Entity_Count': 0,
            'Q_Entity_Type_Count': 0,
            'Q_Content_Word_Ratio': 0,
            
            # ToM complexity
            'Q_Mental_State_Verbs': 0,
            'Q_Perspective_Markers': 0,
            'Q_Question_Word_Count': 0,
            'Q_First_Person_Count': 0,
            'Q_Second_Person_Count': 0,
            'Q_Third_Person_Count': 0,
            
            # Reasoning complexity
            'Q_Logical_Operators': 0,
            'Q_Temporal_References': 0,
            'Q_Negation_Count': 0,
            'Q_Conditional_Count': 0,
            
            # Composite scores
            'Question_Complexity_Score': 0,
            'Q_Syntactic_Complexity': 0,
            'Q_Semantic_Complexity': 0,
            'Q_ToM_Complexity': 0,
            'Q_Reasoning_Complexity': 0
        }


class AnswerDistinctivenessAnalyzer:
    """
    Analyzer for answer distinctiveness across multiple choice options.
    
    Measures how distinct/unique answers are, which indicates question difficulty.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        print("Loading Answer Distinctiveness models...")
        
        # Load spaCy model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        
        print("✓ Answer Distinctiveness Analyzer initialized")
    
    def analyze(self, option_a, option_b, option_c, option_d):
        """Analyze distinctiveness across answer options."""
        options = [option_a, option_b, option_c, option_d]
        
        # Filter out NaN/empty options
        valid_options = [opt for opt in options if pd.notna(opt) and str(opt).strip()]
        
        if len(valid_options) < 2:
            return self._get_empty_features()
        
        features = {}
        features.update(self._semantic_distinctiveness(valid_options))
        features.update(self._lexical_distinctiveness(valid_options))
        features.update(self._length_distinctiveness(valid_options))
        features.update(self._syntactic_distinctiveness(valid_options))
        
        # Calculate overall distinctiveness score
        features['Answer_Distinctiveness_Score'] = self._calculate_overall_score(features)
        
        return features
    
    def _semantic_distinctiveness(self, options):
        """Calculate semantic distinctiveness using sentence embeddings."""
        try:
            embeddings = self.sentence_model.encode(options)
            similarities = cosine_similarity(embeddings)
            
            # Get upper triangle (excluding diagonal)
            n = len(similarities)
            upper_triangle = []
            for i in range(n):
                for j in range(i+1, n):
                    upper_triangle.append(similarities[i][j])
            
            avg_similarity = np.mean(upper_triangle) if upper_triangle else 0
            min_similarity = np.min(upper_triangle) if upper_triangle else 0
            max_similarity = np.max(upper_triangle) if upper_triangle else 0
            
            # Distinctiveness is inverse of similarity
            semantic_distinctiveness = 1 - avg_similarity
            
            return {
                'A_Semantic_Distinctiveness': round(semantic_distinctiveness, 3),
                'A_Avg_Semantic_Similarity': round(avg_similarity, 3),
                'A_Min_Semantic_Similarity': round(min_similarity, 3),
                'A_Max_Semantic_Similarity': round(max_similarity, 3)
            }
        except:
            return {
                'A_Semantic_Distinctiveness': 0,
                'A_Avg_Semantic_Similarity': 0,
                'A_Min_Semantic_Similarity': 0,
                'A_Max_Semantic_Similarity': 0
            }
    
    def _lexical_distinctiveness(self, options):
        """Calculate lexical distinctiveness using word overlap."""
        # Calculate word overlap between options
        word_sets = []
        for option in options:
            doc = self.nlp(str(option).lower())
            words = set(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)
            word_sets.append(words)
        
        # Calculate pairwise Jaccard similarities
        similarities = []
        n = len(word_sets)
        for i in range(n):
            for j in range(i+1, n):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                jaccard = intersection / union if union > 0 else 0
                similarities.append(jaccard)
        
        avg_jaccard = np.mean(similarities) if similarities else 0
        lexical_distinctiveness = 1 - avg_jaccard
        
        # Calculate vocabulary diversity
        all_words = set()
        total_words = 0
        for word_set in word_sets:
            all_words.update(word_set)
            total_words += len(word_set)
        
        vocab_diversity = len(all_words) / total_words if total_words > 0 else 0
        
        return {
            'A_Lexical_Distinctiveness': round(lexical_distinctiveness, 3),
            'A_Avg_Jaccard_Similarity': round(avg_jaccard, 3),
            'A_Vocabulary_Diversity': round(vocab_diversity, 3)
        }
    
    def _length_distinctiveness(self, options):
        """Calculate length-based distinctiveness."""
        lengths = [len(str(option)) for option in options]
        
        length_variance = np.var(lengths) if len(lengths) > 1 else 0
        length_range = max(lengths) - min(lengths) if lengths else 0
        avg_length = np.mean(lengths) if lengths else 0
        
        # Normalize by average length to get relative variance
        length_cv = np.std(lengths) / avg_length if avg_length > 0 else 0
        
        return {
            'A_Length_Distinctiveness': round(length_cv, 3),
            'A_Length_Variance': round(length_variance, 3),
            'A_Length_Range': length_range,
            'A_Avg_Length': round(avg_length, 1)
        }
    
    def _syntactic_distinctiveness(self, options):
        """Calculate syntactic distinctiveness using POS patterns."""
        pos_patterns = []
        
        for option in options:
            doc = self.nlp(str(option))
            pos_sequence = [token.pos_ for token in doc]
            pos_patterns.append(tuple(pos_sequence))
        
        # Count unique POS patterns
        unique_patterns = len(set(pos_patterns))
        total_patterns = len(pos_patterns)
        
        syntactic_diversity = unique_patterns / total_patterns if total_patterns > 0 else 0
        
        return {
            'A_Syntactic_Distinctiveness': round(syntactic_diversity, 3),
            'A_Unique_POS_Patterns': unique_patterns,
            'A_Total_Options': total_patterns
        }
    
    def _calculate_overall_score(self, features):
        """Calculate overall distinctiveness score."""
        semantic_weight = 0.4
        lexical_weight = 0.3
        length_weight = 0.2
        syntactic_weight = 0.1
        
        overall_score = (
            features['A_Semantic_Distinctiveness'] * semantic_weight +
            features['A_Lexical_Distinctiveness'] * lexical_weight +
            features['A_Length_Distinctiveness'] * length_weight +
            features['A_Syntactic_Distinctiveness'] * syntactic_weight
        )
        
        return round(overall_score, 3)
    
    def _get_empty_features(self):
        """Return empty feature dict when analysis fails."""
        return {
            'A_Semantic_Distinctiveness': 0,
            'A_Avg_Semantic_Similarity': 0,
            'A_Min_Semantic_Similarity': 0,
            'A_Max_Semantic_Similarity': 0,
            'A_Lexical_Distinctiveness': 0,
            'A_Avg_Jaccard_Similarity': 0,
            'A_Vocabulary_Diversity': 0,
            'A_Length_Distinctiveness': 0,
            'A_Length_Variance': 0,
            'A_Length_Range': 0,
            'A_Avg_Length': 0,
            'A_Syntactic_Distinctiveness': 0,
            'A_Unique_POS_Patterns': 0,
            'A_Total_Options': 0,
            'Answer_Distinctiveness_Score': 0
        }
