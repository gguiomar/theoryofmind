#!/usr/bin/env python3
"""
Advanced Theory of Mind (ToM) Metrics for deeper cognitive analysis
Focuses on syntactic complexity, perspective tracking, and recursive mental states
"""

import pandas as pd
import numpy as np
import spacy
import re
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class AdvancedToMAnalyzer:
    """
    Advanced analyzer for Theory of Mind complexity metrics.
    
    Focuses on:
    - Mental state embedding depth
    - Perspective shift complexity  
    - Recursive mental state chains
    - Coreference resolution complexity
    - False belief detection
    - Temporal mental state complexity
    """
    
    def __init__(self):
        """Initialize the analyzer with required models."""
        print("Loading spaCy model for advanced ToM analysis...")
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Warning: en_core_web_sm not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
        
        # Mental state verbs categorized by type
        self.mental_state_verbs = {
            'cognitive': {'think', 'believe', 'know', 'understand', 'realize', 'assume', 
                         'suppose', 'imagine', 'remember', 'forget', 'notice', 'recognize',
                         'doubt', 'wonder', 'guess', 'suspect', 'consider', 'conclude'},
            'emotional': {'feel', 'love', 'hate', 'fear', 'worry', 'hope', 'enjoy', 
                         'regret', 'admire', 'despise', 'like', 'dislike', 'appreciate'},
            'volitional': {'want', 'desire', 'wish', 'intend', 'plan', 'decide', 
                          'choose', 'prefer', 'aim', 'seek', 'avoid', 'refuse'}
        }
        
        # All mental state verbs combined
        self.all_ms_verbs = set()
        for category in self.mental_state_verbs.values():
            self.all_ms_verbs.update(category)
        
        # Perspective markers
        self.perspective_markers = {
            'he thinks', 'she believes', 'they know', 'he feels', 'she wants',
            'his opinion', 'her view', 'their perspective', 'from his point',
            'in her mind', 'he assumes', 'she expects', 'they suppose',
            'according to him', 'according to her', 'in his opinion', 'in her opinion'
        }
        
        # False belief indicators
        self.false_belief_patterns = [
            r'thinks?\s+.*\s+but\s+actually',
            r'believes?\s+.*\s+but\s+really',
            r'assumes?\s+.*\s+however',
            r'expects?\s+.*\s+but\s+instead',
            r'thought\s+.*\s+was\s+.*\s+but\s+it\s+was',
            r'believed\s+.*\s+contained\s+.*\s+but'
        ]
        
        # Temporal markers
        self.temporal_markers = {
            'before', 'after', 'when', 'while', 'during', 'then', 'now', 'later',
            'yesterday', 'today', 'tomorrow', 'previously', 'subsequently', 'meanwhile',
            'earlier', 'afterwards', 'recently', 'soon', 'eventually', 'finally'
        }
        
        print("✓ Advanced ToM analyzer initialized")
    
    def analyze_text(self, text, text_type='story'):
        """Analyze text for advanced ToM metrics."""
        if pd.isna(text) or text.strip() == '':
            return self._get_empty_features(text_type)
        
        # Clean and parse text
        text = text.strip()
        doc = self.nlp(text)
        
        features = {}
        
        # 1. Mental State Embedding Depth
        features.update(self._analyze_embedding_depth(doc, text_type))
        
        # 2. Perspective Shift Complexity
        features.update(self._analyze_perspective_shifts(doc, text, text_type))
        
        # 3. Recursive Mental State Depth
        features.update(self._analyze_recursive_mental_states(doc, text_type))
        
        # 4. Coreference Chain Analysis
        features.update(self._analyze_coreference_complexity(doc, text_type))
        
        # 5. False Belief Detection
        features.update(self._analyze_false_beliefs(text, text_type))
        
        # 6. Mental State Argument Complexity
        features.update(self._analyze_ms_argument_complexity(doc, text_type))
        
        # 7. Temporal Mental State Complexity
        features.update(self._analyze_temporal_ms_complexity(doc, text, text_type))
        
        return features
    
    def _analyze_embedding_depth(self, doc, text_type):
        """Analyze mental state embedding depth."""
        max_depth = 0
        avg_depth = 0
        ms_clause_count = 0
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_.lower() in self.all_ms_verbs:
                    depth = self._get_embedding_depth(token)
                    max_depth = max(max_depth, depth)
                    avg_depth += depth
                    ms_clause_count += 1
        
        avg_depth = avg_depth / ms_clause_count if ms_clause_count > 0 else 0
        
        return {
            f'{text_type}_MS_Embedding_Max_Depth': max_depth,
            f'{text_type}_MS_Embedding_Avg_Depth': round(avg_depth, 2),
            f'{text_type}_MS_Clause_Count': ms_clause_count
        }
    
    def _get_embedding_depth(self, token):
        """Calculate embedding depth for a mental state verb."""
        depth = 0
        current = token
        
        # Traverse up the dependency tree counting clausal embeddings
        while current.head != current:
            if current.dep_ in ['ccomp', 'xcomp', 'acl', 'advcl']:
                depth += 1
            current = current.head
        
        return depth
    
    def _analyze_perspective_shifts(self, doc, text, text_type):
        """Analyze perspective shift complexity."""
        text_lower = text.lower()
        
        # Count explicit perspective markers
        perspective_marker_count = sum(1 for marker in self.perspective_markers 
                                     if marker in text_lower)
        
        # Count subject changes in mental state contexts
        subject_changes = 0
        prev_subject = None
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_.lower() in self.all_ms_verbs:
                    # Find the subject of this mental state verb
                    subject = self._find_subject(token)
                    if subject and prev_subject and subject != prev_subject:
                        subject_changes += 1
                    prev_subject = subject
        
        return {
            f'{text_type}_Perspective_Markers': perspective_marker_count,
            f'{text_type}_Subject_Changes': subject_changes,
            f'{text_type}_Perspective_Complexity': perspective_marker_count + subject_changes
        }
    
    def _find_subject(self, verb_token):
        """Find the subject of a verb token."""
        for child in verb_token.children:
            if child.dep_ in ['nsubj', 'nsubjpass']:
                return child.lemma_.lower()
        return None
    
    def _analyze_recursive_mental_states(self, doc, text_type):
        """Analyze recursive mental state depth (A thinks B believes C knows...)."""
        max_recursive_depth = 0
        total_recursive_chains = 0
        
        for sent in doc.sents:
            recursive_depth = self._find_recursive_chain_depth(sent)
            if recursive_depth > 1:
                max_recursive_depth = max(max_recursive_depth, recursive_depth)
                total_recursive_chains += 1
        
        return {
            f'{text_type}_Recursive_MS_Max_Depth': max_recursive_depth,
            f'{text_type}_Recursive_MS_Chains': total_recursive_chains
        }
    
    def _find_recursive_chain_depth(self, sent):
        """Find the depth of recursive mental state chains in a sentence."""
        ms_verbs_in_sent = []
        
        for token in sent:
            if token.lemma_.lower() in self.all_ms_verbs:
                ms_verbs_in_sent.append(token)
        
        if len(ms_verbs_in_sent) <= 1:
            return len(ms_verbs_in_sent)
        
        # Check if mental state verbs are in a recursive chain
        # (one is in the complement of another)
        max_chain = 1
        for i, verb1 in enumerate(ms_verbs_in_sent):
            chain_length = 1
            current_verb = verb1
            
            # Look for mental state verbs in the complement
            for j, verb2 in enumerate(ms_verbs_in_sent[i+1:], i+1):
                if self._is_in_complement(current_verb, verb2):
                    chain_length += 1
                    current_verb = verb2
            
            max_chain = max(max_chain, chain_length)
        
        return max_chain
    
    def _is_in_complement(self, verb1, verb2):
        """Check if verb2 is in the complement of verb1."""
        # Simple heuristic: verb2 is a descendant of verb1 in dependency tree
        current = verb2
        while current.head != current:
            if current.head == verb1:
                return True
            current = current.head
        return False
    
    def _analyze_coreference_complexity(self, doc, text_type):
        """Analyze coreference chain complexity."""
        # Simple pronoun-based coreference analysis
        pronouns = {'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'their'}
        
        pronoun_count = 0
        entities = set()
        
        for token in doc:
            if token.lemma_.lower() in pronouns:
                pronoun_count += 1
            elif token.ent_type_ in ['PERSON', 'ORG']:
                entities.add(token.lemma_.lower())
        
        # Estimate coreference complexity
        coref_complexity = pronoun_count / max(len(entities), 1) if entities else 0
        
        return {
            f'{text_type}_Pronoun_Count': pronoun_count,
            f'{text_type}_Entity_Count': len(entities),
            f'{text_type}_Coref_Complexity': round(coref_complexity, 2)
        }
    
    def _analyze_false_beliefs(self, text, text_type):
        """Analyze false belief indicators."""
        text_lower = text.lower()
        false_belief_count = 0
        
        for pattern in self.false_belief_patterns:
            matches = re.findall(pattern, text_lower)
            false_belief_count += len(matches)
        
        # Additional simple patterns
        contradiction_words = ['but', 'however', 'actually', 'really', 'instead', 'although']
        ms_contradiction_count = 0
        
        sentences = text.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            has_ms_verb = any(verb in sentence_lower for verb in self.all_ms_verbs)
            has_contradiction = any(word in sentence_lower for word in contradiction_words)
            
            if has_ms_verb and has_contradiction:
                ms_contradiction_count += 1
        
        return {
            f'{text_type}_False_Belief_Patterns': false_belief_count,
            f'{text_type}_MS_Contradictions': ms_contradiction_count,
            f'{text_type}_False_Belief_Score': false_belief_count + ms_contradiction_count
        }
    
    def _analyze_ms_argument_complexity(self, doc, text_type):
        """Analyze mental state argument complexity."""
        total_ms_args = 0
        complex_ms_args = 0
        
        for token in doc:
            if token.lemma_.lower() in self.all_ms_verbs:
                # Count arguments (children that are not auxiliary)
                args = [child for child in token.children 
                       if child.dep_ in ['nsubj', 'dobj', 'iobj', 'ccomp', 'xcomp']]
                
                arg_count = len(args)
                total_ms_args += arg_count
                
                # Complex if more than 2 arguments or has clausal complement
                if arg_count > 2 or any(child.dep_ in ['ccomp', 'xcomp'] for child in token.children):
                    complex_ms_args += 1
        
        return {
            f'{text_type}_MS_Total_Args': total_ms_args,
            f'{text_type}_MS_Complex_Args': complex_ms_args,
            f'{text_type}_MS_Arg_Complexity': round(total_ms_args / max(len([t for t in doc if t.lemma_.lower() in self.all_ms_verbs]), 1), 2)
        }
    
    def _analyze_temporal_ms_complexity(self, doc, text, text_type):
        """Analyze temporal mental state complexity."""
        text_lower = text.lower()
        
        # Count temporal markers
        temporal_count = sum(1 for marker in self.temporal_markers if marker in text_lower)
        
        # Count mental state verbs
        ms_verb_count = sum(1 for token in doc if token.lemma_.lower() in self.all_ms_verbs)
        
        # Count tense changes (simplified)
        past_tense_count = sum(1 for token in doc if token.tag_ in ['VBD', 'VBN'])
        present_tense_count = sum(1 for token in doc if token.tag_ in ['VBZ', 'VBP'])
        future_tense_count = text_lower.count('will') + text_lower.count('would')
        
        tense_variety = sum([past_tense_count > 0, present_tense_count > 0, future_tense_count > 0])
        
        # Calculate temporal complexity
        temporal_ms_complexity = temporal_count * ms_verb_count * tense_variety
        
        return {
            f'{text_type}_Temporal_Markers': temporal_count,
            f'{text_type}_Tense_Variety': tense_variety,
            f'{text_type}_Temporal_MS_Complexity': temporal_ms_complexity
        }
    
    def _get_empty_features(self, text_type):
        """Return empty feature dict when analysis fails."""
        return {
            f'{text_type}_MS_Embedding_Max_Depth': 0,
            f'{text_type}_MS_Embedding_Avg_Depth': 0,
            f'{text_type}_MS_Clause_Count': 0,
            f'{text_type}_Perspective_Markers': 0,
            f'{text_type}_Subject_Changes': 0,
            f'{text_type}_Perspective_Complexity': 0,
            f'{text_type}_Recursive_MS_Max_Depth': 0,
            f'{text_type}_Recursive_MS_Chains': 0,
            f'{text_type}_Pronoun_Count': 0,
            f'{text_type}_Entity_Count': 0,
            f'{text_type}_Coref_Complexity': 0,
            f'{text_type}_False_Belief_Patterns': 0,
            f'{text_type}_MS_Contradictions': 0,
            f'{text_type}_False_Belief_Score': 0,
            f'{text_type}_MS_Total_Args': 0,
            f'{text_type}_MS_Complex_Args': 0,
            f'{text_type}_MS_Arg_Complexity': 0,
            f'{text_type}_Temporal_Markers': 0,
            f'{text_type}_Tense_Variety': 0,
            f'{text_type}_Temporal_MS_Complexity': 0
        }

def analyze_text_for_advanced_tom(text, text_type='Story'):
    """Wrapper function for analyzing a single text."""
    if not hasattr(analyze_text_for_advanced_tom, 'analyzer'):
        analyze_text_for_advanced_tom.analyzer = AdvancedToMAnalyzer()
    
    analyzer = analyze_text_for_advanced_tom.analyzer
    return analyzer.analyze_text(text, text_type)

if __name__ == "__main__":
    # Test the analyzer
    test_text = "John thinks that Mary believes Tom knows she is lying, but actually Tom has no idea what Mary thinks."
    
    analyzer = AdvancedToMAnalyzer()
    features = analyzer.analyze_text(test_text, 'Test')
    
    print("Test Analysis Results:")
    for key, value in features.items():
        print(f"  {key}: {value}")
