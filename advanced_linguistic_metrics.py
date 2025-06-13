#!/usr/bin/env python3
"""
Advanced Linguistic Metrics for Theory of Mind Complexity Analysis
Implements syntactic parsing, semantic role labeling, and embedding-based metrics
"""

import pandas as pd
import numpy as np
import spacy
import warnings
warnings.filterwarnings('ignore')

try:
    import benepar
    BENEPAR_AVAILABLE = True
except ImportError:
    BENEPAR_AVAILABLE = False
    print("Warning: benepar not available. Constituency parsing disabled.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Embedding metrics disabled.")

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("Warning: textstat not available. Readability metrics disabled.")

from collections import defaultdict, Counter
import re
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class AdvancedLinguisticAnalyzer:
    """
    Advanced linguistic analyzer implementing syntactic, semantic, and embedding-based metrics
    """
    
    def __init__(self):
        """Initialize the analyzer with required models"""
        print("Initializing Advanced Linguistic Analyzer...")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy model loaded")
        except OSError:
            print("Warning: spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Load constituency parser if available
        if BENEPAR_AVAILABLE and self.nlp:
            try:
                benepar.download('benepar_en3')
                self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
                self.constituency_available = True
                print("✓ Constituency parser loaded")
            except:
                self.constituency_available = False
                print("Warning: Constituency parser failed to load")
        else:
            self.constituency_available = False
        
        # Load sentence transformer if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ Sentence transformer loaded")
            except:
                self.sentence_model = None
                print("Warning: Sentence transformer failed to load")
        else:
            self.sentence_model = None
        
        print("✓ Advanced Linguistic Analyzer initialized")
    
    def analyze_syntactic_complexity(self, text, text_type='Story'):
        """Analyze syntactic complexity using dependency and constituency parsing"""
        features = {}
        
        if not text or str(text).strip() == '' or str(text).lower() == 'nan':
            return self._get_empty_syntactic_features(text_type)
        
        text = str(text).strip()
        
        if not self.nlp:
            return self._get_empty_syntactic_features(text_type)
        
        try:
            doc = self.nlp(text)
            
            # 1. Dependency Parse Depth Analysis
            max_depth, avg_depth, total_depth = self._calculate_dependency_depths(doc)
            
            # 2. Constituency Parse Depth (if available)
            constituency_depth = self._calculate_constituency_depth(doc)
            
            # 3. Syntactic Complexity Metrics
            clause_count = self._count_clauses(doc)
            subordinate_clauses = self._count_subordinate_clauses(doc)
            coordinate_clauses = self._count_coordinate_clauses(doc)
            
            # 4. Embedding Depth (Yngve's metric)
            yngve_depth = self._calculate_yngve_depth(doc)
            
            # 5. Dependency Arc Length
            avg_arc_length, max_arc_length = self._calculate_arc_lengths(doc)
            
            features.update({
                f'{text_type}_Parse_Max_Depth': max_depth,
                f'{text_type}_Parse_Avg_Depth': avg_depth,
                f'{text_type}_Parse_Total_Depth': total_depth,
                f'{text_type}_Constituency_Depth': constituency_depth,
                f'{text_type}_Clause_Count': clause_count,
                f'{text_type}_Subordinate_Clauses': subordinate_clauses,
                f'{text_type}_Coordinate_Clauses': coordinate_clauses,
                f'{text_type}_Yngve_Depth': yngve_depth,
                f'{text_type}_Avg_Arc_Length': avg_arc_length,
                f'{text_type}_Max_Arc_Length': max_arc_length
            })
            
        except Exception as e:
            print(f"Error in syntactic analysis: {e}")
            return self._get_empty_syntactic_features(text_type)
        
        return features
    
    def analyze_semantic_complexity(self, text, text_type='Story'):
        """Analyze semantic complexity using role labeling and graph analysis"""
        features = {}
        
        if not text or str(text).strip() == '' or str(text).lower() == 'nan':
            return self._get_empty_semantic_features(text_type)
        
        text = str(text).strip()
        
        if not self.nlp:
            return self._get_empty_semantic_features(text_type)
        
        try:
            doc = self.nlp(text)
            
            # 1. Semantic Role Analysis (approximated)
            verb_argument_complexity = self._analyze_verb_arguments(doc)
            predicate_count = self._count_predicates(doc)
            
            # 2. Entity-Relation Graph Complexity
            entity_relation_depth = self._analyze_entity_relations(doc)
            
            # 3. Mental State Semantic Depth
            mental_state_depth = self._analyze_mental_state_semantics(doc)
            
            # 4. Causal Chain Analysis
            causal_chain_depth = self._analyze_causal_chains(doc)
            
            # 5. Temporal Semantic Complexity
            temporal_complexity = self._analyze_temporal_semantics(doc)
            
            features.update({
                f'{text_type}_Verb_Argument_Complexity': verb_argument_complexity,
                f'{text_type}_Predicate_Count': predicate_count,
                f'{text_type}_Entity_Relation_Depth': entity_relation_depth,
                f'{text_type}_Semantic_Mental_State_Depth': mental_state_depth,
                f'{text_type}_Causal_Chain_Depth': causal_chain_depth,
                f'{text_type}_Temporal_Semantic_Complexity': temporal_complexity
            })
            
        except Exception as e:
            print(f"Error in semantic analysis: {e}")
            return self._get_empty_semantic_features(text_type)
        
        return features
    
    def analyze_embedding_complexity(self, text, text_type='Story'):
        """Analyze complexity using embedding space representations"""
        features = {}
        
        if not text or str(text).strip() == '' or str(text).lower() == 'nan':
            return self._get_empty_embedding_features(text_type)
        
        text = str(text).strip()
        
        if not self.sentence_model:
            return self._get_empty_embedding_features(text_type)
        
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return self._get_empty_embedding_features(text_type)
            
            # Get sentence embeddings
            embeddings = self.sentence_model.encode(sentences)
            
            # 1. Embedding Dispersion
            embedding_variance = np.var(embeddings, axis=0).mean()
            embedding_std = np.std(embeddings, axis=0).mean()
            
            # 2. Semantic Coherence
            coherence_scores = []
            for i in range(len(embeddings) - 1):
                similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                coherence_scores.append(similarity)
            
            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0
            min_coherence = np.min(coherence_scores) if coherence_scores else 0
            
            # 3. Embedding Trajectory Length
            trajectory_length = 0
            for i in range(len(embeddings) - 1):
                distance = np.linalg.norm(embeddings[i] - embeddings[i + 1])
                trajectory_length += distance
            
            # 4. PCA Analysis
            if len(embeddings) > 2:
                pca = PCA(n_components=min(3, len(embeddings)))
                pca.fit(embeddings)
                explained_variance_ratio = pca.explained_variance_ratio_[0]
                cumulative_variance = np.sum(pca.explained_variance_ratio_[:2])
            else:
                explained_variance_ratio = 0
                cumulative_variance = 0
            
            # 5. Embedding Clustering
            centroid = np.mean(embeddings, axis=0)
            distances_to_centroid = [np.linalg.norm(emb - centroid) for emb in embeddings]
            avg_distance_to_centroid = np.mean(distances_to_centroid)
            max_distance_to_centroid = np.max(distances_to_centroid)
            
            features.update({
                f'{text_type}_Embedding_Variance': embedding_variance,
                f'{text_type}_Embedding_Std': embedding_std,
                f'{text_type}_Semantic_Coherence_Avg': avg_coherence,
                f'{text_type}_Semantic_Coherence_Min': min_coherence,
                f'{text_type}_Embedding_Trajectory_Length': trajectory_length,
                f'{text_type}_PCA_First_Component': explained_variance_ratio,
                f'{text_type}_PCA_Cumulative_Variance': cumulative_variance,
                f'{text_type}_Avg_Distance_To_Centroid': avg_distance_to_centroid,
                f'{text_type}_Max_Distance_To_Centroid': max_distance_to_centroid
            })
            
        except Exception as e:
            print(f"Error in embedding analysis: {e}")
            return self._get_empty_embedding_features(text_type)
        
        return features
    
    def analyze_readability_complexity(self, text, text_type='Story'):
        """Analyze readability and surface complexity metrics"""
        features = {}
        
        if not text or str(text).strip() == '' or str(text).lower() == 'nan':
            return self._get_empty_readability_features(text_type)
        
        text = str(text).strip()
        
        if not TEXTSTAT_AVAILABLE:
            return self._get_empty_readability_features(text_type)
        
        try:
            # Basic readability metrics
            flesch_ease = textstat.flesch_reading_ease(text)
            flesch_grade = textstat.flesch_kincaid_grade(text)
            gunning_fog = textstat.gunning_fog(text)
            smog_index = textstat.smog_index(text)
            automated_readability = textstat.automated_readability_index(text)
            coleman_liau = textstat.coleman_liau_index(text)
            
            # Advanced metrics
            difficult_words = textstat.difficult_words(text)
            dale_chall = textstat.dale_chall_readability_score(text)
            linsear_write = textstat.linsear_write_formula(text)
            
            features.update({
                f'{text_type}_Flesch_Reading_Ease': flesch_ease,
                f'{text_type}_Flesch_Kincaid_Grade': flesch_grade,
                f'{text_type}_Gunning_Fog': gunning_fog,
                f'{text_type}_SMOG_Index': smog_index,
                f'{text_type}_Automated_Readability': automated_readability,
                f'{text_type}_Coleman_Liau': coleman_liau,
                f'{text_type}_Difficult_Words': difficult_words,
                f'{text_type}_Dale_Chall': dale_chall,
                f'{text_type}_Linsear_Write': linsear_write
            })
            
        except Exception as e:
            print(f"Error in readability analysis: {e}")
            return self._get_empty_readability_features(text_type)
        
        return features
    
    def _calculate_dependency_depths(self, doc):
        """Calculate dependency parse depths"""
        depths = []
        
        for token in doc:
            depth = 0
            current = token
            visited = set()
            
            while current.head != current and current not in visited:
                visited.add(current)
                current = current.head
                depth += 1
                if depth > 50:  # Prevent infinite loops
                    break
            
            depths.append(depth)
        
        if depths:
            return max(depths), np.mean(depths), sum(depths)
        else:
            return 0, 0, 0
    
    def _calculate_constituency_depth(self, doc):
        """Calculate constituency parse depth if available"""
        if not self.constituency_available:
            return 0
        
        try:
            max_depth = 0
            for sent in doc.sents:
                if hasattr(sent._, 'parse_tree'):
                    tree = sent._.parse_tree
                    depth = self._tree_depth(tree)
                    max_depth = max(max_depth, depth)
            return max_depth
        except:
            return 0
    
    def _tree_depth(self, tree):
        """Calculate depth of a parse tree"""
        if not hasattr(tree, '__iter__'):
            return 0
        return 1 + max([self._tree_depth(child) for child in tree] + [0])
    
    def _count_clauses(self, doc):
        """Count total clauses"""
        clause_markers = ['ccomp', 'xcomp', 'advcl', 'acl', 'relcl']
        return sum(1 for token in doc if token.dep_ in clause_markers)
    
    def _count_subordinate_clauses(self, doc):
        """Count subordinate clauses"""
        subordinate_markers = ['advcl', 'ccomp', 'xcomp']
        return sum(1 for token in doc if token.dep_ in subordinate_markers)
    
    def _count_coordinate_clauses(self, doc):
        """Count coordinate clauses"""
        return sum(1 for token in doc if token.dep_ == 'conj' and token.pos_ == 'VERB')
    
    def _calculate_yngve_depth(self, doc):
        """Calculate Yngve's depth metric"""
        # Simplified approximation
        depths = []
        for token in doc:
            if token.dep_ in ['ccomp', 'xcomp', 'advcl']:
                depths.append(2)
            elif token.dep_ in ['acl', 'relcl']:
                depths.append(1)
        
        return sum(depths) / len(doc) if len(doc) > 0 else 0
    
    def _calculate_arc_lengths(self, doc):
        """Calculate dependency arc lengths"""
        arc_lengths = []
        for token in doc:
            if token.head != token:
                arc_length = abs(token.i - token.head.i)
                arc_lengths.append(arc_length)
        
        if arc_lengths:
            return np.mean(arc_lengths), max(arc_lengths)
        else:
            return 0, 0
    
    def _analyze_verb_arguments(self, doc):
        """Analyze verb argument complexity"""
        verb_complexities = []
        
        for token in doc:
            if token.pos_ == 'VERB':
                # Count arguments (subjects, objects, etc.)
                args = [child for child in token.children 
                       if child.dep_ in ['nsubj', 'dobj', 'iobj', 'pobj', 'agent']]
                verb_complexities.append(len(args))
        
        return np.mean(verb_complexities) if verb_complexities else 0
    
    def _count_predicates(self, doc):
        """Count predicates (verbs and adjectives)"""
        return sum(1 for token in doc if token.pos_ in ['VERB', 'ADJ'])
    
    def _analyze_entity_relations(self, doc):
        """Analyze entity-relation graph complexity"""
        entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]
        
        if len(entities) < 2:
            return 0
        
        # Simple approximation: entity co-occurrence complexity
        unique_entities = len(set(entities))
        total_entities = len(entities)
        
        return unique_entities * total_entities / len(doc) if len(doc) > 0 else 0
    
    def _analyze_mental_state_semantics(self, doc):
        """Analyze mental state semantic depth"""
        mental_verbs = ['think', 'believe', 'know', 'feel', 'want', 'hope', 'fear', 'doubt']
        
        mental_state_count = 0
        nesting_depth = 0
        
        for token in doc:
            if token.lemma_.lower() in mental_verbs:
                mental_state_count += 1
                # Check for nested mental states
                for child in token.subtree:
                    if child.lemma_.lower() in mental_verbs and child != token:
                        nesting_depth += 1
        
        return mental_state_count + nesting_depth * 2
    
    def _analyze_causal_chains(self, doc):
        """Analyze causal chain depth"""
        causal_markers = ['because', 'since', 'due', 'therefore', 'thus', 'so', 'hence']
        
        causal_count = 0
        for token in doc:
            if token.lemma_.lower() in causal_markers:
                causal_count += 1
        
        return causal_count
    
    def _analyze_temporal_semantics(self, doc):
        """Analyze temporal semantic complexity"""
        temporal_markers = ['before', 'after', 'when', 'while', 'during', 'then', 'now', 'later']
        
        temporal_count = 0
        for token in doc:
            if token.lemma_.lower() in temporal_markers:
                temporal_count += 1
        
        return temporal_count
    
    def _get_empty_syntactic_features(self, text_type):
        """Return empty syntactic features"""
        return {
            f'{text_type}_Parse_Max_Depth': 0,
            f'{text_type}_Parse_Avg_Depth': 0,
            f'{text_type}_Parse_Total_Depth': 0,
            f'{text_type}_Constituency_Depth': 0,
            f'{text_type}_Clause_Count': 0,
            f'{text_type}_Subordinate_Clauses': 0,
            f'{text_type}_Coordinate_Clauses': 0,
            f'{text_type}_Yngve_Depth': 0,
            f'{text_type}_Avg_Arc_Length': 0,
            f'{text_type}_Max_Arc_Length': 0
        }
    
    def _get_empty_semantic_features(self, text_type):
        """Return empty semantic features"""
        return {
            f'{text_type}_Verb_Argument_Complexity': 0,
            f'{text_type}_Predicate_Count': 0,
            f'{text_type}_Entity_Relation_Depth': 0,
            f'{text_type}_Semantic_Mental_State_Depth': 0,
            f'{text_type}_Causal_Chain_Depth': 0,
            f'{text_type}_Temporal_Semantic_Complexity': 0
        }
    
    def _get_empty_embedding_features(self, text_type):
        """Return empty embedding features"""
        return {
            f'{text_type}_Embedding_Variance': 0,
            f'{text_type}_Embedding_Std': 0,
            f'{text_type}_Semantic_Coherence_Avg': 0,
            f'{text_type}_Semantic_Coherence_Min': 0,
            f'{text_type}_Embedding_Trajectory_Length': 0,
            f'{text_type}_PCA_First_Component': 0,
            f'{text_type}_PCA_Cumulative_Variance': 0,
            f'{text_type}_Avg_Distance_To_Centroid': 0,
            f'{text_type}_Max_Distance_To_Centroid': 0
        }
    
    def _get_empty_readability_features(self, text_type):
        """Return empty readability features"""
        return {
            f'{text_type}_Flesch_Reading_Ease': 0,
            f'{text_type}_Flesch_Kincaid_Grade': 0,
            f'{text_type}_Gunning_Fog': 0,
            f'{text_type}_SMOG_Index': 0,
            f'{text_type}_Automated_Readability': 0,
            f'{text_type}_Coleman_Liau': 0,
            f'{text_type}_Difficult_Words': 0,
            f'{text_type}_Dale_Chall': 0,
            f'{text_type}_Linsear_Write': 0
        }
    
    def analyze_comprehensive(self, text, text_type='Story'):
        """Comprehensive analysis combining all advanced metrics"""
        features = {}
        
        # Syntactic complexity
        features.update(self.analyze_syntactic_complexity(text, text_type))
        
        # Semantic complexity
        features.update(self.analyze_semantic_complexity(text, text_type))
        
        # Embedding complexity
        features.update(self.analyze_embedding_complexity(text, text_type))
        
        # Readability complexity
        features.update(self.analyze_readability_complexity(text, text_type))
        
        return features

def analyze_text_advanced(text, text_type='Story'):
    """Wrapper function for advanced linguistic analysis"""
    if not hasattr(analyze_text_advanced, 'analyzer'):
        analyze_text_advanced.analyzer = AdvancedLinguisticAnalyzer()
    
    analyzer = analyze_text_advanced.analyzer
    return analyzer.analyze_comprehensive(text, text_type)

if __name__ == "__main__":
    # Test the analyzer
    test_text = """John thinks that Mary believes Tom knows she is lying about the surprise party. 
    However, Tom actually has no idea what Mary thinks, and he's completely confused by her behavior. 
    Meanwhile, Sarah wonders if John realizes that his plan might backfire, but she's not sure 
    whether she should tell him what she suspects."""
    
    analyzer = AdvancedLinguisticAnalyzer()
    features = analyzer.analyze_comprehensive(test_text, 'Test')
    
    print("Advanced Linguistic Analysis Results:")
    for key, value in sorted(features.items()):
        print(f"  {key}: {value:.4f}")
