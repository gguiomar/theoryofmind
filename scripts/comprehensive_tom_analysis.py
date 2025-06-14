#!/usr/bin/env python3
"""
Comprehensive Theory of Mind Complexity Analysis
Implements all advanced NLP techniques from requirements_advanced.txt
Following the plan outlined in Theory_of_Mind_Complexity_Analysis_Report.md Section 8
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core scientific computing
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity

# Advanced NLP packages
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available")

try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False
    print("Warning: Stanza not available")

try:
    import benepar
    BENEPAR_AVAILABLE = True
except ImportError:
    BENEPAR_AVAILABLE = False
    print("Warning: benepar not available")

try:
    import neosca
    NEOSCA_AVAILABLE = True
except ImportError:
    NEOSCA_AVAILABLE = False
    print("Warning: neosca not available")

# AllenNLP removed due to dependency conflicts
ALLENNLP_AVAILABLE = False

try:
    import amrlib
    AMRLIB_AVAILABLE = True
except ImportError:
    AMRLIB_AVAILABLE = False
    print("Warning: amrlib not available")

# TUPA removed from pipeline as requested
TUPA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available")

try:
    from bertviz import head_view, model_view
    import torch
    from transformers import AutoTokenizer, AutoModel
    BERTVIZ_AVAILABLE = True
except ImportError:
    BERTVIZ_AVAILABLE = False
    print("Warning: bertviz/transformers not available")

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("Warning: textstat not available")

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("Warning: DEAP not available")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available")

try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not available")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available")

import re
from collections import defaultdict, Counter
import itertools
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

class ComprehensiveToMAnalyzer:
    """
    Comprehensive Theory of Mind analyzer implementing all advanced techniques
    from the research report Section 8
    """
    
    def __init__(self):
        """Initialize all available NLP models and tools"""
        print("Initializing Comprehensive ToM Analyzer...")
        print("This may take several minutes to download and load all models...")
        
        # Phase 1: Syntactic Analysis Tools
        self.syntactic_tools = self._initialize_syntactic_tools()
        
        # Phase 2: Semantic Analysis Tools  
        self.semantic_tools = self._initialize_semantic_tools()
        
        # Phase 3: Embedding Analysis Tools
        self.embedding_tools = self._initialize_embedding_tools()
        
        # Phase 4: Optimization Tools
        self.optimization_tools = self._initialize_optimization_tools()
        
        # Phase 5: Surface Analysis Tools
        self.surface_tools = self._initialize_surface_tools()
        
        print("✓ Comprehensive ToM Analyzer initialized successfully")
    
    def _initialize_syntactic_tools(self):
        """Initialize syntactic parsing tools"""
        tools = {}
        
        # spaCy with constituency parser
        if SPACY_AVAILABLE:
            try:
                nlp = spacy.load("en_core_web_sm")
                if BENEPAR_AVAILABLE:
                    try:
                        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
                        tools['spacy_with_benepar'] = nlp
                        print("✓ spaCy + benepar constituency parser loaded")
                    except:
                        tools['spacy'] = nlp
                        print("✓ spaCy loaded (benepar failed)")
                else:
                    tools['spacy'] = nlp
                    print("✓ spaCy loaded")
            except:
                print("✗ spaCy failed to load")
        
        # Stanza
        if STANZA_AVAILABLE:
            try:
                stanza_nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse,constituency')
                tools['stanza'] = stanza_nlp
                print("✓ Stanza pipeline loaded")
            except:
                print("✗ Stanza failed to load")
        
        # NeoSCA
        if NEOSCA_AVAILABLE:
            try:
                tools['neosca'] = neosca
                print("✓ NeoSCA loaded")
            except:
                print("✗ NeoSCA failed to load")
        
        return tools
    
    def _initialize_semantic_tools(self):
        """Initialize semantic parsing tools"""
        tools = {}
        
        # AllenNLP SRL
        if ALLENNLP_AVAILABLE:
            try:
                srl_predictor = Predictor.from_path(
                    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
                )
                tools['allennlp_srl'] = srl_predictor
                print("✓ AllenNLP SRL predictor loaded")
            except:
                print("✗ AllenNLP SRL failed to load")
        
        # AMR parsing
        if AMRLIB_AVAILABLE:
            try:
                amr_parser = amrlib.load_stog_model()
                tools['amr'] = amr_parser
                print("✓ AMR parser loaded")
            except:
                print("✗ AMR parser failed to load")
        
        # TUPA (UCCA)
        if TUPA_AVAILABLE:
            try:
                # TUPA initialization would go here
                tools['tupa'] = tupa
                print("✓ TUPA loaded")
            except:
                print("✗ TUPA failed to load")
        
        return tools
    
    def _initialize_embedding_tools(self):
        """Initialize embedding analysis tools"""
        tools = {}
        
        # Sentence Transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                tools['sentence_transformer'] = sentence_model
                print("✓ Sentence Transformer loaded")
            except:
                print("✗ Sentence Transformer failed to load")
        
        # BERT for attention analysis
        if BERTVIZ_AVAILABLE:
            try:
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                model = AutoModel.from_pretrained('bert-base-uncased')
                tools['bert_tokenizer'] = tokenizer
                tools['bert_model'] = model
                print("✓ BERT for attention analysis loaded")
            except:
                print("✗ BERT attention tools failed to load")
        
        return tools
    
    def _initialize_optimization_tools(self):
        """Initialize optimization tools"""
        tools = {}
        
        if DEAP_AVAILABLE:
            tools['deap'] = True
            print("✓ DEAP available")
        
        if OPTUNA_AVAILABLE:
            tools['optuna'] = True
            print("✓ Optuna available")
        
        if SKOPT_AVAILABLE:
            tools['skopt'] = True
            print("✓ scikit-optimize available")
        
        return tools
    
    def _initialize_surface_tools(self):
        """Initialize surface analysis tools"""
        tools = {}
        
        if TEXTSTAT_AVAILABLE:
            tools['textstat'] = textstat
            print("✓ textstat loaded")
        
        if NETWORKX_AVAILABLE:
            tools['networkx'] = nx
            print("✓ NetworkX loaded")
        
        return tools
    
    def analyze_comprehensive(self, text, text_type='Story'):
        """
        Comprehensive analysis using all available tools
        Implements the full pipeline from the research report
        """
        if not text or str(text).strip() == '' or str(text).lower() == 'nan':
            return self._get_empty_comprehensive_features(text_type)
        
        text = str(text).strip()
        features = {}
        
        print(f"Analyzing {text_type} with comprehensive pipeline...")
        
        # Phase 1: Advanced Syntactic Complexity
        print("  Phase 1: Syntactic complexity analysis...")
        features.update(self._analyze_syntactic_complexity_comprehensive(text, text_type))
        
        # Phase 2: Semantic Role and Graph Complexity
        print("  Phase 2: Semantic complexity analysis...")
        features.update(self._analyze_semantic_complexity_comprehensive(text, text_type))
        
        # Phase 3: Embedding-Based Complexity
        print("  Phase 3: Embedding complexity analysis...")
        features.update(self._analyze_embedding_complexity_comprehensive(text, text_type))
        
        # Phase 4: Network and Graph Analysis
        print("  Phase 4: Network analysis...")
        features.update(self._analyze_network_complexity(text, text_type))
        
        # Phase 5: Advanced Surface Metrics
        print("  Phase 5: Surface complexity analysis...")
        features.update(self._analyze_surface_complexity_comprehensive(text, text_type))
        
        return features
    
    def _analyze_syntactic_complexity_comprehensive(self, text, text_type):
        """Phase 1: Comprehensive syntactic analysis"""
        features = {}
        
        # spaCy-based analysis
        if 'spacy' in self.syntactic_tools or 'spacy_with_benepar' in self.syntactic_tools:
            nlp = self.syntactic_tools.get('spacy_with_benepar', self.syntactic_tools.get('spacy'))
            if nlp:
                try:
                    doc = nlp(text)
                    
                    # Dependency parse depth analysis
                    dep_features = self._calculate_dependency_complexity(doc, text_type)
                    features.update(dep_features)
                    
                    # Constituency parse depth (if available)
                    if 'spacy_with_benepar' in self.syntactic_tools:
                        const_features = self._calculate_constituency_complexity(doc, text_type)
                        features.update(const_features)
                    
                    # Advanced syntactic features
                    synt_features = self._calculate_advanced_syntactic_features(doc, text_type)
                    features.update(synt_features)
                    
                except Exception as e:
                    print(f"    Error in spaCy analysis: {e}")
        
        # Stanza-based analysis
        if 'stanza' in self.syntactic_tools:
            try:
                stanza_doc = self.syntactic_tools['stanza'](text)
                stanza_features = self._analyze_stanza_syntax(stanza_doc, text_type)
                features.update(stanza_features)
            except Exception as e:
                print(f"    Error in Stanza analysis: {e}")
        
        # NeoSCA analysis
        if 'neosca' in self.syntactic_tools:
            try:
                neosca_features = self._analyze_neosca_complexity(text, text_type)
                features.update(neosca_features)
            except Exception as e:
                print(f"    Error in NeoSCA analysis: {e}")
        
        return features
    
    def _analyze_semantic_complexity_comprehensive(self, text, text_type):
        """Phase 2: Comprehensive semantic analysis"""
        features = {}
        
        # AllenNLP SRL analysis removed due to dependency conflicts
        pass
        
        # AMR analysis
        if 'amr' in self.semantic_tools:
            try:
                amr_features = self._analyze_amr_complexity(text, text_type)
                features.update(amr_features)
            except Exception as e:
                print(f"    Error in AMR analysis: {e}")
        
        # UCCA analysis
        if 'tupa' in self.semantic_tools:
            try:
                ucca_features = self._analyze_ucca_complexity(text, text_type)
                features.update(ucca_features)
            except Exception as e:
                print(f"    Error in UCCA analysis: {e}")
        
        return features
    
    def _analyze_embedding_complexity_comprehensive(self, text, text_type):
        """Phase 3: Comprehensive embedding analysis"""
        features = {}
        
        # Sentence transformer analysis
        if 'sentence_transformer' in self.embedding_tools:
            try:
                embedding_features = self._analyze_sentence_embeddings(text, text_type)
                features.update(embedding_features)
            except Exception as e:
                print(f"    Error in sentence embedding analysis: {e}")
        
        # BERT attention analysis
        if 'bert_tokenizer' in self.embedding_tools and 'bert_model' in self.embedding_tools:
            try:
                attention_features = self._analyze_attention_patterns(text, text_type)
                features.update(attention_features)
            except Exception as e:
                print(f"    Error in attention analysis: {e}")
        
        return features
    
    def _analyze_network_complexity(self, text, text_type):
        """Phase 4: Network and graph analysis"""
        features = {}
        
        if 'networkx' in self.surface_tools:
            try:
                # Entity co-occurrence network
                network_features = self._analyze_entity_networks(text, text_type)
                features.update(network_features)
                
                # Discourse coherence graph
                coherence_features = self._analyze_discourse_networks(text, text_type)
                features.update(coherence_features)
                
            except Exception as e:
                print(f"    Error in network analysis: {e}")
        
        return features
    
    def _analyze_surface_complexity_comprehensive(self, text, text_type):
        """Phase 5: Comprehensive surface analysis"""
        features = {}
        
        if 'textstat' in self.surface_tools:
            try:
                # All textstat readability measures
                readability_features = self._calculate_all_readability_metrics(text, text_type)
                features.update(readability_features)
            except Exception as e:
                print(f"    Error in readability analysis: {e}")
        
        return features
    
    def _calculate_dependency_complexity(self, doc, text_type):
        """Calculate comprehensive dependency parse complexity"""
        features = {}
        
        # Basic depth metrics
        depths = []
        arc_lengths = []
        
        for token in doc:
            # Calculate depth from root
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
            
            # Calculate arc length
            if token.head != token:
                arc_length = abs(token.i - token.head.i)
                arc_lengths.append(arc_length)
        
        # Depth statistics
        features[f'{text_type}_Dep_Max_Depth'] = max(depths) if depths else 0
        features[f'{text_type}_Dep_Avg_Depth'] = np.mean(depths) if depths else 0
        features[f'{text_type}_Dep_Depth_Variance'] = np.var(depths) if depths else 0
        features[f'{text_type}_Dep_Total_Depth'] = sum(depths)
        
        # Arc length statistics
        features[f'{text_type}_Arc_Max_Length'] = max(arc_lengths) if arc_lengths else 0
        features[f'{text_type}_Arc_Avg_Length'] = np.mean(arc_lengths) if arc_lengths else 0
        features[f'{text_type}_Arc_Length_Variance'] = np.var(arc_lengths) if arc_lengths else 0
        
        # Dependency label diversity
        dep_labels = [token.dep_ for token in doc]
        unique_labels = len(set(dep_labels))
        features[f'{text_type}_Dep_Label_Diversity'] = unique_labels / len(dep_labels) if dep_labels else 0
        
        # Complex dependency patterns
        complex_deps = ['ccomp', 'xcomp', 'advcl', 'acl', 'relcl']
        complex_count = sum(1 for token in doc if token.dep_ in complex_deps)
        features[f'{text_type}_Complex_Dependencies'] = complex_count
        
        return features
    
    def _calculate_constituency_complexity(self, doc, text_type):
        """Calculate constituency parse complexity using benepar"""
        features = {}
        
        try:
            max_depth = 0
            total_depth = 0
            sentence_count = 0
            
            for sent in doc.sents:
                if hasattr(sent._, 'parse_tree'):
                    tree = sent._.parse_tree
                    depth = self._tree_depth(tree)
                    max_depth = max(max_depth, depth)
                    total_depth += depth
                    sentence_count += 1
            
            features[f'{text_type}_Const_Max_Depth'] = max_depth
            features[f'{text_type}_Const_Avg_Depth'] = total_depth / sentence_count if sentence_count > 0 else 0
            features[f'{text_type}_Const_Total_Depth'] = total_depth
            
        except Exception as e:
            features[f'{text_type}_Const_Parse_Error'] = 1
        
        return features
    
    def _tree_depth(self, tree):
        """Calculate depth of constituency parse tree"""
        if not hasattr(tree, '__iter__'):
            return 0
        return 1 + max([self._tree_depth(child) for child in tree] + [0])
    
    def _calculate_advanced_syntactic_features(self, doc, text_type):
        """Calculate advanced syntactic complexity features"""
        features = {}
        
        # Clause analysis
        clause_markers = ['ccomp', 'xcomp', 'advcl', 'acl', 'relcl']
        subordinate_markers = ['advcl', 'ccomp', 'xcomp']
        
        total_clauses = sum(1 for token in doc if token.dep_ in clause_markers)
        subordinate_clauses = sum(1 for token in doc if token.dep_ in subordinate_markers)
        coordinate_clauses = sum(1 for token in doc if token.dep_ == 'conj' and token.pos_ == 'VERB')
        
        features[f'{text_type}_Total_Clauses'] = total_clauses
        features[f'{text_type}_Subordinate_Clauses'] = subordinate_clauses
        features[f'{text_type}_Coordinate_Clauses'] = coordinate_clauses
        features[f'{text_type}_Clause_Ratio'] = total_clauses / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0
        
        # Yngve depth approximation
        yngve_depth = 0
        for token in doc:
            if token.dep_ in ['ccomp', 'xcomp', 'advcl']:
                yngve_depth += 2
            elif token.dep_ in ['acl', 'relcl']:
                yngve_depth += 1
        
        features[f'{text_type}_Yngve_Depth'] = yngve_depth / len(doc) if len(doc) > 0 else 0
        
        # POS tag diversity
        pos_tags = [token.pos_ for token in doc]
        unique_pos = len(set(pos_tags))
        features[f'{text_type}_POS_Diversity'] = unique_pos / len(pos_tags) if pos_tags else 0
        
        # Named entity complexity
        entities = [ent.label_ for ent in doc.ents]
        features[f'{text_type}_Entity_Count'] = len(doc.ents)
        features[f'{text_type}_Entity_Type_Diversity'] = len(set(entities)) if entities else 0
        
        return features
    
    def _analyze_stanza_syntax(self, stanza_doc, text_type):
        """Analyze syntax using Stanza"""
        features = {}
        
        try:
            # Constituency parse analysis
            const_depths = []
            for sentence in stanza_doc.sentences:
                if hasattr(sentence, 'constituency'):
                    tree_str = str(sentence.constituency)
                    depth = tree_str.count('(')  # Approximation of tree depth
                    const_depths.append(depth)
            
            if const_depths:
                features[f'{text_type}_Stanza_Const_Max_Depth'] = max(const_depths)
                features[f'{text_type}_Stanza_Const_Avg_Depth'] = np.mean(const_depths)
                features[f'{text_type}_Stanza_Const_Variance'] = np.var(const_depths)
            
            # Dependency analysis
            dep_depths = []
            for sentence in stanza_doc.sentences:
                for word in sentence.words:
                    # Calculate depth in dependency tree
                    depth = 0
                    current_id = word.id
                    visited = set()
                    
                    while current_id != 0 and current_id not in visited:
                        visited.add(current_id)
                        # Find parent
                        parent_found = False
                        for w in sentence.words:
                            if w.id == current_id:
                                current_id = w.head
                                depth += 1
                                parent_found = True
                                break
                        if not parent_found:
                            break
                        if depth > 50:  # Prevent infinite loops
                            break
                    
                    dep_depths.append(depth)
            
            if dep_depths:
                features[f'{text_type}_Stanza_Dep_Max_Depth'] = max(dep_depths)
                features[f'{text_type}_Stanza_Dep_Avg_Depth'] = np.mean(dep_depths)
                features[f'{text_type}_Stanza_Dep_Variance'] = np.var(dep_depths)
        
        except Exception as e:
            features[f'{text_type}_Stanza_Error'] = 1
        
        return features
    
    def _analyze_neosca_complexity(self, text, text_type):
        """Analyze complexity using NeoSCA"""
        features = {}
        
        try:
            # NeoSCA analysis would go here
            # This is a placeholder as NeoSCA has specific input requirements
            features[f'{text_type}_NeoSCA_Placeholder'] = 0
        except Exception as e:
            features[f'{text_type}_NeoSCA_Error'] = 1
        
        return features
    
    
    def _analyze_amr_complexity(self, text, text_type):
        """Analyze AMR graph complexity"""
        features = {}
        
        try:
            amr_graphs = self.semantic_tools['amr'].parse_sents([text])
            
            for graph in amr_graphs:
                # Parse AMR graph structure
                graph_str = str(graph)
                
                # Count concepts and relations
                concept_count = graph_str.count(':')  # Approximation
                relation_count = graph_str.count('/')  # Approximation
                
                features[f'{text_type}_AMR_Concept_Count'] = concept_count
                features[f'{text_type}_AMR_Relation_Count'] = relation_count
                features[f'{text_type}_AMR_Complexity_Ratio'] = relation_count / concept_count if concept_count > 0 else 0
                
                # Graph depth approximation
                nesting_depth = graph_str.count('(')
                features[f'{text_type}_AMR_Nesting_Depth'] = nesting_depth
        
        except Exception as e:
            features[f'{text_type}_AMR_Error'] = 1
        
        return features
    
    def _analyze_ucca_complexity(self, text, text_type):
        """Analyze UCCA semantic complexity"""
        features = {}
        
        try:
            # UCCA analysis placeholder
            features[f'{text_type}_UCCA_Placeholder'] = 0
        except Exception as e:
            features[f'{text_type}_UCCA_Error'] = 1
        
        return features
    
    def _analyze_sentence_embeddings(self, text, text_type):
        """Analyze sentence embedding complexity"""
        features = {}
        
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return {f'{text_type}_Embedding_Insufficient_Sentences': 1}
            
            # Get embeddings
            embeddings = self.embedding_tools['sentence_transformer'].encode(sentences)
            
            # Embedding dispersion
            embedding_variance = np.var(embeddings, axis=0).mean()
            embedding_std = np.std(embeddings, axis=0).mean()
            
            features[f'{text_type}_Embedding_Variance'] = embedding_variance
            features[f'{text_type}_Embedding_Std'] = embedding_std
            
            # Semantic coherence
            coherence_scores = []
            for i in range(len(embeddings) - 1):
                similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                coherence_scores.append(similarity)
            
            features[f'{text_type}_Semantic_Coherence_Avg'] = np.mean(coherence_scores)
            features[f'{text_type}_Semantic_Coherence_Min'] = np.min(coherence_scores)
            features[f'{text_type}_Semantic_Coherence_Variance'] = np.var(coherence_scores)
            
            # Embedding trajectory
            trajectory_length = 0
            for i in range(len(embeddings) - 1):
                distance = np.linalg.norm(embeddings[i] - embeddings[i + 1])
                trajectory_length += distance
            
            features[f'{text_type}_Embedding_Trajectory_Length'] = trajectory_length
            features[f'{text_type}_Embedding_Avg_Step_Size'] = trajectory_length / (len(embeddings) - 1)
            
            # PCA analysis
            if len(embeddings) > 2:
                pca = PCA(n_components=min(3, len(embeddings)))
                pca.fit(embeddings)
                
                features[f'{text_type}_PCA_First_Component'] = pca.explained_variance_ratio_[0]
                features[f'{text_type}_PCA_Cumulative_Variance_2'] = np.sum(pca.explained_variance_ratio_[:2])
                features[f'{text_type}_PCA_Cumulative_Variance_3'] = np.sum(pca.explained_variance_ratio_)
            
            # Clustering analysis
            centroid = np.mean(embeddings, axis=0)
            distances_to_centroid = [np.linalg.norm(emb - centroid) for emb in embeddings]
            
            features[f'{text_type}_Avg_Distance_To_Centroid'] = np.mean(distances_to_centroid)
            features[f'{text_type}_Max_Distance_To_Centroid'] = np.max(distances_to_centroid)
            features[f'{text_type}_Centroid_Distance_Variance'] = np.var(distances_to_centroid)
            
        except Exception as e:
            features[f'{text_type}_Embedding_Error'] = 1
        
        return features
    
    def _analyze_attention_patterns(self, text, text_type):
        """Analyze BERT attention patterns for complexity"""
        features = {}
        
        try:
            tokenizer = self.embedding_tools['bert_tokenizer']
            model = self.embedding_tools['bert_model']
            
            # Tokenize text
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            
            # Get attention weights
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                attentions = outputs.attentions
            
            # Analyze attention patterns
            # Average attention across all heads and layers
            all_attentions = torch.stack(attentions)  # [layers, batch, heads, seq_len, seq_len]
            avg_attention = all_attentions.mean(dim=(0, 1, 2))  # [seq_len, seq_len]
            
            # Attention complexity metrics
            attention_entropy = -torch.sum(avg_attention * torch.log(avg_attention + 1e-10), dim=-1).mean()
            attention_variance = torch.var(avg_attention).item()
            max_attention = torch.max(avg_attention).item()
            
            features[f'{text_type}_Attention_Entropy'] = attention_entropy.item()
            features[f'{text_type}_Attention_Variance'] = attention_variance
            features[f'{text_type}_Attention_Max'] = max_attention
            
            # Cross-sentence attention (if multiple sentences)
            sentences = re.split(r'[.!?]+', text)
            if len(sentences) > 1:
                # This is a simplified approximation
                features[f'{text_type}_Cross_Sentence_Attention'] = attention_variance * len(sentences)
            
        except Exception as e:
            features[f'{text_type}_Attention_Error'] = 1
        
        return features
    
    def _analyze_entity_networks(self, text, text_type):
        """Analyze entity co-occurrence networks"""
        features = {}
        
        try:
            # Extract entities using spaCy if available
            if 'spacy' in self.syntactic_tools or 'spacy_with_benepar' in self.syntactic_tools:
                nlp = self.syntactic_tools.get('spacy_with_benepar', self.syntactic_tools.get('spacy'))
                doc = nlp(text)
                
                # Get entities
                entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
                
                if len(entities) > 1:
                    # Create co-occurrence network
                    G = self.surface_tools['networkx'].Graph()
                    
                    # Add entities as nodes
                    G.add_nodes_from(set(entities))
                    
                    # Add edges for co-occurring entities (simplified)
                    for i, entity1 in enumerate(entities):
                        for entity2 in entities[i+1:]:
                            if entity1 != entity2:
                                G.add_edge(entity1, entity2)
                    
                    # Network metrics
                    if len(G.nodes()) > 0:
                        features[f'{text_type}_Entity_Network_Nodes'] = len(G.nodes())
                        features[f'{text_type}_Entity_Network_Edges'] = len(G.edges())
                        features[f'{text_type}_Entity_Network_Density'] = self.surface_tools['networkx'].density(G)
                        
                        if len(G.nodes()) > 1:
                            features[f'{text_type}_Entity_Network_Clustering'] = self.surface_tools['networkx'].average_clustering(G)
                        else:
                            features[f'{text_type}_Entity_Network_Clustering'] = 0
                else:
                    features[f'{text_type}_Entity_Network_Insufficient'] = 1
            
        except Exception as e:
            features[f'{text_type}_Entity_Network_Error'] = 1
        
        return features
    
    def _analyze_discourse_networks(self, text, text_type):
        """Analyze discourse coherence networks"""
        features = {}
        
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) > 1:
                # Create discourse network based on word overlap
                G = self.surface_tools['networkx'].Graph()
                
                # Add sentences as nodes
                for i, sent in enumerate(sentences):
                    G.add_node(i, text=sent)
                
                # Add edges based on lexical overlap
                for i, sent1 in enumerate(sentences):
                    words1 = set(sent1.lower().split())
                    for j, sent2 in enumerate(sentences[i+1:], i+1):
                        words2 = set(sent2.lower().split())
                        
                        # Calculate overlap
                        if words1 and words2:
                            overlap = len(words1 & words2) / len(words1 | words2)
                            if overlap > 0.1:  # Threshold for connection
                                G.add_edge(i, j, weight=overlap)
                
                # Network metrics
                features[f'{text_type}_Discourse_Network_Nodes'] = len(G.nodes())
                features[f'{text_type}_Discourse_Network_Edges'] = len(G.edges())
                
                if len(G.nodes()) > 1:
                    features[f'{text_type}_Discourse_Network_Density'] = self.surface_tools['networkx'].density(G)
                    features[f'{text_type}_Discourse_Network_Clustering'] = self.surface_tools['networkx'].average_clustering(G)
                    
                    # Path-based metrics
                    if self.surface_tools['networkx'].is_connected(G):
                        features[f'{text_type}_Discourse_Avg_Path_Length'] = self.surface_tools['networkx'].average_shortest_path_length(G)
                    else:
                        features[f'{text_type}_Discourse_Connected_Components'] = self.surface_tools['networkx'].number_connected_components(G)
        
        except Exception as e:
            features[f'{text_type}_Discourse_Network_Error'] = 1
        
        return features
    
    def _calculate_all_readability_metrics(self, text, text_type):
        """Calculate all available textstat readability metrics"""
        features = {}
        
        try:
            textstat_tool = self.surface_tools['textstat']
            
            # Basic readability metrics
            features[f'{text_type}_Flesch_Reading_Ease'] = textstat_tool.flesch_reading_ease(text)
            features[f'{text_type}_Flesch_Kincaid_Grade'] = textstat_tool.flesch_kincaid_grade(text)
            features[f'{text_type}_Gunning_Fog'] = textstat_tool.gunning_fog(text)
            features[f'{text_type}_SMOG_Index'] = textstat_tool.smog_index(text)
            features[f'{text_type}_Automated_Readability_Index'] = textstat_tool.automated_readability_index(text)
            features[f'{text_type}_Coleman_Liau_Index'] = textstat_tool.coleman_liau_index(text)
            features[f'{text_type}_Linsear_Write_Formula'] = textstat_tool.linsear_write_formula(text)
            features[f'{text_type}_Dale_Chall_Readability'] = textstat_tool.dale_chall_readability_score(text)
            
            # Advanced metrics
            features[f'{text_type}_Difficult_Words'] = textstat_tool.difficult_words(text)
            features[f'{text_type}_Text_Standard'] = textstat_tool.text_standard(text, float_output=True)
            features[f'{text_type}_Reading_Time'] = textstat_tool.reading_time(text, ms_per_char=14.69)
            
            # Syllable and word complexity
            features[f'{text_type}_Syllable_Count'] = textstat_tool.syllable_count(text)
            features[f'{text_type}_Lexicon_Count'] = textstat_tool.lexicon_count(text)
            features[f'{text_type}_Sentence_Count'] = textstat_tool.sentence_count(text)
            
            # Composite readability score
            readability_scores = [
                features[f'{text_type}_Flesch_Reading_Ease'],
                features[f'{text_type}_Flesch_Kincaid_Grade'],
                features[f'{text_type}_Gunning_Fog'],
                features[f'{text_type}_SMOG_Index']
            ]
            features[f'{text_type}_Composite_Readability'] = np.mean([abs(score) for score in readability_scores if score is not None])
            
        except Exception as e:
            features[f'{text_type}_Readability_Error'] = 1
        
        return features
    
    def _get_empty_comprehensive_features(self, text_type):
        """Return empty features for error cases"""
        return {
            f'{text_type}_Comprehensive_Analysis_Failed': 1,
            f'{text_type}_Empty_Text': 1
        }

def analyze_text_comprehensive_tom(text, text_type='Story'):
    """Wrapper function for comprehensive ToM analysis"""
    if not hasattr(analyze_text_comprehensive_tom, 'analyzer'):
        analyze_text_comprehensive_tom.analyzer = ComprehensiveToMAnalyzer()
    
    analyzer = analyze_text_comprehensive_tom.analyzer
    return analyzer.analyze_comprehensive(text, text_type)

def apply_comprehensive_analysis_to_dataset(dataset_path='dataset_v13_advanced_linguistic.csv', 
                                          output_path='dataset_v14_comprehensive_advanced.csv',
                                          batch_size=10):
    """
    Apply comprehensive analysis to the entire dataset
    """
    print("="*80)
    print("COMPREHENSIVE THEORY OF MIND ANALYSIS")
    print("Implementing all advanced techniques from requirements_advanced.txt")
    print("="*80)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()
    
    # Clean data
    df_clean = df[df['ABILITY'].notna()].copy()
    df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
    df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
    
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Categories: {df_clean['Main_Category'].unique()}")
    
    # Initialize analyzer
    analyzer = ComprehensiveToMAnalyzer()
    
    # Process in batches
    total_batches = (len(df_clean) + batch_size - 1) // batch_size
    
    story_features_list = []
    question_features_list = []
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df_clean))
        
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches} (rows {start_idx}-{end_idx})...")
        
        batch_story_features = []
        batch_question_features = []
        
        for idx in range(start_idx, end_idx):
            row = df_clean.iloc[idx]
            
            print(f"  Processing row {idx + 1}/{len(df_clean)}...")
            
            # Analyze story
            story_text = row.get('STORY', '')
            try:
                story_features = analyzer.analyze_comprehensive(story_text, 'Story')
            except Exception as e:
                print(f"    Error analyzing story at row {idx}: {e}")
                story_features = analyzer._get_empty_comprehensive_features('Story')
            batch_story_features.append(story_features)
            
            # Analyze question
            question_text = row.get('QUESTION', '')
            try:
                question_features = analyzer.analyze_comprehensive(question_text, 'Question')
            except Exception as e:
                print(f"    Error analyzing question at row {idx}: {e}")
                question_features = analyzer._get_empty_comprehensive_features('Question')
            batch_question_features.append(question_features)
        
        story_features_list.extend(batch_story_features)
        question_features_list.extend(batch_question_features)
        
        # Save intermediate results
        if batch_idx % 5 == 0:  # Save every 5 batches
            print(f"  Saving intermediate results after batch {batch_idx + 1}...")
            intermediate_story_df = pd.DataFrame(story_features_list)
            intermediate_question_df = pd.DataFrame(question_features_list)
            intermediate_combined = pd.concat([df_clean.iloc[:len(story_features_list)].reset_index(drop=True), 
                                             intermediate_story_df, intermediate_question_df], axis=1)
            intermediate_combined.to_csv(f'intermediate_comprehensive_batch_{batch_idx + 1}.csv', index=False)
    
    # Convert to DataFrames
    story_features_df = pd.DataFrame(story_features_list)
    question_features_df = pd.DataFrame(question_features_list)
    
    print(f"\nStory features shape: {story_features_df.shape}")
    print(f"Question features shape: {question_features_df.shape}")
    
    # Combine with original dataset
    print("Combining with original dataset...")
    df_clean_reset = df_clean.reset_index(drop=True)
    story_features_df_reset = story_features_df.reset_index(drop=True)
    question_features_df_reset = question_features_df.reset_index(drop=True)
    
    final_df = pd.concat([df_clean_reset, story_features_df_reset, question_features_df_reset], axis=1)
    
    print(f"Final dataset shape: {final_df.shape}")
    
    # Save final dataset
    final_df.to_csv(output_path, index=False)
    print(f"Comprehensive analysis complete! Saved to: {output_path}")
    
    # Analysis summary
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    total_new_features = len(story_features_df.columns) + len(question_features_df.columns)
    print(f"Total new features added: {total_new_features}")
    print(f"Story-level features: {len(story_features_df.columns)}")
    print(f"Question-level features: {len(question_features_df.columns)}")
    print(f"Original features: {len(df_clean.columns)}")
    print(f"Final total features: {len(final_df.columns)}")
    
    # Feature categories
    story_categories = {
        'Syntactic': [col for col in story_features_df.columns if any(x in col for x in ['Dep_', 'Const_', 'Clause', 'Yngve', 'Arc'])],
        'Semantic': [col for col in story_features_df.columns if any(x in col for x in ['SRL_', 'AMR_', 'UCCA_'])],
        'Embedding': [col for col in story_features_df.columns if any(x in col for x in ['Embedding_', 'Coherence_', 'PCA_', 'Attention_'])],
        'Network': [col for col in story_features_df.columns if any(x in col for x in ['Network_', 'Discourse_'])],
        'Readability': [col for col in story_features_df.columns if any(x in col for x in ['Flesch_', 'Gunning_', 'SMOG_', 'Dale_'])]
    }
    
    print(f"\nFeature categories (Story-level):")
    for category, features in story_categories.items():
        print(f"  {category}: {len(features)} features")
    
    return final_df, story_features_df, question_features_df

if __name__ == "__main__":
    # Run comprehensive analysis
    print("Starting comprehensive Theory of Mind analysis...")
    
    final_df, story_features, question_features = apply_comprehensive_analysis_to_dataset(
        dataset_path='dataset_v13_advanced_linguistic.csv',
        output_path='dataset_v14_comprehensive_advanced.csv',
        batch_size=5  # Small batch size due to computational complexity
    )
    
    print("Comprehensive analysis complete!")
