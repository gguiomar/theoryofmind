#!/usr/bin/env python3
"""
Next-Generation Theory of Mind Metrics
Implementing sophisticated multiplicative and dynamic complexity measures
designed to challenge even 70B models
"""

import pandas as pd
import numpy as np
import spacy
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class NextGenToMAnalyzer:
    """
    Next-generation Theory of Mind analyzer with multiplicative and dynamic complexity measures
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        print("Initializing next-generation ToM analyzer...")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Enhanced mental state vocabularies
        self.mental_state_verbs = {
            'cognitive': ['think', 'know', 'believe', 'understand', 'realize', 'remember', 'forget', 'wonder', 'doubt'],
            'emotional': ['feel', 'love', 'hate', 'fear', 'worry', 'hope', 'enjoy', 'surprise', 'anger'],
            'volitional': ['want', 'desire', 'wish', 'intend', 'plan', 'decide', 'choose', 'prefer'],
            'perceptual': ['see', 'hear', 'notice', 'observe', 'watch', 'listen', 'sense']
        }
        
        # Uncertainty markers (expanded)
        self.uncertainty_markers = {
            'explicit': ['maybe', 'perhaps', 'possibly', 'probably', 'might', 'could', 'seem', 'appear'],
            'implicit': ['I think', 'I believe', 'I guess', 'sort of', 'kind of', 'somewhat'],
            'hedging': ['rather', 'quite', 'fairly', 'relatively', 'supposedly', 'allegedly']
        }
        
        # Causal markers (expanded)
        self.causal_markers = {
            'explicit': ['because', 'since', 'due to', 'therefore', 'thus', 'hence', 'so'],
            'implicit': ['after', 'when', 'then', 'following', 'resulting in'],
            'counterfactual': ['if', 'unless', 'would have', 'could have', 'should have']
        }
        
        # Emotional transition words
        self.emotion_transitions = [
            'suddenly', 'then', 'but', 'however', 'although', 'despite', 'nevertheless',
            'meanwhile', 'later', 'afterwards', 'eventually', 'finally'
        ]
        
        print("✓ Next-generation ToM analyzer initialized")
    
    def analyze_multiplicative_complexity(self, text, text_type='Story'):
        """Analyze multiplicative complexity indices"""
        features = {}
        
        if not text or str(text).strip() == '' or str(text).lower() == 'nan':
            return self._get_empty_multiplicative_features(text_type)
        
        text = str(text).strip()
        
        # 1. Cognitive Load Index (Entity × Causal × Uncertainty × Temporal)
        entity_density = self._calculate_entity_density(text)
        causal_depth = self._calculate_causal_depth(text)
        uncertainty_level = self._calculate_uncertainty_level(text)
        temporal_complexity = self._calculate_temporal_complexity(text)
        
        cognitive_load_index = entity_density * causal_depth * uncertainty_level * temporal_complexity
        
        features.update({
            f'{text_type}_Entity_Density': entity_density,
            f'{text_type}_Causal_Depth': causal_depth,
            f'{text_type}_Uncertainty_Level': uncertainty_level,
            f'{text_type}_Temporal_Complexity': temporal_complexity,
            f'{text_type}_Cognitive_Load_Index': cognitive_load_index
        })
        
        # 2. Mental State Interaction Complexity
        ms_depth = self._calculate_mental_state_depth(text)
        entity_switches = self._calculate_entity_switches(text)
        emotional_transitions = self._calculate_emotional_transitions(text)
        
        mental_state_interaction = ms_depth * entity_switches * emotional_transitions
        
        features.update({
            f'{text_type}_Mental_State_Depth': ms_depth,
            f'{text_type}_Entity_Switches': entity_switches,
            f'{text_type}_Emotional_Transitions': emotional_transitions,
            f'{text_type}_Mental_State_Interaction': mental_state_interaction
        })
        
        # 3. Inference Chain Complexity
        inference_depth = self._calculate_inference_chain_depth(text)
        contradiction_resolution = self._calculate_contradiction_resolution(text)
        perspective_shifts = self._calculate_perspective_shifts(text)
        
        inference_complexity = inference_depth * contradiction_resolution * perspective_shifts
        
        features.update({
            f'{text_type}_Inference_Chain_Depth': inference_depth,
            f'{text_type}_Contradiction_Resolution': contradiction_resolution,
            f'{text_type}_Perspective_Shifts': perspective_shifts,
            f'{text_type}_Inference_Complexity': inference_complexity
        })
        
        return features
    
    def analyze_dynamic_complexity(self, text, text_type='Story'):
        """Analyze dynamic complexity measures"""
        features = {}
        
        if not text or str(text).strip() == '' or str(text).lower() == 'nan':
            return self._get_empty_dynamic_features(text_type)
        
        text = str(text).strip()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return self._get_empty_dynamic_features(text_type)
        
        # Calculate complexity for each sentence
        sentence_complexities = []
        for sentence in sentences:
            # Simple complexity measure per sentence
            entity_count = len(self._extract_entities(sentence))
            mental_state_count = self._count_mental_state_verbs(sentence)
            uncertainty_count = self._count_uncertainty_markers(sentence)
            
            sentence_complexity = entity_count + mental_state_count + uncertainty_count
            sentence_complexities.append(sentence_complexity)
        
        # Dynamic measures
        complexity_gradient = (max(sentence_complexities) - min(sentence_complexities)) / len(sentences) if len(sentences) > 0 else 0
        complexity_variance = np.var(sentence_complexities) if len(sentence_complexities) > 1 else 0
        complexity_peaks = sum(1 for c in sentence_complexities if c > np.mean(sentence_complexities) + np.std(sentence_complexities))
        
        # Working memory load (simultaneous tracking)
        working_memory_load = self._calculate_working_memory_load(text)
        
        # Attention switching cost
        attention_switching = self._calculate_attention_switching(text)
        
        features.update({
            f'{text_type}_Complexity_Gradient': complexity_gradient,
            f'{text_type}_Complexity_Variance': complexity_variance,
            f'{text_type}_Complexity_Peaks': complexity_peaks,
            f'{text_type}_Working_Memory_Load': working_memory_load,
            f'{text_type}_Attention_Switching_Cost': attention_switching
        })
        
        return features
    
    def analyze_meta_cognitive_complexity(self, text, text_type='Story'):
        """Analyze meta-cognitive complexity"""
        features = {}
        
        if not text or str(text).strip() == '' or str(text).lower() == 'nan':
            return self._get_empty_meta_features(text_type)
        
        text = str(text).strip()
        
        # Recursive mental state depth (A thinks B believes C knows...)
        recursive_depth = self._calculate_recursive_mental_state_depth(text)
        
        # Meta-uncertainty (uncertainty about mental states)
        meta_uncertainty = self._calculate_meta_uncertainty(text)
        
        # Perspective shift complexity
        perspective_complexity = self._calculate_dynamic_perspective_complexity(text)
        
        # Theory of mind about theory of mind
        meta_tom = self._calculate_meta_tom_complexity(text)
        
        # Cognitive interference score
        cognitive_interference = self._calculate_cognitive_interference(text)
        
        features.update({
            f'{text_type}_Recursive_Mental_State_Depth': recursive_depth,
            f'{text_type}_Meta_Uncertainty_Index': meta_uncertainty,
            f'{text_type}_Dynamic_Perspective_Complexity': perspective_complexity,
            f'{text_type}_Meta_ToM_Complexity': meta_tom,
            f'{text_type}_Cognitive_Interference_Score': cognitive_interference
        })
        
        return features
    
    def _calculate_entity_density(self, text):
        """Calculate entity density with coreference complexity"""
        if not self.nlp:
            return len(re.findall(r'\b[A-Z][a-z]+\b', text)) / len(text.split()) if text.split() else 0
        
        try:
            doc = self.nlp(text)
            entities = [ent for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]
            
            # Weight by coreference complexity
            pronouns = len([token for token in doc if token.pos_ == 'PRON'])
            entity_density = (len(entities) + pronouns * 0.5) / len(doc) if len(doc) > 0 else 0
            
            return entity_density
        except:
            return 0
    
    def _calculate_causal_depth(self, text):
        """Calculate nested causal reasoning depth"""
        text_lower = text.lower()
        
        # Count explicit causal markers
        explicit_causal = sum(1 for marker in self.causal_markers['explicit'] if marker in text_lower)
        
        # Count implicit causal markers
        implicit_causal = sum(1 for marker in self.causal_markers['implicit'] if marker in text_lower)
        
        # Count counterfactual markers (higher complexity)
        counterfactual = sum(1 for marker in self.causal_markers['counterfactual'] if marker in text_lower)
        
        # Nested causality patterns
        nested_patterns = [
            r'because.*because',
            r'since.*since',
            r'if.*then.*because',
            r'when.*then.*so'
        ]
        
        nested_count = sum(len(re.findall(pattern, text_lower)) for pattern in nested_patterns)
        
        # Weighted causal depth
        causal_depth = explicit_causal + implicit_causal * 1.5 + counterfactual * 2 + nested_count * 3
        
        return causal_depth / len(text.split()) if text.split() else 0
    
    def _calculate_uncertainty_level(self, text):
        """Calculate multi-level uncertainty"""
        text_lower = text.lower()
        
        # Count different types of uncertainty
        explicit_uncertainty = sum(1 for marker in self.uncertainty_markers['explicit'] if marker in text_lower)
        implicit_uncertainty = sum(1 for marker in self.uncertainty_markers['implicit'] if marker in text_lower)
        hedging = sum(1 for marker in self.uncertainty_markers['hedging'] if marker in text_lower)
        
        # Nested uncertainty patterns
        nested_uncertainty_patterns = [
            r'maybe.*think.*might',
            r'perhaps.*believe.*could',
            r'possibly.*seem.*appear'
        ]
        
        nested_uncertainty = sum(len(re.findall(pattern, text_lower)) for pattern in nested_uncertainty_patterns)
        
        # Weighted uncertainty level
        uncertainty_level = explicit_uncertainty + implicit_uncertainty * 1.5 + hedging * 2 + nested_uncertainty * 3
        
        return uncertainty_level / len(text.split()) if text.split() else 0
    
    def _calculate_temporal_complexity(self, text):
        """Calculate temporal reasoning complexity"""
        temporal_markers = ['before', 'after', 'during', 'while', 'when', 'then', 'later', 'earlier', 'meanwhile', 'suddenly']
        
        text_lower = text.lower()
        temporal_count = sum(1 for marker in temporal_markers if marker in text_lower)
        
        # Temporal shifts and sequences
        temporal_shift_patterns = [
            r'before.*after',
            r'first.*then.*finally',
            r'earlier.*now.*later',
            r'meanwhile.*then'
        ]
        
        temporal_shifts = sum(len(re.findall(pattern, text_lower)) for pattern in temporal_shift_patterns)
        
        return (temporal_count + temporal_shifts * 2) / len(text.split()) if text.split() else 0
    
    def _calculate_mental_state_depth(self, text):
        """Calculate depth of mental state attributions"""
        text_lower = text.lower()
        
        # Count mental state verbs by category
        total_ms_verbs = 0
        for category, verbs in self.mental_state_verbs.items():
            count = sum(1 for verb in verbs if verb in text_lower)
            # Weight cognitive verbs higher
            weight = 2 if category == 'cognitive' else 1
            total_ms_verbs += count * weight
        
        # Nested mental state patterns
        nested_patterns = [
            r'think.*believe.*know',
            r'believe.*think.*feel',
            r'know.*think.*want'
        ]
        
        nested_ms = sum(len(re.findall(pattern, text_lower)) for pattern in nested_patterns)
        
        return (total_ms_verbs + nested_ms * 3) / len(text.split()) if text.split() else 0
    
    def _calculate_entity_switches(self, text):
        """Calculate entity focus switching"""
        if not self.nlp:
            # Simple approximation
            pronouns = ['he', 'she', 'they', 'him', 'her', 'them']
            text_lower = text.lower()
            return sum(1 for pronoun in pronouns if pronoun in text_lower) / len(text.split()) if text.split() else 0
        
        try:
            doc = self.nlp(text)
            
            # Track entity mentions across sentences
            sentences = list(doc.sents)
            entity_switches = 0
            
            for i in range(1, len(sentences)):
                prev_entities = set([ent.text.lower() for ent in sentences[i-1].ents if ent.label_ in ['PERSON']])
                curr_entities = set([ent.text.lower() for ent in sentences[i].ents if ent.label_ in ['PERSON']])
                
                # Count switches (new entities introduced)
                new_entities = curr_entities - prev_entities
                entity_switches += len(new_entities)
            
            return entity_switches / len(sentences) if len(sentences) > 0 else 0
        except:
            return 0
    
    def _calculate_emotional_transitions(self, text):
        """Calculate emotional state transitions"""
        text_lower = text.lower()
        
        # Count emotional transition markers
        transition_count = sum(1 for marker in self.emotion_transitions if marker in text_lower)
        
        # Emotional state words
        emotions = ['happy', 'sad', 'angry', 'surprised', 'afraid', 'excited', 'worried', 'confused']
        emotion_count = sum(1 for emotion in emotions if emotion in text_lower)
        
        # Emotional contrast patterns
        contrast_patterns = [
            r'happy.*but.*sad',
            r'excited.*however.*worried',
            r'surprised.*then.*confused'
        ]
        
        emotional_contrasts = sum(len(re.findall(pattern, text_lower)) for pattern in contrast_patterns)
        
        return (transition_count + emotion_count + emotional_contrasts * 2) / len(text.split()) if text.split() else 0
    
    def _calculate_inference_chain_depth(self, text):
        """Calculate logical inference chain depth"""
        # Inference markers
        inference_markers = ['therefore', 'thus', 'hence', 'so', 'consequently', 'as a result']
        
        text_lower = text.lower()
        inference_count = sum(1 for marker in inference_markers if marker in text_lower)
        
        # Multi-step inference patterns
        multi_step_patterns = [
            r'if.*then.*therefore',
            r'because.*so.*thus',
            r'since.*therefore.*consequently'
        ]
        
        multi_step = sum(len(re.findall(pattern, text_lower)) for pattern in multi_step_patterns)
        
        return (inference_count + multi_step * 3) / len(text.split()) if text.split() else 0
    
    def _calculate_contradiction_resolution(self, text):
        """Calculate contradictory information resolution difficulty"""
        contradiction_markers = ['but', 'however', 'although', 'despite', 'nevertheless', 'yet', 'still']
        
        text_lower = text.lower()
        contradiction_count = sum(1 for marker in contradiction_markers if marker in text_lower)
        
        # Complex contradiction patterns
        complex_contradictions = [
            r'think.*but.*actually',
            r'believe.*however.*really',
            r'seem.*although.*truth'
        ]
        
        complex_count = sum(len(re.findall(pattern, text_lower)) for pattern in complex_contradictions)
        
        return (contradiction_count + complex_count * 2) / len(text.split()) if text.split() else 0
    
    def _calculate_perspective_shifts(self, text):
        """Calculate perspective shifting complexity"""
        perspective_markers = ['from his perspective', 'in her view', 'according to', 'from their point of view']
        
        text_lower = text.lower()
        perspective_count = sum(1 for marker in perspective_markers if marker in text_lower)
        
        # Implicit perspective shifts (pronouns changing reference)
        pronoun_shifts = len(re.findall(r'he.*she.*he', text_lower)) + len(re.findall(r'she.*he.*she', text_lower))
        
        return (perspective_count * 2 + pronoun_shifts) / len(text.split()) if text.split() else 0
    
    def _calculate_working_memory_load(self, text):
        """Calculate simultaneous tracking requirements"""
        if not self.nlp:
            return 0
        
        try:
            doc = self.nlp(text)
            
            # Count entities that need to be tracked simultaneously
            entities = [ent for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]
            
            # Count mental states that need tracking
            mental_state_count = sum(1 for token in doc if token.lemma_ in 
                                   [verb for verbs in self.mental_state_verbs.values() for verb in verbs])
            
            # Simultaneous tracking load
            return (len(entities) * mental_state_count) / len(doc) if len(doc) > 0 else 0
        except:
            return 0
    
    def _calculate_attention_switching(self, text):
        """Calculate attention switching cost"""
        # Topic change markers
        topic_markers = ['meanwhile', 'elsewhere', 'at the same time', 'on the other hand', 'in contrast']
        
        text_lower = text.lower()
        topic_switches = sum(1 for marker in topic_markers if marker in text_lower)
        
        # Sentence-level topic changes (approximation)
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 2:
            return 0
        
        # Simple topic change detection based on entity changes
        topic_changes = 0
        for i in range(1, len(sentences)):
            prev_words = set(sentences[i-1].lower().split())
            curr_words = set(sentences[i].lower().split())
            
            # If less than 30% word overlap, consider it a topic change
            overlap = len(prev_words & curr_words) / len(prev_words | curr_words) if prev_words | curr_words else 0
            if overlap < 0.3:
                topic_changes += 1
        
        return (topic_switches * 2 + topic_changes) / len(sentences) if len(sentences) > 0 else 0
    
    def _calculate_recursive_mental_state_depth(self, text):
        """Calculate recursive mental state depth (A thinks B believes C knows...)"""
        # Patterns for recursive mental states
        recursive_patterns = [
            r'think.*believe.*know',
            r'believe.*think.*feel',
            r'know.*believe.*think',
            r'feel.*think.*believe'
        ]
        
        text_lower = text.lower()
        recursive_count = sum(len(re.findall(pattern, text_lower)) for pattern in recursive_patterns)
        
        # More complex recursive patterns
        complex_recursive = [
            r'think.*that.*believe.*that.*know',
            r'believe.*that.*think.*that.*feel'
        ]
        
        complex_count = sum(len(re.findall(pattern, text_lower)) for pattern in complex_recursive)
        
        return (recursive_count + complex_count * 3) / len(text.split()) if text.split() else 0
    
    def _calculate_meta_uncertainty(self, text):
        """Calculate uncertainty about mental states"""
        meta_uncertainty_patterns = [
            r'not sure.*think',
            r'uncertain.*believe',
            r'doubt.*know',
            r'wonder.*feel'
        ]
        
        text_lower = text.lower()
        meta_uncertainty = sum(len(re.findall(pattern, text_lower)) for pattern in meta_uncertainty_patterns)
        
        return meta_uncertainty / len(text.split()) if text.split() else 0
    
    def _calculate_dynamic_perspective_complexity(self, text):
        """Calculate dynamic perspective complexity"""
        # Perspective shift markers with temporal elements
        dynamic_perspective_patterns = [
            r'first.*thought.*then.*realized',
            r'initially.*believed.*later.*understood',
            r'at first.*seemed.*but then.*appeared'
        ]
        
        text_lower = text.lower()
        dynamic_count = sum(len(re.findall(pattern, text_lower)) for pattern in dynamic_perspective_patterns)
        
        return dynamic_count / len(text.split()) if text.split() else 0
    
    def _calculate_meta_tom_complexity(self, text):
        """Calculate theory of mind about theory of mind"""
        meta_tom_patterns = [
            r'understand.*thinking',
            r'realize.*belief',
            r'recognize.*feeling',
            r'aware.*mental'
        ]
        
        text_lower = text.lower()
        meta_tom = sum(len(re.findall(pattern, text_lower)) for pattern in meta_tom_patterns)
        
        return meta_tom / len(text.split()) if text.split() else 0
    
    def _calculate_cognitive_interference(self, text):
        """Calculate cognitive interference from conflicting information"""
        interference_patterns = [
            r'said.*but.*meant',
            r'appeared.*but.*actually',
            r'seemed.*however.*really',
            r'looked.*although.*felt'
        ]
        
        text_lower = text.lower()
        interference = sum(len(re.findall(pattern, text_lower)) for pattern in interference_patterns)
        
        return interference / len(text.split()) if text.split() else 0
    
    def _extract_entities(self, text):
        """Extract entities from text"""
        if not self.nlp:
            return re.findall(r'\b[A-Z][a-z]+\b', text)
        
        try:
            doc = self.nlp(text)
            return [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]
        except:
            return []
    
    def _count_mental_state_verbs(self, text):
        """Count mental state verbs in text"""
        text_lower = text.lower()
        count = 0
        for verbs in self.mental_state_verbs.values():
            count += sum(1 for verb in verbs if verb in text_lower)
        return count
    
    def _count_uncertainty_markers(self, text):
        """Count uncertainty markers in text"""
        text_lower = text.lower()
        count = 0
        for markers in self.uncertainty_markers.values():
            count += sum(1 for marker in markers if marker in text_lower)
        return count
    
    def _get_empty_multiplicative_features(self, text_type):
        """Return empty multiplicative features"""
        return {
            f'{text_type}_Entity_Density': 0,
            f'{text_type}_Causal_Depth': 0,
            f'{text_type}_Uncertainty_Level': 0,
            f'{text_type}_Temporal_Complexity': 0,
            f'{text_type}_Cognitive_Load_Index': 0,
            f'{text_type}_Mental_State_Depth': 0,
            f'{text_type}_Entity_Switches': 0,
            f'{text_type}_Emotional_Transitions': 0,
            f'{text_type}_Mental_State_Interaction': 0,
            f'{text_type}_Inference_Chain_Depth': 0,
            f'{text_type}_Contradiction_Resolution': 0,
            f'{text_type}_Perspective_Shifts': 0,
            f'{text_type}_Inference_Complexity': 0
        }
    
    def _get_empty_dynamic_features(self, text_type):
        """Return empty dynamic features"""
        return {
            f'{text_type}_Complexity_Gradient': 0,
            f'{text_type}_Complexity_Variance': 0,
            f'{text_type}_Complexity_Peaks': 0,
            f'{text_type}_Working_Memory_Load': 0,
            f'{text_type}_Attention_Switching_Cost': 0
        }
    
    def _get_empty_meta_features(self, text_type):
        """Return empty meta features"""
        return {
            f'{text_type}_Recursive_Mental_State_Depth': 0,
            f'{text_type}_Meta_Uncertainty_Index': 0,
            f'{text_type}_Dynamic_Perspective_Complexity': 0,
            f'{text_type}_Meta_ToM_Complexity': 0,
            f'{text_type}_Cognitive_Interference_Score': 0
        }
    
    def analyze_comprehensive(self, text, text_type='Story'):
        """Comprehensive analysis combining all next-generation metrics"""
        features = {}
        
        # Multiplicative complexity
        features.update(self.analyze_multiplicative_complexity(text, text_type))
        
        # Dynamic complexity
        features.update(self.analyze_dynamic_complexity(text, text_type))
        
        # Meta-cognitive complexity
        features.update(self.analyze_meta_cognitive_complexity(text, text_type))
        
        return features

def analyze_text_next_gen(text, text_type='Story'):
    """Wrapper function for next-generation analysis"""
    if not hasattr(analyze_text_next_gen, 'analyzer'):
        analyze_text_next_gen.analyzer = NextGenToMAnalyzer()
    
    analyzer = analyze_text_next_gen.analyzer
    return analyzer.analyze_comprehensive(text, text_type)

if __name__ == "__main__":
    # Test the analyzer
    test_text = """John thinks that Mary believes Tom knows she is lying about the surprise party. 
    However, Tom actually has no idea what Mary thinks, and he's completely confused by her behavior. 
    Meanwhile, Sarah wonders if John realizes that his plan might backfire, but she's not sure 
    whether she should tell him what she suspects."""
    
    analyzer = NextGenToMAnalyzer()
    features = analyzer.analyze_comprehensive(test_text, 'Test')
    
    print("Next-Generation ToM Analysis Results:")
    for key, value in sorted(features.items()):
        print(f"  {key}: {value:.4f}")
