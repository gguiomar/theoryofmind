#!/usr/bin/env python3
"""
Expanded NLP Theory of Mind Metrics
Using multiple advanced NLP packages to extract comprehensive ToM-related features
"""

import pandas as pd
import numpy as np
import spacy
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
from textblob import TextBlob
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class ExpandedToMAnalyzer:
    """
    Comprehensive Theory of Mind analyzer using multiple NLP packages
    """
    
    def __init__(self):
        """Initialize all NLP tools and resources"""
        print("Initializing expanded ToM analyzer...")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize NLTK tools
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except:
            self.sentiment_analyzer = None
            self.lemmatizer = None
            self.stop_words = set()
        
        # Define comprehensive mental state vocabularies
        self.mental_state_categories = {
            'cognitive_verbs': {
                'basic': ['think', 'know', 'believe', 'understand', 'realize', 'remember', 'forget'],
                'advanced': ['contemplate', 'ponder', 'deliberate', 'comprehend', 'perceive', 'conceive'],
                'epistemic': ['assume', 'suppose', 'presume', 'infer', 'deduce', 'conclude'],
                'metacognitive': ['wonder', 'doubt', 'question', 'puzzle', 'confuse', 'clarify']
            },
            'emotional_verbs': {
                'basic': ['feel', 'love', 'hate', 'fear', 'worry', 'hope', 'enjoy'],
                'complex': ['admire', 'despise', 'cherish', 'loathe', 'appreciate', 'resent'],
                'social': ['empathize', 'sympathize', 'relate', 'connect', 'bond', 'alienate']
            },
            'volitional_verbs': {
                'intention': ['want', 'desire', 'wish', 'intend', 'plan', 'aim'],
                'decision': ['decide', 'choose', 'select', 'prefer', 'opt', 'determine'],
                'motivation': ['strive', 'aspire', 'yearn', 'crave', 'seek', 'pursue']
            },
            'perceptual_verbs': {
                'visual': ['see', 'look', 'watch', 'observe', 'notice', 'spot'],
                'auditory': ['hear', 'listen', 'sound', 'echo', 'whisper', 'shout'],
                'general': ['sense', 'detect', 'perceive', 'experience', 'encounter']
            }
        }
        
        # Flatten all mental state verbs
        self.all_mental_verbs = set()
        for category in self.mental_state_categories.values():
            for subcategory in category.values():
                self.all_mental_verbs.update(subcategory)
        
        # Theory of Mind specific patterns
        self.tom_patterns = {
            'false_belief': [
                r'thinks?\s+.*\s+but\s+(actually|really|in\s+fact)',
                r'believes?\s+.*\s+but\s+(actually|really|in\s+fact)',
                r'thought\s+.*\s+was\s+.*\s+but\s+it\s+was',
                r'believed\s+.*\s+contained\s+.*\s+but'
            ],
            'perspective_taking': [
                r'from\s+.*\s+point\s+of\s+view',
                r'in\s+.*\s+opinion',
                r'.*\s+thinks?\s+that',
                r'.*\s+believes?\s+that',
                r'according\s+to\s+.*'
            ],
            'mental_state_attribution': [
                r'he\s+(thinks?|believes?|knows?|feels?|wants?)',
                r'she\s+(thinks?|believes?|knows?|feels?|wants?)',
                r'they\s+(think|believe|know|feel|want)'
            ],
            'recursive_thinking': [
                r'thinks?\s+that\s+.*\s+(thinks?|believes?|knows?)',
                r'believes?\s+that\s+.*\s+(thinks?|believes?|knows?)',
                r'knows?\s+that\s+.*\s+(thinks?|believes?|knows?)'
            ]
        }
        
        # Discourse markers and connectives
        self.discourse_markers = {
            'causal': ['because', 'since', 'as', 'due to', 'owing to', 'therefore', 'thus', 'hence'],
            'contrast': ['but', 'however', 'although', 'though', 'whereas', 'while', 'nevertheless'],
            'temporal': ['when', 'while', 'before', 'after', 'during', 'until', 'since', 'then'],
            'conditional': ['if', 'unless', 'provided', 'assuming', 'suppose', 'in case']
        }
        
        print("✓ Expanded ToM analyzer initialized successfully")
    
    def analyze_text_comprehensive(self, text, text_type='Story'):
        """Comprehensive analysis using all available NLP tools"""
        if not text or str(text).strip() == '' or str(text).lower() == 'nan':
            return self._get_empty_features(text_type)
        
        text = str(text).strip()
        features = {}
        
        # 1. Basic text statistics
        features.update(self._analyze_basic_stats(text, text_type))
        
        # 2. Readability metrics
        features.update(self._analyze_readability(text, text_type))
        
        # 3. Sentiment and emotion analysis
        features.update(self._analyze_sentiment_emotion(text, text_type))
        
        # 4. Mental state verb analysis (expanded)
        features.update(self._analyze_mental_state_verbs_expanded(text, text_type))
        
        # 5. Theory of Mind pattern analysis
        features.update(self._analyze_tom_patterns(text, text_type))
        
        # 6. Discourse and coherence analysis
        features.update(self._analyze_discourse_coherence(text, text_type))
        
        # 7. Syntactic complexity (enhanced)
        features.update(self._analyze_syntactic_complexity_enhanced(text, text_type))
        
        # 8. Semantic complexity
        features.update(self._analyze_semantic_complexity(text, text_type))
        
        # 9. Pragmatic features
        features.update(self._analyze_pragmatic_features(text, text_type))
        
        # 10. Narrative structure analysis
        features.update(self._analyze_narrative_structure(text, text_type))
        
        return features
    
    def _analyze_basic_stats(self, text, text_type):
        """Enhanced basic text statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Character-level stats
        char_count = len(text)
        alpha_char_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        punct_count = sum(1 for c in text if c in '.,!?;:')
        
        # Word-level stats
        word_count = len(words)
        unique_words = len(set(word.lower() for word in words))
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Sentence-level stats
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return {
            f'{text_type}_Char_Count': char_count,
            f'{text_type}_Alpha_Char_Ratio': alpha_char_count / char_count if char_count > 0 else 0,
            f'{text_type}_Digit_Ratio': digit_count / char_count if char_count > 0 else 0,
            f'{text_type}_Punct_Ratio': punct_count / char_count if char_count > 0 else 0,
            f'{text_type}_Word_Count_Enhanced': word_count,
            f'{text_type}_Unique_Word_Ratio': unique_words / word_count if word_count > 0 else 0,
            f'{text_type}_Avg_Word_Length_Enhanced': avg_word_length,
            f'{text_type}_Sentence_Count_Enhanced': sentence_count,
            f'{text_type}_Avg_Sentence_Length': avg_sentence_length
        }
    
    def _analyze_readability(self, text, text_type):
        """Comprehensive readability analysis"""
        try:
            flesch_ease = flesch_reading_ease(text)
            flesch_grade = flesch_kincaid_grade(text)
            ari = automated_readability_index(text)
        except:
            flesch_ease = flesch_grade = ari = 0
        
        return {
            f'{text_type}_Flesch_Reading_Ease': flesch_ease,
            f'{text_type}_Flesch_Kincaid_Grade': flesch_grade,
            f'{text_type}_Automated_Readability_Index': ari,
            f'{text_type}_Readability_Composite': (flesch_ease + flesch_grade + ari) / 3
        }
    
    def _analyze_sentiment_emotion(self, text, text_type):
        """Sentiment and emotional analysis"""
        features = {}
        
        # NLTK VADER sentiment
        if self.sentiment_analyzer:
            try:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                features.update({
                    f'{text_type}_Sentiment_Positive': sentiment_scores['pos'],
                    f'{text_type}_Sentiment_Negative': sentiment_scores['neg'],
                    f'{text_type}_Sentiment_Neutral': sentiment_scores['neu'],
                    f'{text_type}_Sentiment_Compound': sentiment_scores['compound']
                })
            except:
                features.update({
                    f'{text_type}_Sentiment_Positive': 0,
                    f'{text_type}_Sentiment_Negative': 0,
                    f'{text_type}_Sentiment_Neutral': 0,
                    f'{text_type}_Sentiment_Compound': 0
                })
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            features.update({
                f'{text_type}_TextBlob_Polarity': blob.sentiment.polarity,
                f'{text_type}_TextBlob_Subjectivity': blob.sentiment.subjectivity
            })
        except:
            features.update({
                f'{text_type}_TextBlob_Polarity': 0,
                f'{text_type}_TextBlob_Subjectivity': 0
            })
        
        # Emotional word counting
        emotion_words = {
            'joy': ['happy', 'joy', 'pleased', 'delighted', 'cheerful', 'glad'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'sorrowful'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed'],
            'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened']
        }
        
        text_lower = text.lower()
        for emotion, words in emotion_words.items():
            count = sum(1 for word in words if word in text_lower)
            features[f'{text_type}_Emotion_{emotion.title()}_Count'] = count
        
        return features
    
    def _analyze_mental_state_verbs_expanded(self, text, text_type):
        """Expanded mental state verb analysis"""
        text_lower = text.lower()
        features = {}
        
        # Count by category and subcategory
        for category, subcategories in self.mental_state_categories.items():
            category_total = 0
            for subcategory, verbs in subcategories.items():
                count = sum(1 for verb in verbs if verb in text_lower)
                features[f'{text_type}_MS_{category}_{subcategory}_Count'] = count
                category_total += count
            features[f'{text_type}_MS_{category}_Total'] = category_total
        
        # Overall mental state verb density
        total_ms_verbs = sum(1 for verb in self.all_mental_verbs if verb in text_lower)
        word_count = len(text.split())
        features[f'{text_type}_MS_Verb_Density'] = total_ms_verbs / word_count if word_count > 0 else 0
        
        return features
    
    def _analyze_tom_patterns(self, text, text_type):
        """Theory of Mind specific pattern analysis"""
        features = {}
        
        for pattern_type, patterns in self.tom_patterns.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, text, re.IGNORECASE))
            features[f'{text_type}_ToM_{pattern_type}_Count'] = count
        
        # Complex ToM reasoning patterns
        complex_patterns = [
            r'if\s+.*\s+(thinks?|believes?|knows?)',
            r'(thinks?|believes?|knows?)\s+.*\s+would',
            r'pretend\s+.*\s+(thinks?|believes?|knows?)',
            r'(thinks?|believes?|knows?)\s+.*\s+because'
        ]
        
        complex_count = 0
        for pattern in complex_patterns:
            complex_count += len(re.findall(pattern, text, re.IGNORECASE))
        features[f'{text_type}_ToM_Complex_Reasoning'] = complex_count
        
        return features
    
    def _analyze_discourse_coherence(self, text, text_type):
        """Discourse markers and coherence analysis"""
        features = {}
        text_lower = text.lower()
        
        # Count discourse markers by type
        for marker_type, markers in self.discourse_markers.items():
            count = sum(1 for marker in markers if marker in text_lower)
            features[f'{text_type}_Discourse_{marker_type}_Count'] = count
        
        # Coherence metrics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 1:
            # Lexical cohesion (word overlap between adjacent sentences)
            overlaps = []
            for i in range(len(sentences) - 1):
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[i + 1].lower().split())
                if words1 and words2:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    overlaps.append(overlap)
            
            features[f'{text_type}_Lexical_Cohesion'] = np.mean(overlaps) if overlaps else 0
        else:
            features[f'{text_type}_Lexical_Cohesion'] = 0
        
        return features
    
    def _analyze_syntactic_complexity_enhanced(self, text, text_type):
        """Enhanced syntactic complexity using spaCy"""
        features = {}
        
        if not self.nlp:
            return {f'{text_type}_Syntactic_Enhanced_Error': 1}
        
        try:
            doc = self.nlp(text)
            
            # Dependency tree depth and complexity
            depths = []
            clause_counts = []
            
            for sent in doc.sents:
                # Calculate dependency tree depth
                def get_depth(token, depth=0):
                    if not list(token.children):
                        return depth
                    return max(get_depth(child, depth + 1) for child in token.children)
                
                if sent.root:
                    depths.append(get_depth(sent.root))
                
                # Count clauses (SBAR, S tags approximated by specific patterns)
                clause_count = len([token for token in sent if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl']])
                clause_counts.append(clause_count)
            
            features.update({
                f'{text_type}_Max_Dependency_Depth_Enhanced': max(depths) if depths else 0,
                f'{text_type}_Avg_Dependency_Depth_Enhanced': np.mean(depths) if depths else 0,
                f'{text_type}_Total_Clauses_Enhanced': sum(clause_counts),
                f'{text_type}_Avg_Clauses_Per_Sentence': np.mean(clause_counts) if clause_counts else 0
            })
            
            # POS tag diversity
            pos_tags = [token.pos_ for token in doc]
            unique_pos = len(set(pos_tags))
            features[f'{text_type}_POS_Diversity'] = unique_pos / len(pos_tags) if pos_tags else 0
            
            # Named entity analysis
            entities = [ent.label_ for ent in doc.ents]
            features[f'{text_type}_Entity_Count_Enhanced'] = len(doc.ents)
            features[f'{text_type}_Entity_Type_Diversity'] = len(set(entities)) if entities else 0
            
        except Exception as e:
            features[f'{text_type}_Syntactic_Enhanced_Error'] = 1
        
        return features
    
    def _analyze_semantic_complexity(self, text, text_type):
        """Semantic complexity analysis"""
        features = {}
        
        # Abstract vs concrete word ratio
        abstract_words = ['idea', 'concept', 'thought', 'belief', 'opinion', 'theory', 'principle']
        concrete_words = ['table', 'chair', 'book', 'car', 'house', 'tree', 'water']
        
        text_lower = text.lower()
        abstract_count = sum(1 for word in abstract_words if word in text_lower)
        concrete_count = sum(1 for word in concrete_words if word in text_lower)
        
        features[f'{text_type}_Abstract_Word_Count'] = abstract_count
        features[f'{text_type}_Concrete_Word_Count'] = concrete_count
        features[f'{text_type}_Abstract_Concrete_Ratio'] = abstract_count / (concrete_count + 1)
        
        # Semantic density (content words vs function words)
        if self.nlp:
            try:
                doc = self.nlp(text)
                content_words = [token for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
                function_words = [token for token in doc if token.pos_ in ['ADP', 'CONJ', 'DET', 'PRON']]
                
                features[f'{text_type}_Content_Word_Count'] = len(content_words)
                features[f'{text_type}_Function_Word_Count'] = len(function_words)
                features[f'{text_type}_Semantic_Density'] = len(content_words) / len(doc) if len(doc) > 0 else 0
            except:
                features[f'{text_type}_Semantic_Density_Error'] = 1
        
        return features
    
    def _analyze_pragmatic_features(self, text, text_type):
        """Pragmatic and communicative features"""
        features = {}
        
        # Question types
        question_patterns = {
            'wh_questions': r'\b(what|when|where|who|why|how)\b.*\?',
            'yes_no_questions': r'\b(is|are|was|were|do|does|did|can|could|will|would)\b.*\?',
            'tag_questions': r'.*,\s*(isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|doesn\'t|didn\'t)\s+(it|he|she|they)\?'
        }
        
        for q_type, pattern in question_patterns.items():
            count = len(re.findall(pattern, text, re.IGNORECASE))
            features[f'{text_type}_Question_{q_type}_Count'] = count
        
        # Politeness markers
        politeness_markers = ['please', 'thank you', 'excuse me', 'sorry', 'pardon']
        politeness_count = sum(1 for marker in politeness_markers if marker in text.lower())
        features[f'{text_type}_Politeness_Markers'] = politeness_count
        
        # Hedging and uncertainty
        hedging_words = ['maybe', 'perhaps', 'possibly', 'probably', 'might', 'could', 'seem', 'appear']
        hedging_count = sum(1 for word in hedging_words if word in text.lower())
        features[f'{text_type}_Hedging_Count'] = hedging_count
        
        return features
    
    def _analyze_narrative_structure(self, text, text_type):
        """Narrative structure and story elements"""
        features = {}
        
        # Temporal progression markers
        temporal_markers = ['first', 'then', 'next', 'after', 'before', 'finally', 'meanwhile', 'suddenly']
        temporal_count = sum(1 for marker in temporal_markers if marker in text.lower())
        features[f'{text_type}_Temporal_Progression'] = temporal_count
        
        # Character introduction patterns
        character_patterns = [
            r'there\s+was\s+a\s+\w+',
            r'once\s+upon\s+a\s+time',
            r'a\s+\w+\s+named\s+\w+',
            r'meet\s+\w+'
        ]
        
        character_intro_count = 0
        for pattern in character_patterns:
            character_intro_count += len(re.findall(pattern, text, re.IGNORECASE))
        features[f'{text_type}_Character_Introduction'] = character_intro_count
        
        # Dialogue markers
        dialogue_count = text.count('"') + text.count("'")
        features[f'{text_type}_Dialogue_Markers'] = dialogue_count
        
        return features
    
    def _get_empty_features(self, text_type):
        """Return empty feature dict when analysis fails"""
        # Return a comprehensive set of zero features
        empty_features = {}
        
        # Basic stats
        basic_features = ['Char_Count', 'Alpha_Char_Ratio', 'Digit_Ratio', 'Punct_Ratio', 
                         'Word_Count_Enhanced', 'Unique_Word_Ratio', 'Avg_Word_Length_Enhanced',
                         'Sentence_Count_Enhanced', 'Avg_Sentence_Length']
        
        # Readability
        readability_features = ['Flesch_Reading_Ease', 'Flesch_Kincaid_Grade', 
                               'Automated_Readability_Index', 'Readability_Composite']
        
        # Sentiment
        sentiment_features = ['Sentiment_Positive', 'Sentiment_Negative', 'Sentiment_Neutral',
                             'Sentiment_Compound', 'TextBlob_Polarity', 'TextBlob_Subjectivity']
        
        all_feature_names = basic_features + readability_features + sentiment_features
        
        for feature in all_feature_names:
            empty_features[f'{text_type}_{feature}'] = 0
        
        return empty_features

def analyze_text_expanded_tom(text, text_type='Story'):
    """Wrapper function for expanded ToM analysis"""
    if not hasattr(analyze_text_expanded_tom, 'analyzer'):
        analyze_text_expanded_tom.analyzer = ExpandedToMAnalyzer()
    
    analyzer = analyze_text_expanded_tom.analyzer
    return analyzer.analyze_text_comprehensive(text, text_type)

if __name__ == "__main__":
    # Test the analyzer
    test_text = """John thinks that Mary believes Tom knows she is lying about the surprise party. 
    However, Tom actually has no idea what Mary thinks, and he's completely confused by her behavior. 
    Meanwhile, Sarah wonders if John realizes that his plan might backfire."""
    
    analyzer = ExpandedToMAnalyzer()
    features = analyzer.analyze_text_comprehensive(test_text, 'Test')
    
    print("Expanded ToM Analysis Results:")
    for key, value in sorted(features.items()):
        print(f"  {key}: {value}")
