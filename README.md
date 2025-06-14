# Theory of Mind Complexity Analysis

## Overview

This repository contains a comprehensive analysis of **Theory of Mind (ToM) complexity features** that are universally predictive across both human subjects and Large Language Models (LLMs). The analysis identifies 292 universal features significantly correlated with ToM performance.

## Key Findings

🎯 **292 universal features** identified across 7 subjects (humans + 6 LLMs)  
📊 **1,692 significant correlations** discovered with robust statistical validation  
🔬 **Cross-cognitive validity** demonstrated for both biological and artificial intelligence  

## Repository Structure

```
theoryofmind/
├── data/
│   ├── final/
│   │   └── dataset_v14_comprehensive_advanced.csv    # 468 features × 2,860 samples
│   └── original/
│       └── dataset.csv                               # Original ToM dataset
├── scripts/
│   ├── final_correlation_analysis.py                # Main analysis script
│   ├── comprehensive_tom_analysis.py                # Feature extraction engine
│   ├── performance_matrix_pdf.py                    # Performance data extraction
│   └── advanced_tom_metrics.py                      # Advanced metrics
├── results/
│   ├── correlations/
│   │   ├── all_correlations.csv                     # Complete correlation results
│   │   ├── significant_correlations.csv             # Significant correlations only
│   │   └── universal_features.csv                   # Universal features list
│   ├── visualizations/
│   │   ├── correlation_matrix.pdf                   # Publication-ready heatmap
│   │   └── correlation_matrix.png                   # High-resolution visualization
│   └── reports/
│       └── final_analysis_summary.md                # Comprehensive analysis report
├── docs/
│   └── comprehensive_analysis_summary.md            # Technical documentation
└── archive/                                         # Historical files (167 items)
```

## Quick Start

### Run the Analysis
```bash
cd scripts
python final_correlation_analysis.py
```

### View Results
- **Correlation Matrix**: `results/visualizations/correlation_matrix.pdf`
- **Universal Features**: `results/correlations/universal_features.csv`
- **Full Report**: `results/reports/final_analysis_summary.md`

## Key Results

### Top Universal Features (Significant Across All 7 Subjects)

#### Question Complexity Features
- `Question_Automated_Readability_Index.1` - Readability complexity
- `Question_Avg_Clauses_Per_Sentence` - Syntactic complexity  
- `Question_Causal_Depth` - Semantic reasoning depth
- `Question_Dale_Chall` - Vocabulary difficulty

#### Story Semantic Features
- `Story_Max_Distance_To_Centroid` - Semantic dispersion (r=-0.494 for humans)
- `Story_PCA_Cumulative_Variance` - Semantic dimensionality
- `Story_PCA_First_Component` - Primary semantic component

#### Story Mental State Features
- `Story_MS_Verb_Density` - Mental state verb frequency
- `Story_Pronoun_Count` - Reference complexity
- `Story_MS_perceptual_verbs_Total` - Perceptual verb count

### Strongest Correlations
1. **Story_Digit_Ratio** (Phi3_Mini): r = -0.689
2. **Story_Dialogue_Markers** (InternLM_1.8B): r = 0.663
3. **Story_Temporal_Markers** (Human): r = -0.419

## Subjects Analyzed

### Human Performance
- Performance data from established ToM benchmarks
- 6 categories: Belief, Intention, Emotion, Knowledge, Desire, NLC

### LLM Performance  
- **Llama 3.1 70B** - Meta's flagship model
- **Qwen 2.5 32B** - Alibaba's advanced model
- **OLMo 13B** - Allen Institute's open model
- **Mistral 7B** - Mistral AI's efficient model
- **Phi-3 Mini** - Microsoft's compact model
- **InternLM 1.8B** - Shanghai AI Lab's model

## Technical Details

### Feature Engineering
- **468 total features** extracted using 20+ advanced NLP packages
- **95.2% package compatibility** achieved with modern Python environment
- **Robust processing** of 2,860 samples across 6 ToM categories

### Statistical Methodology
- **Pearson correlations** with significance testing
- **FDR correction** for multiple comparisons (α = 0.05)
- **Cross-subject validation** across cognitive architectures

### Advanced NLP Tools Used
- **spaCy** - Dependency parsing, POS tagging
- **Stanza** - Advanced syntactic analysis  
- **sentence-transformers** - Semantic embeddings
- **textstat** - Readability metrics
- **networkx** - Graph-based features
- **scikit-learn** - Dimensionality reduction

## Applications

### Research
- **Benchmark creation** for ToM complexity assessment
- **Cross-cognitive studies** of reasoning difficulty
- **Feature selection** for predictive modeling

### Practical Use
- **Educational tools** with controlled complexity
- **Model evaluation** across ToM dimensions  
- **Assessment instruments** for cognitive abilities

## Requirements

```bash
# Core dependencies
pip install pandas numpy scipy matplotlib seaborn scikit-learn
pip install statsmodels

# Advanced NLP (see requirements_advanced.txt for full list)
pip install spacy stanza sentence-transformers textstat networkx
```

## Citation

If you use this work, please cite:

```bibtex
@misc{tom_complexity_analysis_2024,
  title={Universal Theory of Mind Complexity Features: Cross-Cognitive Analysis},
  author={[Your Name]},
  year={2024},
  note={Comprehensive analysis of ToM complexity across humans and LLMs}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about this analysis or collaboration opportunities, please open an issue or contact [your email].

---

**Last Updated**: June 2024  
**Analysis Version**: v14 (Comprehensive Advanced)  
**Total Features**: 468  
**Universal Features**: 292  
**Subjects Analyzed**: 7 (1 human + 6 LLMs)
