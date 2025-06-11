# Comprehensive Theory of Mind (ToM) Analysis Pipeline

This folder contains a unified analysis pipeline that combines all Theory of Mind analysis components into a single, comprehensive workflow.

## Overview

The pipeline integrates four key analysis dimensions:

1. **Idea Density Analysis (DEPID)** - Measures propositional density in stories
2. **RST Discourse Analysis** - Extracts rhetorical structure features  
3. **Question Complexity Analysis** - Multi-dimensional question difficulty assessment
4. **Answer Distinctiveness Analysis** - Measures uniqueness across answer options

## Files

### Core Components

- **`tom_analyzers.py`** - Contains all analyzer classes consolidated from individual analysis repos
- **`main.py`** - Main pipeline script that runs all analyses in sequence
- **`README.md`** - This documentation file

### Analyzer Classes

#### `IdeaDensityAnalyzer`
- **Purpose**: Calculates Dependency-based Propositional Idea Density (DEPID)
- **Input**: Story text
- **Output**: `Idea_Density`, `Word_Count`
- **Method**: Uses syntactic dependencies to count propositions

#### `RSTAnalyzer` 
- **Purpose**: Extracts Rhetorical Structure Theory features
- **Input**: Story text
- **Output**: `RST_EDUs`, `RST_Tree_Depth`, `RST_attribution`, `RST_causal`, `RST_explanation`
- **Method**: Parses discourse structure using pre-trained RST model

#### `QuestionComplexityAnalyzer`
- **Purpose**: Multi-dimensional question complexity assessment
- **Input**: Question text
- **Output**: 20+ complexity features across 4 dimensions
- **Dimensions**:
  - Syntactic: dependency depth, clause count
  - Semantic: lexical diversity, entity count
  - ToM-specific: mental state verbs, perspective markers
  - Reasoning: logical operators, temporal references

#### `AnswerDistinctivenessAnalyzer`
- **Purpose**: Measures distinctiveness across multiple choice options
- **Input**: Four answer options (A, B, C, D)
- **Output**: 15+ distinctiveness features
- **Methods**:
  - Semantic: sentence embedding similarity
  - Lexical: word overlap (Jaccard similarity)
  - Length: variance in response lengths
  - Syntactic: POS pattern diversity

## Usage

### Prerequisites

Ensure you have the `tom` conda environment activated with all required packages:

```bash
conda activate tom
```

Required packages:
- pandas, numpy, scikit-learn
- spacy (with en_core_web_sm model)
- sentence-transformers
- ideadensity
- isanlp_rst
- tqdm

### Running the Analysis

1. **Navigate to the Joined_Analysis folder**:
   ```bash
   cd Joined_Analysis
   ```

2. **Run the comprehensive analysis**:
   ```bash
   python main.py
   ```

3. **Monitor progress**: The script will show progress bars and status updates for each analysis stage.

### Input Requirements

The script expects `../dataset.csv` (in the parent directory) with the following columns:
- `STORY` - Story text for idea density and RST analysis
- `QUESTION` - Question text for complexity analysis  
- `OPTION-A`, `OPTION-B`, `OPTION-C`, `OPTION-D` - Answer options for distinctiveness analysis

### Output

**File**: `comprehensive_tom_analysis.csv`

**Structure**: Original dataset + all analysis features organized in logical groups:

1. **Original columns** (unchanged)
2. **Idea Density features**: `Idea_Density`, `Word_Count`
3. **RST features**: `RST_EDUs`, `RST_Tree_Depth`, `RST_attribution`, `RST_causal`, `RST_explanation`
4. **Question Complexity features**: `Question_Complexity_Score`, `Q_*` (20+ features)
5. **Answer Distinctiveness features**: `Answer_Distinctiveness_Score`, `A_*` (15+ features)

## Feature Descriptions

### Idea Density Features
- **`Idea_Density`**: Propositions per 100 words (DEPID score)
- **`Word_Count`**: Total word count in story

### RST Features
- **`RST_EDUs`**: Number of Elementary Discourse Units
- **`RST_Tree_Depth`**: Maximum depth of discourse tree
- **`RST_attribution`**: Count of attribution relations (who said/thought what)
- **`RST_causal`**: Count of causal relations (cause-effect chains)
- **`RST_explanation`**: Count of explanation relations (why something happened)

### Question Complexity Features
- **`Question_Complexity_Score`**: Overall complexity score (0-1 scale)
- **`Q_Syntactic_Complexity`**: Syntactic difficulty score
- **`Q_Semantic_Complexity`**: Semantic difficulty score  
- **`Q_ToM_Complexity`**: Theory of Mind specific complexity
- **`Q_Reasoning_Complexity`**: Logical reasoning complexity
- **Additional Q_* features**: 15+ detailed linguistic metrics

### Answer Distinctiveness Features
- **`Answer_Distinctiveness_Score`**: Overall distinctiveness score (0-1 scale)
- **`A_Semantic_Distinctiveness`**: Semantic uniqueness across options
- **`A_Lexical_Distinctiveness`**: Word-level uniqueness
- **`A_Length_Distinctiveness`**: Length variation across options
- **`A_Syntactic_Distinctiveness`**: Grammatical pattern diversity
- **Additional A_* features**: 10+ detailed distinctiveness metrics

## Performance Notes

- **Processing time**: ~1-2 seconds per sample (depends on text length)
- **Memory usage**: ~2-4GB RAM (mainly for RST and sentence transformer models)
- **GPU support**: RST analysis can use GPU (set `cuda_device=0` in RSTAnalyzer)
- **Batch processing**: Processes samples sequentially with progress tracking

## Error Handling

The pipeline includes robust error handling:
- Individual sample failures don't stop the entire process
- Missing columns are handled gracefully
- Empty/invalid text inputs return appropriate default values
- Detailed error messages and warnings are provided

## Extending the Pipeline

To add new analysis components:

1. **Create analyzer class** in `tom_analyzers.py` following the existing pattern
2. **Add initialization** in `main.py` 
3. **Add analysis call** in the processing loop
4. **Update feature grouping** in the column organization section

## Example Output Statistics

For a typical ToM dataset (2,860 samples):

```
Idea Density Analysis:     2860/2860 samples
RST Analysis:              2860/2860 samples  
Question Complexity:       2860/2860 samples
Answer Distinctiveness:    2860/2860 samples

Key Metric Statistics:
Idea_Density:
  Mean: 0.395 ± 0.071
  Range: [0.237, 0.587]

RST_EDUs:
  Mean: 21.2 ± 8.4
  Range: [5, 45]

Question_Complexity_Score:
  Mean: 0.234 ± 0.156
  Range: [0.000, 0.892]

Answer_Distinctiveness_Score:
  Mean: 0.678 ± 0.142
  Range: [0.234, 0.945]
```

This comprehensive analysis provides a rich feature set for Theory of Mind research, enabling detailed investigation of story complexity, question difficulty, and answer quality across multiple linguistic and cognitive dimensions.
