import pandas as pd
import numpy as np
from isanlp_rst.parser import Parser

class RSTFeatureExtractor:
    """
    Extract RST features relevant for Theory of Mind (ToM) analysis.
    
    Core ToM-relevant relations tracked:
    - attribution: Who said/thought what (perspective tracking)
    - causal: Cause-effect chains for mental state reasoning
    - explanation: Why characters act/think certain ways
    """
    def __init__(self, version='gumrrg', cuda_device=-1):
        """
        Initialize RST parser.
        
        Args:
            version: Model version ('gumrrg', 'rstdt', or 'rstreebank')
            cuda_device: GPU device number (use -1 for CPU)
        """
        self.parser = Parser(
            hf_model_name='tchewik/isanlp_rst_v3', 
            hf_model_version=version, 
            cuda_device=cuda_device
        )
    
    def extract_features(self, text):
        """Extract core RST features from a single text."""
        try:
            # Parse the text
            result = self.parser(text)
            
            if not result or 'rst' not in result or not result['rst']:
                return self._get_empty_features()
            
            # Get the first RST tree (there might be multiple for long texts)
            rst_tree = result['rst'][0]
            
            # Extract core features
            features = {
                'num_edus': self._count_edus(rst_tree),
                'tree_depth': self._calculate_depth(rst_tree),
            }
            
            # Extract relation type counts
            relation_counts = self._count_relation_types(rst_tree)
            features.update(relation_counts)
            
            return features
            
        except:
            return self._get_empty_features()
    
    def _count_edus(self, rst_tree):
        """Count the number of EDUs in the RST tree."""
        # Count all nodes that have 'elementary' relation (these are the EDUs)
        edu_count = 0
        self._count_elementary_nodes(rst_tree, edu_count_ref := [0])
        return edu_count_ref[0] if edu_count_ref[0] > 0 else 1
    
    def _count_elementary_nodes(self, node, count_ref):
        """Count nodes with 'elementary' relation (EDUs)."""
        if node is None:
            return
        
        # Check if this is an elementary node (EDU)
        if hasattr(node, 'relation') and node.relation.lower() == 'elementary':
            count_ref[0] += 1
        
        # Recurse on children
        if hasattr(node, 'left') and node.left:
            self._count_elementary_nodes(node.left, count_ref)
        
        if hasattr(node, 'right') and node.right:
            self._count_elementary_nodes(node.right, count_ref)
    
    def _count_leaves(self, node):
        """Count leaf nodes (EDUs) in the tree - kept for compatibility."""
        count_ref = [0]
        self._count_elementary_nodes(node, count_ref)
        return count_ref[0]
    
    def _calculate_depth(self, node, current_depth=0):
        """Calculate maximum depth of the RST tree."""
        if not hasattr(node, 'left') and not hasattr(node, 'right'):
            # Leaf node
            return current_depth
        
        max_depth = current_depth
        
        if hasattr(node, 'left') and node.left:
            max_depth = max(max_depth, self._calculate_depth(node.left, current_depth + 1))
        
        if hasattr(node, 'right') and node.right:
            max_depth = max(max_depth, self._calculate_depth(node.right, current_depth + 1))
        
        return max_depth
    
    def _count_relation_types(self, node):
        """Count occurrences of each relation type."""
        # Core ToM-relevant relations only
        relation_types = [
            'attribution',  # Who said/thought what (perspective tracking)
            'causal',       # Cause-effect chains for mental state reasoning
            'explanation'   # Why something happened (understanding motivations)
        ]
        
        counts = {f'rel_{rel}': 0 for rel in relation_types}
        self._traverse_relations(node, counts)
        
        return counts
    
    def _traverse_relations(self, node, counts):
        """Traverse tree and count relations."""
        # Check if this node has a relation
        if hasattr(node, 'relation'):
            rel_type = node.relation.lower().replace('-', '_').replace(' ', '_')
            
            # Skip 'elementary' as it's just EDU markers, not actual relations
            if rel_type == 'elementary':
                pass
            else:
                key = f'rel_{rel_type}'
                if key in counts:
                    counts[key] += 1
        
        # Recurse on children
        if hasattr(node, 'left') and node.left:
            self._traverse_relations(node.left, counts)
        
        if hasattr(node, 'right') and node.right:
            self._traverse_relations(node.right, counts)
    
    def _get_empty_features(self):
        """Return empty feature dict when parsing fails."""
        features = {
            'num_edus': 0,
            'tree_depth': 0
        }
        
        # Add all relation type counts as 0
        relation_types = [
            'attribution', 'causal', 'explanation'
        ]
        
        for rel in relation_types:
            features[f'rel_{rel}'] = 0
        
        return features

def process_stories_from_csv(csv_path, output_path, model_version='gumrrg', cuda_device=-1):
    """
    Process all stories from the CSV and add RST features.
    
    Args:
        csv_path: Path to input CSV
        output_path: Path to output CSV with RST features
        model_version: RST model version to use
        cuda_device: GPU device (-1 for CPU)
    """
    # Read your CSV
    df = pd.read_csv(csv_path)
    
    # Initialize extractor
    extractor = RSTFeatureExtractor(version=model_version, cuda_device=cuda_device)
    
    # Extract features for each story
    rst_features_list = []
    
    for idx, row in df.iterrows():
        story = row['STORY']
        if (idx + 1) % 100 == 0:
            print(f"Processing story {idx + 1}/{len(df)}...")
        
        features = extractor.extract_features(story)
        rst_features_list.append(features)
    
    # Convert to dataframe
    rst_df = pd.DataFrame(rst_features_list)
    
    # Merge with original data
    df_with_features = pd.concat([df, rst_df], axis=1)
    
    # Save enhanced dataset
    df_with_features.to_csv(output_path, index=False)
    
    # Print summary statistics
    print("\nRST Feature Summary:")
    print(f"Average EDUs per story: {rst_df['num_edus'].mean():.2f}")
    print(f"Average tree depth: {rst_df['tree_depth'].mean():.2f}")
    print(f"Stories successfully parsed: {(rst_df['num_edus'] > 0).sum()}/{len(rst_df)}")
    
    # Show distribution of EDUs
    print("\nEDU distribution:")
    edu_counts = rst_df['num_edus'].value_counts().sort_index()
    for edu_count, freq in edu_counts.head(10).items():
        print(f"  {edu_count} EDUs: {freq} stories")
    
    print("\nMost common relation types:")
    rel_cols = [col for col in rst_df.columns if col.startswith('rel_')]
    rel_sums = rst_df[rel_cols].sum().sort_values(ascending=False)
    for rel, count in rel_sums.head(5).items():
        if count > 0:
            print(f"  {rel}: {count}")
    
    # Highlight ToM-relevant relations
    print("\nKey ToM-relevant relations:")
    tom_relations = ['rel_attribution', 'rel_causal', 'rel_explanation']
    for rel in tom_relations:
        if rel in rst_df.columns:
            count = rst_df[rel].sum()
            print(f"  {rel}: {count}")

# Example usage
if __name__ == "__main__":
    # Process your CSV with different model versions if needed
    input_csv = "../dataset.csv"
    output_csv = "dataset_RST.csv"
    
    # Try with 'gumrrg' model (best for diverse text types including stories)
    process_stories_from_csv(
        input_csv, 
        output_csv,
        model_version='gumrrg',  # or 'rstdt' or 'rstreebank'
        cuda_device=-1  # Use CPU, change to 0 for GPU
    )
