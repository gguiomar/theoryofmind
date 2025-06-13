#!/usr/bin/env python3
"""
Comprehensive Analysis Runner
Applies advanced metrics and runs evolutionary optimization to find universal correlations
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    dependencies = {
        'spacy': False,
        'textstat': False,
        'sentence_transformers': False,
        'deap': False,
        'optuna': False,
        'sklearn': False
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
            print(f"✓ {dep} available")
        except ImportError:
            print(f"✗ {dep} not available")
    
    return dependencies

def run_basic_optimization():
    """Run basic optimization with available packages"""
    print("\n" + "="*80)
    print("RUNNING BASIC OPTIMIZATION WITH AVAILABLE PACKAGES")
    print("="*80)
    
    try:
        from evolutionary_metric_optimization import MetricOptimizer
        
        # Initialize optimizer
        optimizer = MetricOptimizer()
        
        # Run basic optimization methods
        results = []
        
        # 1. Exhaustive search (always available)
        print("\n1. Running exhaustive search...")
        exhaustive_result = optimizer.exhaustive_search(max_metrics=3, max_combinations=1000)
        if exhaustive_result:
            results.append(exhaustive_result)
        
        # 2. Random search (always available)
        print("\n2. Running random search...")
        random_result = optimizer.random_search(n_trials=500, max_metrics=5)
        if random_result:
            results.append(random_result)
        
        # 3. Try genetic algorithm if DEAP available
        try:
            import deap
            print("\n3. Running genetic algorithm...")
            ga_result = optimizer.genetic_algorithm_optimization(population_size=30, generations=20, max_metrics=5)
            if ga_result:
                results.append(ga_result)
        except ImportError:
            print("\n3. Skipping genetic algorithm (DEAP not available)")
        
        # 4. Try Bayesian optimization if Optuna available
        try:
            import optuna
            print("\n4. Running Bayesian optimization...")
            bayesian_result = optimizer.bayesian_optimization(n_trials=50, max_metrics=5)
            if bayesian_result:
                results.append(bayesian_result)
        except ImportError:
            print("\n4. Skipping Bayesian optimization (Optuna not available)")
        
        # Compare results
        if results:
            print("\n" + "="*80)
            print("OPTIMIZATION RESULTS")
            print("="*80)
            
            # Sort by fitness
            results.sort(key=lambda x: x['fitness'], reverse=True)
            
            for i, result in enumerate(results):
                print(f"\n{i+1}. {result['method'].upper()}")
                print(f"   Fitness: {result['fitness']:.4f}")
                print(f"   Metrics: {len(result['metrics'])}")
                print(f"   Top metrics: {result['metrics'][:3]}")
                if 'significance_rate' in result:
                    print(f"   Significance rate: {result['significance_rate']:.3f}")
            
            # Test best result
            best_result = results[0]
            print(f"\n" + "="*80)
            print("TESTING BEST RESULT")
            print("="*80)
            
            optimizer.test_metric_combination(best_result['metrics'], 
                                           best_result.get('combination_type', 'weighted'))
            
            # Save results
            results_df = pd.DataFrame(results)
            results_df.to_csv('basic_optimization_results.csv', index=False)
            print(f"\nResults saved to: basic_optimization_results.csv")
            
            return results
        else:
            print("No optimization results obtained.")
            return []
            
    except Exception as e:
        print(f"Error running optimization: {e}")
        return []

def test_advanced_metrics():
    """Test advanced metrics with available packages"""
    print("\n" + "="*80)
    print("TESTING ADVANCED METRICS")
    print("="*80)
    
    try:
        from advanced_linguistic_metrics import AdvancedLinguisticAnalyzer
        
        # Test text
        test_text = """John thinks that Mary believes Tom knows she is lying about the surprise party. 
        However, Tom actually has no idea what Mary thinks, and he's completely confused by her behavior. 
        Meanwhile, Sarah wonders if John realizes that his plan might backfire, but she's not sure 
        whether she should tell him what she suspects."""
        
        # Initialize analyzer
        analyzer = AdvancedLinguisticAnalyzer()
        
        # Test analysis
        features = analyzer.analyze_comprehensive(test_text, 'Test')
        
        print(f"Successfully extracted {len(features)} advanced features:")
        
        # Group features by type
        feature_groups = {
            'Syntactic': [k for k in features.keys() if any(x in k for x in ['Parse', 'Clause', 'Yngve', 'Arc'])],
            'Semantic': [k for k in features.keys() if any(x in k for x in ['Verb_Argument', 'Predicate', 'Entity_Relation', 'Semantic_Mental'])],
            'Embedding': [k for k in features.keys() if any(x in k for x in ['Embedding', 'Coherence', 'Trajectory', 'PCA'])],
            'Readability': [k for k in features.keys() if any(x in k for x in ['Flesch', 'Gunning', 'SMOG', 'Dale_Chall'])]
        }
        
        for group, group_features in feature_groups.items():
            if group_features:
                print(f"\n{group} features ({len(group_features)}):")
                for feature in group_features[:3]:  # Show first 3
                    print(f"  {feature}: {features[feature]:.4f}")
                if len(group_features) > 3:
                    print(f"  ... and {len(group_features) - 3} more")
        
        return True
        
    except Exception as e:
        print(f"Error testing advanced metrics: {e}")
        return False

def run_simple_correlation_test():
    """Run a simple correlation test with existing metrics"""
    print("\n" + "="*80)
    print("SIMPLE CORRELATION TEST WITH EXISTING METRICS")
    print("="*80)
    
    try:
        # Load dataset
        df = pd.read_csv('./dataset_v12_final_universal.csv')
        df.columns = df.columns.str.strip()
        
        # Clean data
        df_clean = df[df['ABILITY'].notna()].copy()
        df_clean['Main_Category'] = df_clean['ABILITY'].str.split(':').str[0].str.strip()
        df_clean['Main_Category'] = df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
        
        print(f"Loaded dataset with {len(df_clean)} rows and {len(df_clean.columns)} columns")
        
        # Test a few promising metric combinations from our previous analysis
        test_combinations = [
            ['Story_Entity_Density', 'Story_Uncertainty_Level', 'Story_Causal_Depth'],
            ['Final_Universal_Metric_Multiplicative'],
            ['Story_Cognitive_Load_Index', 'Story_Mental_State_Interaction'],
            ['Story_Entity_Count', 'Story_Discourse_causal_Count', 'Story_Hedging_Count']
        ]
        
        from evolutionary_metric_optimization import MetricOptimizer
        optimizer = MetricOptimizer()
        
        print(f"\nTesting {len(test_combinations)} promising combinations:")
        
        best_fitness = 0
        best_combination = None
        
        for i, combination in enumerate(test_combinations):
            print(f"\n{i+1}. Testing: {combination}")
            
            # Filter to available metrics
            available_metrics = [m for m in combination if m in df_clean.columns]
            
            if available_metrics:
                fitness, sig_rate, avg_corr = optimizer.evaluate_metric_combination(available_metrics)
                print(f"   Fitness: {fitness:.4f}, Significance rate: {sig_rate:.3f}, Avg |correlation|: {avg_corr:.3f}")
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_combination = available_metrics
            else:
                print(f"   No available metrics in this combination")
        
        if best_combination:
            print(f"\n" + "="*60)
            print("BEST COMBINATION DETAILED TEST")
            print("="*60)
            optimizer.test_metric_combination(best_combination)
        
        return True
        
    except Exception as e:
        print(f"Error in correlation test: {e}")
        return False

def main():
    """Main comprehensive analysis function"""
    print("="*80)
    print("COMPREHENSIVE THEORY OF MIND ANALYSIS")
    print("="*80)
    
    # Check dependencies
    dependencies = check_dependencies()
    
    # Run simple correlation test first
    print("\n" + "="*80)
    print("PHASE 1: SIMPLE CORRELATION TEST")
    print("="*80)
    
    correlation_success = run_simple_correlation_test()
    
    # Test advanced metrics if possible
    print("\n" + "="*80)
    print("PHASE 2: ADVANCED METRICS TEST")
    print("="*80)
    
    advanced_success = test_advanced_metrics()
    
    # Run optimization
    print("\n" + "="*80)
    print("PHASE 3: OPTIMIZATION")
    print("="*80)
    
    optimization_results = run_basic_optimization()
    
    # Summary
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"Dependencies available: {sum(dependencies.values())}/{len(dependencies)}")
    print(f"Simple correlation test: {'✓ Success' if correlation_success else '✗ Failed'}")
    print(f"Advanced metrics test: {'✓ Success' if advanced_success else '✗ Failed'}")
    print(f"Optimization results: {len(optimization_results) if optimization_results else 0} methods completed")
    
    if optimization_results:
        best_result = optimization_results[0]
        print(f"\nBest optimization result:")
        print(f"  Method: {best_result['method']}")
        print(f"  Fitness: {best_result['fitness']:.4f}")
        print(f"  Metrics: {len(best_result['metrics'])}")
        
        # Check if we achieved 70B significance
        if best_result['fitness'] > 0.8:
            print(f"  🎉 HIGH FITNESS ACHIEVED! Potential breakthrough.")
        elif best_result['fitness'] > 0.6:
            print(f"  🔥 GOOD FITNESS! Strong progress toward universal correlation.")
        else:
            print(f"  📈 Moderate fitness. Room for improvement.")
    
    print(f"\nNext steps:")
    if not advanced_success:
        print(f"  1. Install missing dependencies for advanced metrics")
        print(f"  2. pip install spacy sentence-transformers textstat")
        print(f"  3. python -m spacy download en_core_web_sm")
    
    if not optimization_results or len(optimization_results) < 3:
        print(f"  4. Install optimization packages: pip install deap optuna")
    
    print(f"  5. Run full analysis: python apply_advanced_linguistic_metrics.py")
    print(f"  6. Run evolutionary optimization: python evolutionary_metric_optimization.py")
    
    return {
        'dependencies': dependencies,
        'correlation_success': correlation_success,
        'advanced_success': advanced_success,
        'optimization_results': optimization_results
    }

if __name__ == "__main__":
    results = main()
