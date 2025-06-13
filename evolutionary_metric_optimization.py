#!/usr/bin/env python3
"""
Evolutionary Optimization for Theory of Mind Metric Combinations
Uses genetic algorithms and Bayesian optimization to find optimal metric combinations
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("Warning: DEAP not available. Genetic algorithm disabled.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Bayesian optimization disabled.")

try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not available. Gaussian process optimization disabled.")

import itertools
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import random
import time

class MetricOptimizer:
    """
    Evolutionary optimizer for finding optimal metric combinations
    """
    
    def __init__(self, dataset_path='dataset_v12_final_universal.csv'):
        """Initialize the optimizer"""
        print("Initializing Metric Optimizer...")
        
        # Load dataset
        self.df = pd.read_csv(dataset_path)
        self.df.columns = self.df.columns.str.strip()
        
        # Clean data
        self.df_clean = self.df[self.df['ABILITY'].notna()].copy()
        self.df_clean['Main_Category'] = self.df_clean['ABILITY'].str.split(':').str[0].str.strip()
        self.df_clean['Main_Category'] = self.df_clean['Main_Category'].replace('Non-Literal Communication', 'NLC')
        
        # Get performance matrix
        self.performance_matrix = self._calculate_performance_matrix()
        
        # Get available metrics
        self.available_metrics = self._get_available_metrics()
        
        print(f"✓ Loaded dataset with {len(self.df_clean)} rows")
        print(f"✓ Found {len(self.available_metrics)} available metrics")
        print(f"✓ Performance matrix shape: {self.performance_matrix.shape}")
    
    def _calculate_performance_matrix(self):
        """Calculate model performance matrix"""
        categories = self.df_clean['Main_Category'].unique()
        
        # Human performance data
        human_performance = {
            'Emotion': 86.4,
            'Desire': 90.4,
            'Intention': 82.2,
            'Knowledge': 89.3,
            'Belief': 89.0,
            'NLC': 86.1
        }
        
        # Model columns
        model_cols = [col for col in self.df.columns if any(model in col for model in 
                      ['meta_llama', 'Qwen', 'allenai', 'mistralai', 'microsoft', 'internlm'])]
        
        model_display_names = {
            'meta_llama_Llama_3.1_70B_Instruct': 'Llama 3.1 70B',
            'Qwen_Qwen2.5_32B_Instruct': 'Qwen 2.5 32B',
            'allenai_OLMo_2_1124_13B_Instruct': 'OLMo 13B',
            'mistralai_Mistral_7B_Instruct_v0.3': 'Mistral 7B',
            'microsoft_Phi_3_mini_4k_instruct': 'Phi-3 Mini',
            'internlm_internlm2_5_1_8b_chat': 'InternLM 1.8B'
        }
        
        def calculate_accuracy(df, model_col, category):
            subset = df[df['Main_Category'] == category]
            if len(subset) == 0:
                return 0
            correct = (subset[model_col] == subset['ANSWER']).sum()
            return (correct / len(subset) * 100)
        
        # Create performance matrix
        performance_matrix = pd.DataFrame(index=categories)
        performance_matrix['Human'] = [human_performance.get(cat, np.nan) for cat in categories]
        
        for model in model_cols:
            if model in model_display_names:
                display_name = model_display_names[model]
                performance_matrix[display_name] = [
                    calculate_accuracy(self.df_clean, model, cat) for cat in categories
                ]
        
        return performance_matrix
    
    def _get_available_metrics(self):
        """Get list of available metrics for optimization"""
        # Exclude non-metric columns
        exclude_cols = [
            'Unnamed: 0', 'ABILITY', 'TASK', 'INDEX', 'STORY', 'QUESTION', 
            'OPTION-A', 'OPTION-B', 'OPTION-C', 'OPTION-D', 'ANSWER',
            'Main_Category', 'Volition', 'Cognition', 'Emotion'
        ]
        
        # Get model columns
        model_cols = [col for col in self.df.columns if any(model in col for model in 
                      ['meta_llama', 'Qwen', 'allenai', 'mistralai', 'microsoft', 'internlm'])]
        
        # Get all numeric columns that aren't models or excluded
        metric_cols = []
        for col in self.df.columns:
            if col not in exclude_cols and col not in model_cols:
                if self.df[col].dtype in ['int64', 'float64']:
                    # Check if column has sufficient variance
                    if self.df[col].std() > 1e-10 and self.df[col].notna().sum() > len(self.df) * 0.5:
                        metric_cols.append(col)
        
        return metric_cols
    
    def evaluate_metric_combination(self, metrics, combination_type='weighted', weights=None):
        """Evaluate a specific metric combination"""
        if not metrics or len(metrics) == 0:
            return 0, 0, 0  # fitness, significance_rate, avg_correlation
        
        try:
            # Filter to available metrics
            available_metrics = [m for m in metrics if m in self.available_metrics and m in self.df_clean.columns]
            
            if len(available_metrics) == 0:
                return 0, 0, 0
            
            # Create combined metric
            combined_metric = self._create_combined_metric(available_metrics, combination_type, weights)
            
            if combined_metric is None:
                return 0, 0, 0
            
            # Calculate correlations
            correlations, p_values = self._calculate_correlations(combined_metric)
            
            # Calculate fitness metrics
            significance_rate = sum(p < 0.05 for p in p_values) / len(p_values)
            avg_abs_correlation = np.mean([abs(c) for c in correlations])
            
            # Special bonus for 70B significance
            llama_70b_significant = False
            if len(p_values) >= 2:  # Assuming Llama 70B is index 1
                llama_70b_significant = p_values[1] < 0.05
            
            # Fitness function (maximize significance rate and correlation strength)
            fitness = (significance_rate * 0.6 + 
                      avg_abs_correlation * 0.3 + 
                      (0.1 if llama_70b_significant else 0))
            
            return fitness, significance_rate, avg_abs_correlation
            
        except Exception as e:
            print(f"Error evaluating combination: {e}")
            return 0, 0, 0
    
    def _create_combined_metric(self, metrics, combination_type='weighted', weights=None):
        """Create combined metric from individual metrics"""
        try:
            # Get metric data
            metric_data = self.df_clean[metrics].fillna(0)
            
            if len(metric_data) == 0:
                return None
            
            # Standardize metrics
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(metric_data)
            standardized_df = pd.DataFrame(standardized_data, columns=metrics, index=metric_data.index)
            
            if combination_type == 'weighted':
                if weights is None:
                    weights = np.ones(len(metrics)) / len(metrics)
                combined = np.sum(standardized_df * weights, axis=1)
                
            elif combination_type == 'multiplicative':
                # Add 1 to avoid zeros, then multiply
                shifted_data = standardized_df - standardized_df.min() + 1
                combined = np.prod(shifted_data, axis=1)
                
            elif combination_type == 'pca':
                if len(metrics) > 1:
                    pca = PCA(n_components=1)
                    combined = pca.fit_transform(standardized_df).flatten()
                else:
                    combined = standardized_df.iloc[:, 0]
                    
            else:  # simple sum
                combined = np.sum(standardized_df, axis=1)
            
            return combined
            
        except Exception as e:
            print(f"Error creating combined metric: {e}")
            return None
    
    def _calculate_correlations(self, combined_metric):
        """Calculate correlations between combined metric and performance"""
        categories = self.performance_matrix.index.tolist()
        subjects = self.performance_matrix.columns.tolist()
        
        # Calculate metric averages by category
        metric_by_category = self.df_clean.groupby('Main_Category')[combined_metric.index].apply(
            lambda x: combined_metric[x.index].mean()
        )
        
        correlations = []
        p_values = []
        
        for subject in subjects:
            performance_values = self.performance_matrix[subject].values
            metric_values = [metric_by_category.get(cat, 0) for cat in categories]
            
            # Calculate correlation
            valid_mask = ~(pd.isna(metric_values) | pd.isna(performance_values))
            if sum(valid_mask) >= 3:
                x = np.array(metric_values)[valid_mask]
                y = np.array(performance_values)[valid_mask]
                
                if np.std(x) > 0 and np.std(y) > 0:
                    corr, p_val = stats.pearsonr(x, y)
                    correlations.append(corr)
                    p_values.append(p_val)
                else:
                    correlations.append(0)
                    p_values.append(1)
            else:
                correlations.append(0)
                p_values.append(1)
        
        return correlations, p_values
    
    def genetic_algorithm_optimization(self, population_size=100, generations=50, max_metrics=10):
        """Use genetic algorithm to find optimal metric combinations"""
        if not DEAP_AVAILABLE:
            print("DEAP not available. Skipping genetic algorithm.")
            return None
        
        print(f"Starting genetic algorithm optimization...")
        print(f"Population size: {population_size}, Generations: {generations}")
        
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Attribute generator: binary selection of metrics
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                        toolbox.attr_bool, len(self.available_metrics))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate_individual(individual):
            # Convert binary representation to metric list
            selected_metrics = [self.available_metrics[i] for i, selected in enumerate(individual) 
                              if selected and i < len(self.available_metrics)]
            
            # Limit number of metrics
            if len(selected_metrics) > max_metrics:
                selected_metrics = selected_metrics[:max_metrics]
            
            fitness, _, _ = self.evaluate_metric_combination(selected_metrics, 'weighted')
            return (fitness,)
        
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        population = toolbox.population(n=population_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        
        # Run genetic algorithm
        population, logbook = algorithms.eaSimple(
            population, toolbox, cxpb=0.5, mutpb=0.2, 
            ngen=generations, stats=stats, verbose=True
        )
        
        # Get best individual
        best_individual = tools.selBest(population, 1)[0]
        best_metrics = [self.available_metrics[i] for i, selected in enumerate(best_individual) 
                       if selected and i < len(self.available_metrics)]
        
        print(f"Best GA solution: {len(best_metrics)} metrics")
        print(f"Best fitness: {best_individual.fitness.values[0]:.4f}")
        
        return {
            'metrics': best_metrics,
            'fitness': best_individual.fitness.values[0],
            'method': 'genetic_algorithm'
        }
    
    def bayesian_optimization(self, n_trials=100, max_metrics=10):
        """Use Bayesian optimization to find optimal metric combinations"""
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Skipping Bayesian optimization.")
            return None
        
        print(f"Starting Bayesian optimization with {n_trials} trials...")
        
        def objective(trial):
            # Select metrics using trial suggestions
            selected_metrics = []
            for i, metric in enumerate(self.available_metrics[:50]):  # Limit to first 50 metrics
                if trial.suggest_categorical(f'metric_{i}', [True, False]):
                    selected_metrics.append(metric)
            
            # Limit number of metrics
            if len(selected_metrics) > max_metrics:
                selected_metrics = selected_metrics[:max_metrics]
            
            # Select combination type
            combination_type = trial.suggest_categorical('combination_type', 
                                                       ['weighted', 'multiplicative', 'pca'])
            
            fitness, _, _ = self.evaluate_metric_combination(selected_metrics, combination_type)
            return fitness
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best trial
        best_trial = study.best_trial
        
        # Extract best metrics
        best_metrics = []
        for i, metric in enumerate(self.available_metrics[:50]):
            if f'metric_{i}' in best_trial.params and best_trial.params[f'metric_{i}']:
                best_metrics.append(metric)
        
        best_combination_type = best_trial.params.get('combination_type', 'weighted')
        
        print(f"Best Bayesian solution: {len(best_metrics)} metrics")
        print(f"Best fitness: {best_trial.value:.4f}")
        print(f"Best combination type: {best_combination_type}")
        
        return {
            'metrics': best_metrics,
            'fitness': best_trial.value,
            'combination_type': best_combination_type,
            'method': 'bayesian_optimization'
        }
    
    def exhaustive_search(self, max_metrics=5, max_combinations=10000):
        """Exhaustive search for small metric combinations"""
        print(f"Starting exhaustive search for combinations up to {max_metrics} metrics...")
        
        best_result = None
        best_fitness = 0
        combinations_tested = 0
        
        # Test combinations of different sizes
        for size in range(1, max_metrics + 1):
            print(f"Testing combinations of size {size}...")
            
            # Generate all combinations of this size
            for combination in itertools.combinations(self.available_metrics, size):
                if combinations_tested >= max_combinations:
                    break
                
                fitness, sig_rate, avg_corr = self.evaluate_metric_combination(list(combination))
                combinations_tested += 1
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_result = {
                        'metrics': list(combination),
                        'fitness': fitness,
                        'significance_rate': sig_rate,
                        'avg_correlation': avg_corr,
                        'method': 'exhaustive_search'
                    }
                
                if combinations_tested % 1000 == 0:
                    print(f"  Tested {combinations_tested} combinations, best fitness: {best_fitness:.4f}")
            
            if combinations_tested >= max_combinations:
                break
        
        print(f"Exhaustive search complete. Tested {combinations_tested} combinations.")
        if best_result:
            print(f"Best fitness: {best_result['fitness']:.4f}")
            print(f"Best metrics: {len(best_result['metrics'])} metrics")
        
        return best_result
    
    def random_search(self, n_trials=1000, max_metrics=10):
        """Random search baseline"""
        print(f"Starting random search with {n_trials} trials...")
        
        best_result = None
        best_fitness = 0
        
        for trial in range(n_trials):
            # Random number of metrics
            n_metrics = random.randint(1, min(max_metrics, len(self.available_metrics)))
            
            # Random selection of metrics
            selected_metrics = random.sample(self.available_metrics, n_metrics)
            
            # Random combination type
            combination_type = random.choice(['weighted', 'multiplicative', 'pca'])
            
            fitness, sig_rate, avg_corr = self.evaluate_metric_combination(selected_metrics, combination_type)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_result = {
                    'metrics': selected_metrics,
                    'fitness': fitness,
                    'significance_rate': sig_rate,
                    'avg_correlation': avg_corr,
                    'combination_type': combination_type,
                    'method': 'random_search'
                }
            
            if trial % 100 == 0:
                print(f"  Trial {trial}, best fitness: {best_fitness:.4f}")
        
        print(f"Random search complete. Best fitness: {best_fitness:.4f}")
        return best_result
    
    def run_comprehensive_optimization(self):
        """Run all optimization methods and compare results"""
        print("="*80)
        print("COMPREHENSIVE METRIC OPTIMIZATION")
        print("="*80)
        
        results = []
        
        # 1. Exhaustive search (small combinations)
        print("\n1. Running exhaustive search...")
        exhaustive_result = self.exhaustive_search(max_metrics=4, max_combinations=5000)
        if exhaustive_result:
            results.append(exhaustive_result)
        
        # 2. Random search
        print("\n2. Running random search...")
        random_result = self.random_search(n_trials=1000, max_metrics=8)
        if random_result:
            results.append(random_result)
        
        # 3. Genetic algorithm
        print("\n3. Running genetic algorithm...")
        ga_result = self.genetic_algorithm_optimization(population_size=50, generations=30, max_metrics=8)
        if ga_result:
            results.append(ga_result)
        
        # 4. Bayesian optimization
        print("\n4. Running Bayesian optimization...")
        bayesian_result = self.bayesian_optimization(n_trials=100, max_metrics=8)
        if bayesian_result:
            results.append(bayesian_result)
        
        # Compare results
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS COMPARISON")
        print("="*80)
        
        if results:
            # Sort by fitness
            results.sort(key=lambda x: x['fitness'], reverse=True)
            
            for i, result in enumerate(results):
                print(f"\n{i+1}. {result['method'].upper()}")
                print(f"   Fitness: {result['fitness']:.4f}")
                print(f"   Metrics: {len(result['metrics'])}")
                print(f"   Top metrics: {result['metrics'][:5]}")
                if 'significance_rate' in result:
                    print(f"   Significance rate: {result['significance_rate']:.3f}")
                if 'combination_type' in result:
                    print(f"   Combination type: {result['combination_type']}")
            
            # Test best result
            best_result = results[0]
            print(f"\n" + "="*80)
            print("TESTING BEST RESULT")
            print("="*80)
            
            self.test_metric_combination(best_result['metrics'], 
                                       best_result.get('combination_type', 'weighted'))
            
            return results
        else:
            print("No optimization results obtained.")
            return []
    
    def test_metric_combination(self, metrics, combination_type='weighted'):
        """Test a specific metric combination and show detailed results"""
        print(f"\nTesting combination of {len(metrics)} metrics:")
        for i, metric in enumerate(metrics):
            print(f"  {i+1}. {metric}")
        
        # Create combined metric
        combined_metric = self._create_combined_metric(metrics, combination_type)
        
        if combined_metric is None:
            print("Failed to create combined metric.")
            return
        
        # Calculate correlations
        correlations, p_values = self._calculate_correlations(combined_metric)
        
        # Show results
        subjects = self.performance_matrix.columns.tolist()
        
        print(f"\nCorrelation Results:")
        significant_count = 0
        for i, subject in enumerate(subjects):
            if i < len(correlations):
                sig_marker = "*" if p_values[i] < 0.05 else ""
                if p_values[i] < 0.05:
                    significant_count += 1
                
                if subject == 'Llama 3.1 70B':
                    if p_values[i] < 0.05:
                        print(f"  🎉 {subject}: r = {correlations[i]:.3f}, p = {p_values[i]:.3f}{sig_marker} ⭐ BREAKTHROUGH!")
                    else:
                        print(f"  🔍 {subject}: r = {correlations[i]:.3f}, p = {p_values[i]:.3f}{sig_marker}")
                else:
                    print(f"     {subject}: r = {correlations[i]:.3f}, p = {p_values[i]:.3f}{sig_marker}")
        
        significance_rate = significant_count / len(subjects)
        avg_abs_correlation = np.mean([abs(c) for c in correlations])
        
        print(f"\nSummary:")
        print(f"  Significant correlations: {significant_count}/{len(subjects)} ({significance_rate*100:.1f}%)")
        print(f"  Average |correlation|: {avg_abs_correlation:.3f}")
        print(f"  70B significant: {'YES ⭐' if len(p_values) > 1 and p_values[1] < 0.05 else 'NO'}")

def main():
    """Main optimization function"""
    print("Starting comprehensive metric optimization...")
    
    # Initialize optimizer
    optimizer = MetricOptimizer()
    
    # Run comprehensive optimization
    results = optimizer.run_comprehensive_optimization()
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('optimization_results.csv', index=False)
        print(f"\nResults saved to: optimization_results.csv")
    
    return results

if __name__ == "__main__":
    results = main()
