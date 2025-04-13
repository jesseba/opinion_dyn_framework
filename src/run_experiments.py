#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

# Import from other modules
from model import SUPPORTER, UNDECIDED, OPPOSITION
from experiments import (
    run_grassroots_vs_establishment_experiment,
    run_network_battleground_experiment,
    run_timing_experiment,
    run_intervention_pattern_experiment,
    run_blitz_vs_sustained_experiment,
    run_critical_mass_experiment,
    run_intervention_sensitivity_experiment,
    run_targeted_seeding_experiment,
    run_opponent_composition_experiment,
    run_transition_rate_asymmetry_experiment
)
from utils import save_experiment_results, load_experiment_results, find_experiment_results

# Set better visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Set random seed for reproducibility
np.random.seed(42)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run opinion dynamics experiments')
    
    # Experiment selection
    parser.add_argument('--experiment', type=str, choices=[
        'grassroots_vs_establishment', 
        'network_battleground', 
        'timing', 
        'intervention_pattern',
        'blitz_vs_sustained',
        'critical_mass',
        'intervention_sensitivity',
        'targeted_seeding',
        'opponent_composition',
        'transition_rate_asymmetry',
        'all'
    ], default='intervention_pattern', help='Experiment to run')
    
    # Common parameters
    parser.add_argument('--n_nodes', type=int, default=1000, help='Number of nodes in the network')
    parser.add_argument('--shock_duration', type=int, default=20, help='Duration of shock in time steps')
    parser.add_argument('--total_steps', type=int, default=150, help='Total simulation duration')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of trials to run')
    parser.add_argument('--lambda_s', type=float, default=0.12, help='Base rate towards supporter state')
    parser.add_argument('--lambda_o', type=float, default=0.12, help='Base rate towards opposition state')
    
    # Specific parameters for certain experiments
    parser.add_argument('--vary_intensity', action='store_true', help='Vary intervention intensity for blitz vs sustained experiment')
    
    # Parameters for critical mass experiment
    parser.add_argument('--initial_min', type=float, default=0.05, help='Minimum initial supporter proportion for critical mass experiment')
    parser.add_argument('--initial_max', type=float, default=0.95, help='Maximum initial supporter proportion for critical mass experiment')
    parser.add_argument('--initial_steps', type=int, default=19, help='Number of steps between min and max initial proportion')

    # Parameters for targeted seeding experiment, opponent composition experiment, and transition rate asymmetry experiment
    parser.add_argument('--undecided_ratio_steps', type=int, default=11, 
                   help='Number of steps between 0 and 1 for undecided ratio in opponent composition experiment')

    parser.add_argument('--lambda_ratio_steps', type=int, default=11,
                   help='Number of lambda ratio steps in transition rate asymmetry experiment')

    parser.add_argument('--initial_supporter_percent', type=float, default=0.3,
                   help='Initial percentage of supporters for targeted seeding and opponent composition experiments')

    
    # Parameters for intervention sensitivity experiment
    parser.add_argument('--intervention_type', type=str, choices=['establishment', 'grassroots'], 
                       default='establishment', help='Type of intervention for sensitivity experiment')
    parser.add_argument('--sensitivity_steps', type=int, default=10, 
                       help='Number of steps between min and max initial proportion for sensitivity experiment')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='../results', help='Directory to save results')
    parser.add_argument('--no_save', action='store_true', help='Do not save results')
    parser.add_argument('--no_plots', action='store_true', help='Do not display plots')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Configure experiments to run
    experiments_to_run = []
    if args.experiment == 'all':
        experiments_to_run = [
            'grassroots_vs_establishment', 
            'network_battleground', 
            'timing', 
            'intervention_pattern',
            'blitz_vs_sustained',
            'critical_mass',
            'intervention_sensitivity',
            'targeted_seeding',
            'opponent_composition',
            'transition_rate_asymmetry'
        ]
    else:
        experiments_to_run = [args.experiment]
    
    # Configure common experiment parameters
    common_params = {
        'n_nodes': args.n_nodes,
        'shock_duration': args.shock_duration,
        'total_steps': args.total_steps,
        'num_trials': args.num_trials,
        'lambda_s': args.lambda_s,
        'lambda_o': args.lambda_o
    }
    
    # Create output directory
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run each experiment
    for experiment in experiments_to_run:
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT: {experiment.upper()}")
        print(f"{'='*80}")
        
        # Configure experiment-specific parameters and run
        config = common_params.copy()
        results = None
        
        if experiment == 'grassroots_vs_establishment':
            results = run_grassroots_vs_establishment_experiment(**config)
            
        elif experiment == 'network_battleground':
            results = run_network_battleground_experiment(**config)
            
        elif experiment == 'timing':
            results = run_timing_experiment(**config)
            
        elif experiment == 'intervention_pattern':
            # This experiment doesn't use shock_duration
            del config['shock_duration']
            results = run_intervention_pattern_experiment(**config)
            
        elif experiment == 'blitz_vs_sustained':
            # This experiment doesn't use shock_duration but adds vary_intensity
            del config['shock_duration']
            config['vary_intensity'] = args.vary_intensity
            results = run_blitz_vs_sustained_experiment(**config)
            
        elif experiment == 'critical_mass':
            # This experiment doesn't use shock_duration
            del config['shock_duration']
            # Get the default range for initial supporter proportions
            initial_supporter_range = np.linspace(args.initial_min, args.initial_max, args.initial_steps)
            config['initial_supporter_range'] = initial_supporter_range
            results = run_critical_mass_experiment(**config)

        elif experiment == 'targeted_seeding':
            # This experiment doesn't use shock_duration
            del config['shock_duration']
            config['initial_supporter_percent'] = args.initial_supporter_percent
            results = run_targeted_seeding_experiment(**config)
            
        elif experiment == 'opponent_composition':
            # This experiment doesn't use shock_duration
            del config['shock_duration']
            undecided_ratio_range = np.linspace(0.0, 1.0, args.undecided_ratio_steps)
            config['initial_supporter_percent'] = args.initial_supporter_percent
            config['undecided_ratio_range'] = undecided_ratio_range
            results = run_opponent_composition_experiment(**config)
            
        elif experiment == 'transition_rate_asymmetry':
            # This experiment doesn't use shock_duration
            del config['shock_duration']
            # This experiment uses lambda_base instead of lambda_s and lambda_o
            del config['lambda_s']
            del config['lambda_o']
            lambda_ratios = np.logspace(-1, 1, args.lambda_ratio_steps)  # 0.1 to 10 in log space
            config['lambda_base'] = args.lambda_s  # Use lambda_s as the base rate
            config['lambda_ratios'] = lambda_ratios
            results = run_transition_rate_asymmetry_experiment(**config)
            
        elif experiment == 'intervention_sensitivity':
            # This experiment uses shock_duration, so we keep it (no need to delete)
            # Get the default range for initial supporter proportions and set default intervention type
            initial_supporter_range = np.linspace(args.initial_min, args.initial_max, args.sensitivity_steps)
            config['initial_supporter_range'] = initial_supporter_range
            config['intervention_type'] = args.intervention_type
            results = run_intervention_sensitivity_experiment(**config)
        
        # Save results
        if not args.no_save and results is not None:
            filepath = save_experiment_results(experiment, results, config, args.output_dir)
            print(f"Results saved to: {filepath}")
            
            # Save a summary file for easy reference
            summary_path = os.path.join(args.output_dir, f"{experiment}_summary.json")
            with open(summary_path, 'w') as f:
                # Create a JSON-serializable copy of the config
                json_config = {}
                for key, value in config.items():
                    # Convert NumPy arrays to lists
                    if isinstance(value, np.ndarray):
                        json_config[key] = value.tolist()
                    # Convert NumPy data types to Python types
                    elif isinstance(value, (np.int32, np.int64)):
                        json_config[key] = int(value)
                    elif isinstance(value, (np.float32, np.float64)):
                        json_config[key] = float(value)
                    else:
                        json_config[key] = value
                        
                summary = {
                    'config': json_config,
                    'filepath': filepath,
                    'timestamp': os.path.basename(filepath).split('_')[1]
                }
                json.dump(summary, f, indent=2)
    
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main() 