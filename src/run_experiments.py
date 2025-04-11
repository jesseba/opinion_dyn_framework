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
    run_blitz_vs_sustained_experiment
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
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
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
            'blitz_vs_sustained'
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
        
        # Save results
        if not args.no_save and results is not None:
            filepath = save_experiment_results(experiment, results, config, args.output_dir)
            print(f"Results saved to: {filepath}")
            
            # Save a summary file for easy reference
            summary_path = os.path.join(args.output_dir, f"{experiment}_summary.json")
            with open(summary_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                summary = {
                    'config': config,
                    'filepath': filepath,
                    'timestamp': os.path.basename(filepath).split('_')[1]
                }
                json.dump(summary, f, indent=2)
    
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main() 