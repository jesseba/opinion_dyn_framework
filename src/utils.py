import os
import json
import pickle
import datetime
from src.model import SUPPORTER, UNDECIDED, OPPOSITION
import numpy as np

def save_experiment_results(experiment_name, results, config, output_dir="results"):
    """
    Save experiment results to a pickle file.
    
    Parameters:
    -----------
    experiment_name : str
        Name of the experiment
    results : dict
        Results dictionary from the experiment
    config : dict
        Configuration parameters used for the experiment
    output_dir : str
        Directory to save results (will be created if it doesn't exist)
        
    Returns:
    --------
    str
        Path to the saved file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiment subdirectory
    exp_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a readable string from config
    config_str = "_".join([f"{k}_{v}" for k, v in config.items() 
                          if k in ['n_nodes', 'total_steps', 'shock_duration']])
    
    # Create filename
    filename = f"{experiment_name}_{timestamp}_{config_str}.pkl"
    filepath = os.path.join(exp_dir, filename)
    
    # Save results
    with open(filepath, 'wb') as f:
        pickle.dump({
            'experiment': experiment_name,
            'config': config,
            'results': results,
            'timestamp': timestamp
        }, f)
    
    print(f"Results saved to: {filepath}")
    return filepath

def load_experiment_results(filepath):
    """
    Load experiment results from a pickle file.
    
    Parameters:
    -----------
    filepath : str
        Path to the results file
        
    Returns:
    --------
    dict
        Dictionary containing experiment results
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def find_experiment_results(experiment_name=None, output_dir="results"):
    """
    Find all experiment result files.
    
    Parameters:
    -----------
    experiment_name : str or None
        Filter by experiment name, or None for all experiments
    output_dir : str
        Base directory containing results
        
    Returns:
    --------
    list
        List of filepaths to result files
    """
    import glob
    
    if experiment_name:
        pattern = os.path.join(output_dir, experiment_name, "*.pkl")
    else:
        pattern = os.path.join(output_dir, "*", "*.pkl")
    
    return sorted(glob.glob(pattern))

def calculate_network_metrics(model):
    """
    Calculate various network analysis metrics based on the current opinion distribution.
    
    Parameters:
    -----------
    model : OpinionDynamicsModel
        The model containing the network and opinion states
    
    Returns:
    --------
    dict
        Dictionary of network metrics
    """
    metrics = {}
    
    # Homophily - tendency of nodes to connect with similar opinions
    state_edges = {
        'supporter-supporter': 0,
        'undecided-undecided': 0,
        'opposition-opposition': 0,
        'supporter-undecided': 0,
        'supporter-opposition': 0,
        'undecided-opposition': 0
    }
    
    total_edges = 0
    
    # Count edges between different opinion groups
    for edge in model.network.edges():
        state_i = model.states[edge[0]]
        state_j = model.states[edge[1]]
        total_edges += 1
        
        if state_i == state_j:
            if state_i == SUPPORTER:
                state_edges['supporter-supporter'] += 1
            elif state_i == UNDECIDED:
                state_edges['undecided-undecided'] += 1
            else:  # OPPOSITION
                state_edges['opposition-opposition'] += 1
        else:
            if (state_i == SUPPORTER and state_j == UNDECIDED) or (state_i == UNDECIDED and state_j == SUPPORTER):
                state_edges['supporter-undecided'] += 1
            elif (state_i == SUPPORTER and state_j == OPPOSITION) or (state_i == OPPOSITION and state_j == SUPPORTER):
                state_edges['supporter-opposition'] += 1
            else:  # undecided-opposition
                state_edges['undecided-opposition'] += 1
    
    # Homophily metric: proportion of same-opinion connections
    same_opinion_edges = state_edges['supporter-supporter'] + state_edges['undecided-undecided'] + state_edges['opposition-opposition']
    metrics['homophily'] = same_opinion_edges / total_edges if total_edges > 0 else 0
    
    # Opinion polarization: ratio of supporter-opposition edges to total edges
    metrics['polarization'] = state_edges['supporter-opposition'] / total_edges if total_edges > 0 else 0
    
    # Segregation: how much each opinion group connects mainly within itself
    opinion_counts = {
        SUPPORTER: np.sum(model.states == SUPPORTER),
        UNDECIDED: np.sum(model.states == UNDECIDED),
        OPPOSITION: np.sum(model.states == OPPOSITION)
    }
    
    # Calculate expected proportion of edges between same opinions based on random mixing
    if total_edges > 0:
        expected_same_opinion = (
            (opinion_counts[SUPPORTER] / model.num_nodes) ** 2 + 
            (opinion_counts[UNDECIDED] / model.num_nodes) ** 2 + 
            (opinion_counts[OPPOSITION] / model.num_nodes) ** 2
        )
        
        # Segregation index: how much more same-opinion connections than expected by chance
        observed_same_opinion = same_opinion_edges / total_edges
        metrics['segregation_index'] = (observed_same_opinion - expected_same_opinion) / (1 - expected_same_opinion) if expected_same_opinion < 1 else 0
    else:
        metrics['segregation_index'] = 0
    
    return metrics 