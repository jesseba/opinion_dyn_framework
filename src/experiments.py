import numpy as np
from tqdm import tqdm
from model import OpinionDynamicsModel, SUPPORTER, UNDECIDED, OPPOSITION
from networks import create_scale_free_network, create_small_world_network, create_random_network
import matplotlib.pyplot as plt
import networkx as nx

def run_grassroots_vs_establishment_experiment(
    n_nodes=1000, 
    shock_duration=20, 
    total_steps=100, 
    num_trials=5,
    lambda_s=0.12,
    lambda_o=0.12
):
    """
    Run experiment comparing grassroots vs. establishment targeting strategies
    across different network types.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in each network
    shock_duration : int
        Duration of shock in time steps
    total_steps : int
        Total simulation duration
    num_trials : int
        Number of simulation trials to run for each condition
    lambda_s : float
        Base rate of movement toward supporter state
    lambda_o : float
        Base rate of movement toward opposition state
    
    Returns:
    --------
    dict
        Dictionary containing results of all simulations
    """
    # Network types to test
    network_types = ['scale-free', 'small-world', 'random']
    
    # Shock strategies to compare - Enhanced with more dramatic differences
    strategies = {
        'No shock': {'func': None, 'params': None},
        'Establishment (High-degree targets)': {
            'func': 'apply_targeted_shock_high_degree',
            'params': {'top_percent': 0.05, 'lambda_s_factor': 5.0}  # Stronger factor for clearer effects
        },
        'Grassroots (Random targets)': {
            'func': 'apply_targeted_shock_random',
            'params': {'target_percent': 0.25, 'lambda_s_factor': 3.0}  # Stronger factor
        }
    }
    
    # Store results
    results = {
        network_type: {
            strategy: {
                'supporter_final': [],
                'undecided_final': [],
                'opposition_final': [],
                'history': [],
                'all_histories': []  # Add storage for all trial histories
            } for strategy in strategies
        } for network_type in network_types
    }
    
    # Run simulations
    for network_type in network_types:
        print(f"Running simulations for {network_type} networks...")
        
        for trial in tqdm(range(num_trials)):
            # Create a new network for each trial
            if network_type == 'scale-free':
                network = create_scale_free_network(n=n_nodes, m=3)
            elif network_type == 'small-world':
                network = create_small_world_network(n=n_nodes, k=6, p=0.1)
            else:  # random
                network = create_random_network(n=n_nodes, k=6)
            
            # Create initial states (same for all strategies in this trial)
            initial_states = np.random.choice(
                [SUPPORTER, UNDECIDED, OPPOSITION], 
                size=n_nodes, 
                p=[0.3, 0.4, 0.3]
            )
            
            # Run each strategy
            for strategy_name, strategy in strategies.items():
                # Create model with consistent initial conditions
                model = OpinionDynamicsModel(
                    network=network,
                    initial_states=initial_states.copy(),  # Ensure each copy is independent
                    lambda_s=lambda_s,
                    lambda_o=lambda_o
                )
                
                # Set up shock function and parameters if applicable
                shock_func = None
                shock_params = None
                
                if strategy['func'] is not None:
                    shock_func = getattr(model, strategy['func'])
                    shock_params = strategy['params']
                
                # Run simulation
                model.run(
                    steps=total_steps,
                    shock_start=10,
                    shock_end=10 + shock_duration,
                    shock_func=shock_func,
                    shock_params=shock_params
                )
                
                # Store results
                final_props = model.get_opinion_proportions()
                results[network_type][strategy_name]['supporter_final'].append(final_props[SUPPORTER])
                results[network_type][strategy_name]['undecided_final'].append(final_props[UNDECIDED])
                results[network_type][strategy_name]['opposition_final'].append(final_props[OPPOSITION])
                
                # Store history for this trial
                history = model.get_history_proportions()
                results[network_type][strategy_name]['all_histories'].append(history)
                
                # Store full history for all representative trial
                if trial == 0:
                    results[network_type][strategy_name]['history'] = history
    
    return results

def run_network_battleground_experiment(
    n_nodes=1000,
    shock_duration=20,
    total_steps=100,
    num_trials=5,
    lambda_s=0.12,
    lambda_o=0.12
):
    """
    Run experiment comparing the effectiveness of strategies across different network types,
    simulating a campaign manager's decision about which network battlegrounds to focus on.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in each network
    shock_duration : int
        Duration of shock in time steps
    total_steps : int
        Total simulation duration
    num_trials : int
        Number of simulation trials to run for each condition
    lambda_s : float
        Base rate of movement toward supporter state
    lambda_o : float
        Base rate of movement toward opposition state
    
    Returns:
    --------
    dict
        Dictionary containing results of all simulations
    """
    # Network types to test - with different parameters representing
    # urban (scale-free), suburban (small-world), and rural (random) networks
    network_types = {
        'urban_center': {'type': 'scale-free', 'params': {'n': n_nodes, 'm': 5}},  # Higher connectivity
        'suburban_area': {'type': 'small-world', 'params': {'n': n_nodes, 'k': 6, 'p': 0.1}},
        'rural_community': {'type': 'random', 'params': {'n': n_nodes, 'k': 4}}  # Lower connectivity
    }
    
    # Shock strategies
    strategies = {
        'Establishment (High-degree targets)': {
            'func': 'apply_targeted_shock_high_degree',
            'params': {'top_percent': 0.05, 'lambda_s_factor': 5.0}
        }
    }
    
    # Store results
    results = {
        network_name: {
            'supporter_gain': [],  # Track gain in supporters
            'resource_efficiency': [],  # Gain per "resource unit" spent (top_percent)
            'undecided_final': [],
            'opposition_final': []
        } for network_name in network_types
    }
    
    # Run simulations
    for network_name, network_config in network_types.items():
        print(f"Running simulations for {network_name}...")
        
        for trial in tqdm(range(num_trials)):
            # Create network based on configuration
            if network_config['type'] == 'scale-free':
                network = create_scale_free_network(**network_config['params'])
            elif network_config['type'] == 'small-world':
                network = create_small_world_network(**network_config['params'])
            else:  # random
                network = create_random_network(**network_config['params'])
            
            # Create initial states
            initial_states = np.random.choice(
                [SUPPORTER, UNDECIDED, OPPOSITION], 
                size=n_nodes, 
                p=[0.3, 0.4, 0.3]
            )
            
            # Create model
            model = OpinionDynamicsModel(
                network=network,
                initial_states=initial_states.copy(),
                lambda_s=lambda_s,
                lambda_o=lambda_o
            )
            
            # Store initial supporter proportion
            initial_props = model.get_opinion_proportions()
            initial_supporters = initial_props[SUPPORTER]
            
            # Set up shock function and parameters
            strategy = strategies['Establishment (High-degree targets)']
            shock_func = getattr(model, strategy['func'])
            shock_params = strategy['params']
            
            # Run simulation
            model.run(
                steps=total_steps,
                shock_start=10,
                shock_end=10 + shock_duration,
                shock_func=shock_func,
                shock_params=shock_params
            )
            
            # Store results
            final_props = model.get_opinion_proportions()
            supporter_gain = final_props[SUPPORTER] - initial_supporters
            
            # Calculate efficiency (gain per resource unit)
            resource_spent = shock_params['top_percent']  # Proxy for campaign resources
            efficiency = supporter_gain / resource_spent
            
            results[network_name]['supporter_gain'].append(supporter_gain)
            results[network_name]['resource_efficiency'].append(efficiency)
            results[network_name]['undecided_final'].append(final_props[UNDECIDED])
            results[network_name]['opposition_final'].append(final_props[OPPOSITION])
    
    return results

def run_timing_experiment(
    n_nodes=1000,
    shock_duration=20,
    total_steps=100,
    num_trials=5,
    lambda_s=0.12,
    lambda_o=0.12
):
    """
    Run experiment comparing early vs. late campaign timing.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in each network
    shock_duration : int
        Duration of shock in time steps
    total_steps : int
        Total simulation duration
    num_trials : int
        Number of simulation trials to run for each condition
    lambda_s : float
        Base rate of movement toward supporter state
    lambda_o : float
        Base rate of movement toward opposition state
    
    Returns:
    --------
    dict
        Dictionary containing results of all simulations
    """
    # Network type - using scale-free for this experiment
    network_params = {'n': n_nodes, 'm': 3}
    
    # Shock timing options
    timings = {
        'Early Campaign': 10,  # Start shock at time step 10
        'Late Campaign': 50    # Start shock at time step 50
    }
    
    # Store results
    results = {
        timing: {
            'supporter_final': [],
            'undecided_final': [],
            'opposition_final': [],
            'history': [],
            'all_histories': []  # Add storage for all trial histories
        } for timing in timings
    }
    
    # Run simulations
    for timing_name, shock_start in timings.items():
        print(f"Running simulations for {timing_name}...")
        
        for trial in tqdm(range(num_trials)):
            # Create network
            network = create_scale_free_network(**network_params)
            
            # Create initial states
            initial_states = np.random.choice(
                [SUPPORTER, UNDECIDED, OPPOSITION], 
                size=n_nodes, 
                p=[0.3, 0.4, 0.3]
            )
            
            # Create model
            model = OpinionDynamicsModel(
                network=network,
                initial_states=initial_states.copy(),
                lambda_s=lambda_s,
                lambda_o=lambda_o
            )
            
            # Apply high-degree targeting strategy
            shock_func = model.apply_targeted_shock_high_degree
            shock_params = {'top_percent': 0.05, 'lambda_s_factor': 5.0}
            
            # Run simulation
            model.run(
                steps=total_steps,
                shock_start=shock_start,
                shock_end=shock_start + shock_duration,
                shock_func=shock_func,
                shock_params=shock_params
            )
            
            # Store results
            final_props = model.get_opinion_proportions()
            results[timing_name]['supporter_final'].append(final_props[SUPPORTER])
            results[timing_name]['undecided_final'].append(final_props[UNDECIDED])
            results[timing_name]['opposition_final'].append(final_props[OPPOSITION])
            
            # Store history for all trials
            history = model.get_history_proportions()
            results[timing_name]['all_histories'].append(history)
            
            # Store full history for the first trial
            if trial == 0:
                results[timing_name]['history'] = history
    
    return results

def run_intervention_pattern_experiment(
    n_nodes=1000,
    total_steps=150,
    num_trials=5,
    lambda_s=0.12,
    lambda_o=0.12
):
    """
    Run experiment comparing different intervention patterns.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in the network
    total_steps : int
        Total simulation duration
    num_trials : int
        Number of simulation trials to run for each condition
    lambda_s : float
        Base rate of movement toward supporter state
    lambda_o : float
        Base rate of movement toward opposition state
    
    Returns:
    --------
    dict
        Dictionary containing results of all simulations
    """
    # Create scale-free network for all trials
    network_params = {'n': n_nodes, 'm': 3}
    
    # Intervention patterns
    patterns = {
        'No Intervention': {
            'schedule': []
        },
        'Blitz': {
            'schedule': [{'start': 30, 'end': 40, 'lambda_s_factor': 5.0}]
        },
        'Sustained': {
            'schedule': [{'start': 30, 'end': 70, 'lambda_s_factor': 3.0}]
        },
        'Pulsed': {
            'schedule': [
                {'start': 30, 'end': 40, 'lambda_s_factor': 3.0},
                {'start': 60, 'end': 70, 'lambda_s_factor': 3.0},
                {'start': 90, 'end': 100, 'lambda_s_factor': 3.0}
            ]
        },
        'Early Blitz + Late Reinforcement': {
            'schedule': [
                {'start': 30, 'end': 40, 'lambda_s_factor': 4.0},
                {'start': 90, 'end': 100, 'lambda_s_factor': 2.0}
            ]
        }
    }
    
    # Store results
    results = {
        pattern: {
            'supporter_final': [],
            'undecided_final': [],
            'opposition_final': [],
            'shock_schedule': pattern_info['schedule'],
            'history': [],
            'all_histories': []  # Store all histories for confidence intervals
        } for pattern, pattern_info in patterns.items()
    }
    
    # Run simulations
    for pattern_name, pattern_info in patterns.items():
        print(f"Running simulations for {pattern_name}...")
        
        for trial in tqdm(range(num_trials)):
            # Create network
            network = create_scale_free_network(**network_params)
            
            # Create initial states
            initial_states = np.random.choice(
                [SUPPORTER, UNDECIDED, OPPOSITION], 
                size=n_nodes, 
                p=[0.3, 0.4, 0.3]
            )
            
            # Create model
            model = OpinionDynamicsModel(
                network=network,
                initial_states=initial_states.copy(),
                lambda_s=lambda_s,
                lambda_o=lambda_o
            )
            
            # Execute the pattern
            schedule = pattern_info['schedule']
            current_time = 0
            
            for intervention in schedule:
                # Run until the start of this intervention
                if current_time < intervention['start']:
                    model.run(steps=intervention['start'] - current_time)
                    current_time = intervention['start']
                
                # Apply intervention
                model.apply_broadcast_shock(
                    lambda_s_factor=intervention['lambda_s_factor']
                )
                
                # Run during intervention
                model.run(steps=intervention['end'] - intervention['start'])
                current_time = intervention['end']
                
                # Reset after intervention
                model.reset_shocks()
            
            # Run remaining steps
            if current_time < total_steps:
                model.run(steps=total_steps - current_time)
            
            # Store results
            final_props = model.get_opinion_proportions()
            results[pattern_name]['supporter_final'].append(final_props[SUPPORTER])
            results[pattern_name]['undecided_final'].append(final_props[UNDECIDED])
            results[pattern_name]['opposition_final'].append(final_props[OPPOSITION])
            
            # Store history for this trial
            history = model.get_history_proportions()
            results[pattern_name]['all_histories'].append(history)
            
            # Store representative history (first trial only)
            if trial == 0:
                results[pattern_name]['history'] = history
    
    return results

def run_blitz_vs_sustained_experiment(
    n_nodes=1000,
    total_steps=150,
    num_trials=5,
    lambda_s=0.12,
    lambda_o=0.12,
    vary_intensity=True
):
    """
    Run experiment comparing blitz (short, intense) vs sustained intervention patterns
    across different network types.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in the network
    total_steps : int
        Total simulation duration
    num_trials : int
        Number of simulation trials to run for each condition
    lambda_s : float
        Base rate of movement toward supporter state
    lambda_o : float
        Base rate of movement toward opposition state
    vary_intensity : bool
        If True, maintain consistent total "intervention power" across patterns
        (shorter interventions are more intense, longer are less intense)
    
    Returns:
    --------
    dict
        Dictionary containing results of all simulations
    """
    # Network types to compare
    network_types = {
        'Scale-free': {'create_func': create_scale_free_network, 'params': {'n': n_nodes, 'm': 3}},
        'Small-world': {'create_func': create_small_world_network, 'params': {'n': n_nodes, 'k': 6, 'p': 0.1}},
        'Random': {'create_func': create_random_network, 'params': {'n': n_nodes, 'k': 6}}
    }
    
    # Intervention patterns
    if vary_intensity:
        # Total "intervention power" is roughly equalized
        patterns = {
            'No Intervention': {
                'schedule': []
            },
            'Blitz (10 steps)': {
                'schedule': [{'start': 30, 'end': 40, 'lambda_s_factor': 8.0}]
            },
            'Medium (20 steps)': {
                'schedule': [{'start': 30, 'end': 50, 'lambda_s_factor': 4.0}]
            },
            'Sustained (40 steps)': {
                'schedule': [{'start': 30, 'end': 70, 'lambda_s_factor': 2.0}]
            },
            'Ultra-sustained (60 steps)': {
                'schedule': [{'start': 30, 'end': 90, 'lambda_s_factor': 1.33}]
            }
        }
    else:
        # Same intensity across all patterns
        patterns = {
            'No Intervention': {
                'schedule': []
            },
            'Blitz (10 steps)': {
                'schedule': [{'start': 30, 'end': 40, 'lambda_s_factor': 3.0}]
            },
            'Medium (20 steps)': {
                'schedule': [{'start': 30, 'end': 50, 'lambda_s_factor': 3.0}]
            },
            'Sustained (40 steps)': {
                'schedule': [{'start': 30, 'end': 70, 'lambda_s_factor': 3.0}]
            },
            'Ultra-sustained (60 steps)': {
                'schedule': [{'start': 30, 'end': 90, 'lambda_s_factor': 3.0}]
            }
        }
    
    # Store results - nested dictionary: network_type -> pattern -> metrics
    results = {}
    
    for network_name, network_info in network_types.items():
        print(f"\nRunning simulations for {network_name} network...")
        results[network_name] = {}
        
        for pattern_name, pattern_info in patterns.items():
            print(f"  Pattern: {pattern_name}")
            results[network_name][pattern_name] = {
                'supporter_final': [],
                'undecided_final': [],
                'opposition_final': [],
                'supporter_peak': [],
                'supporter_time_to_peak': [],
                'supporter_decay_rate': [],
                'history': [],
                'shock_schedule': pattern_info['schedule']
            }
            
            for trial in tqdm(range(num_trials)):
                # Create network
                network = network_info['create_func'](**network_info['params'])
                
                # Create initial states
                initial_states = np.random.choice(
                    [SUPPORTER, UNDECIDED, OPPOSITION], 
                    size=n_nodes, 
                    p=[0.3, 0.4, 0.3]
                )
                
                # Create model
                model = OpinionDynamicsModel(
                    network=network,
                    initial_states=initial_states.copy(),
                    lambda_s=lambda_s,
                    lambda_o=lambda_o
                )
                
                # Execute the pattern
                schedule = pattern_info['schedule']
                current_time = 0
                
                for intervention in schedule:
                    # Run until the start of this intervention
                    if current_time < intervention['start']:
                        model.run(steps=intervention['start'] - current_time)
                        current_time = intervention['start']
                    
                    # Apply intervention
                    model.apply_broadcast_shock(
                        lambda_s_factor=intervention['lambda_s_factor']
                    )
                    
                    # Run during intervention
                    model.run(steps=intervention['end'] - intervention['start'])
                    current_time = intervention['end']
                    
                    # Reset after intervention
                    model.reset_shocks()
                
                # Run remaining steps
                if current_time < total_steps:
                    model.run(steps=total_steps - current_time)
                
                # Store results
                final_props = model.get_opinion_proportions()
                results[network_name][pattern_name]['supporter_final'].append(final_props[SUPPORTER])
                results[network_name][pattern_name]['undecided_final'].append(final_props[UNDECIDED])
                results[network_name][pattern_name]['opposition_final'].append(final_props[OPPOSITION])
                
                # Store full history for analysis
                if trial == 0:
                    history = model.get_history_proportions()
                    results[network_name][pattern_name]['history'] = history
                    
                    # Calculate peak metrics
                    supporter_values = [h[SUPPORTER] for h in history]
                    peak_value = max(supporter_values)
                    peak_time = supporter_values.index(peak_value)
                    
                    results[network_name][pattern_name]['supporter_peak'].append(peak_value)
                    results[network_name][pattern_name]['supporter_time_to_peak'].append(peak_time)
                    
                    # Calculate decay rate (if applicable)
                    if peak_time < len(supporter_values) - 1 and peak_time > 0:
                        post_peak = supporter_values[peak_time:]
                        if len(post_peak) > 10:  # Enough data points
                            # Simple linear regression on post-peak values
                            times = np.arange(len(post_peak))
                            if np.std(post_peak) > 0:  # Avoid div by zero
                                slope = np.polyfit(times, post_peak, 1)[0]
                                results[network_name][pattern_name]['supporter_decay_rate'].append(slope)
    
    return results

def run_critical_mass_experiment(
    n_nodes=1000,
    total_steps=100,
    num_trials=5,
    lambda_s=0.12,
    lambda_o=0.12,
    initial_supporter_range=np.linspace(0.05, 0.95, 19)
):
    """
    Run experiment to test how the final state varies with different initial supporter proportions.
    This experiment investigates critical mass effects in opinion dynamics.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in each network
    total_steps : int
        Total simulation duration
    num_trials : int
        Number of simulation trials to run for each condition
    lambda_s : float
        Base rate of movement toward supporter state
    lambda_o : float
        Base rate of movement toward opposition state
    initial_supporter_range : array-like
        Range of initial supporter proportions to test
        
    Returns:
    --------
    dict
        Dictionary containing results of all simulations
    """
    # Network types to test
    network_types = ['scale-free', 'small-world', 'random']
    
    # Store results
    results = {
        network_type: {
            'initial_proportions': initial_supporter_range,
            'final_supporters': [],
            'final_supporters_std': [],
            'final_undecided': [],
            'final_undecided_std': [],
            'final_opposition': [],
            'final_opposition_std': [],
            'trials': {p: [] for p in initial_supporter_range},  # Store individual trial results
        } for network_type in network_types
    }
    
    # Run simulations for each network type
    for network_type in network_types:
        print(f"Running critical mass simulations for {network_type} networks...")
        
        # For each initial supporter proportion
        for init_proportion in tqdm(initial_supporter_range):
            supporter_finals = []
            undecided_finals = []
            opposition_finals = []
            
            # Run multiple trials
            for trial in range(num_trials):
                # Create network
                if network_type == 'scale-free':
                    network = create_scale_free_network(n=n_nodes, m=3)
                elif network_type == 'small-world':
                    network = create_small_world_network(n=n_nodes, k=6, p=0.1)
                else:  # random
                    network = create_random_network(n=n_nodes, k=6)
                
                # Create initial states according to specified proportion
                undecided_opposition_total = 1.0 - init_proportion
                # Equal split between undecided and opposition from remaining proportion
                undecided_prop = opposition_prop = undecided_opposition_total / 2
                
                initial_states = np.random.choice(
                    [SUPPORTER, UNDECIDED, OPPOSITION], 
                    size=n_nodes, 
                    p=[init_proportion, undecided_prop, opposition_prop]
                )
                
                # Create and run model without interventions
                model = OpinionDynamicsModel(
                    network=network,
                    initial_states=initial_states,
                    lambda_s=lambda_s,
                    lambda_o=lambda_o
                )
                
                # Run simulation without shocks
                model.run(steps=total_steps)
                
                # Store results
                final_props = model.get_opinion_proportions()
                supporter_finals.append(final_props[SUPPORTER])
                undecided_finals.append(final_props[UNDECIDED])
                opposition_finals.append(final_props[OPPOSITION])
                
                # Store full trial data
                results[network_type]['trials'][init_proportion].append({
                    'final_supporters': final_props[SUPPORTER],
                    'final_undecided': final_props[UNDECIDED],
                    'final_opposition': final_props[OPPOSITION],
                    'history': model.get_history_proportions() if trial == 0 else None  # Store history for first trial only
                })
            
            # Calculate statistics across trials
            results[network_type]['final_supporters'].append(np.mean(supporter_finals))
            results[network_type]['final_supporters_std'].append(np.std(supporter_finals))
            results[network_type]['final_undecided'].append(np.mean(undecided_finals))
            results[network_type]['final_undecided_std'].append(np.std(undecided_finals))
            results[network_type]['final_opposition'].append(np.mean(opposition_finals))
            results[network_type]['final_opposition_std'].append(np.std(opposition_finals))
    
    return results

def run_targeted_seeding_experiment(
    n_nodes=1000, 
    total_steps=100, 
    num_trials=5,
    lambda_s=0.12,
    lambda_o=0.12,
    initial_supporter_percent=0.15  # Lower percentage makes seeding strategy more important
):
    """
    Run experiment comparing different seeding strategies for initial supporters.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in each network
    total_steps : int
        Total simulation duration
    num_trials : int
        Number of simulation trials to run for each condition
    lambda_s : float
        Base rate of movement toward supporter state
    lambda_o : float
        Base rate of movement toward opposition state
    initial_supporter_percent : float
        Percentage of nodes that will be initial supporters
        
    Returns:
    --------
    dict
        Dictionary containing results of all simulations
    """
    # Network type - using scale-free for this experiment as it has clear hubs
    network = create_scale_free_network(n=n_nodes, m=3)
    
    # Seeding strategies to compare
    strategies = {
        'Random Seeding': 'random',
        'High-Degree Seeding': 'high_degree',
        'Betweenness Seeding': 'betweenness',
        'Clustered Seeding': 'clustered'
    }
    
    # Store results
    results = {
        strategy: {
            'supporter_final': [],
            'undecided_final': [],
            'opposition_final': [],
            'history': [],
            'all_histories': []
        } for strategy in strategies
    }
    
    # Calculate centrality measures once for efficiency
    degree_dict = dict(network.degree())
    betweenness_dict = nx.betweenness_centrality(network)
    
    # Get community structure for clustered seeding
    communities = list(nx.algorithms.community.greedy_modularity_communities(network))
    
    # Number of initial supporters
    n_supporters = int(n_nodes * initial_supporter_percent)
    
    # Run simulations
    for strategy_name, strategy_type in strategies.items():
        print(f"Running simulations for {strategy_name}...")
        
        for trial in tqdm(range(num_trials)):
            # Create initial states - everyone starts as undecided
            initial_states = np.full(n_nodes, UNDECIDED)
            
            # Apply seeding strategy
            if strategy_type == 'random':
                # Random seeding
                supporter_indices = np.random.choice(n_nodes, size=n_supporters, replace=False)
            
            elif strategy_type == 'high_degree':
                # High-degree seeding
                sorted_nodes = sorted(range(n_nodes), key=lambda i: degree_dict[i], reverse=True)
                supporter_indices = sorted_nodes[:n_supporters]
                
            elif strategy_type == 'betweenness':
                # Betweenness centrality seeding
                sorted_nodes = sorted(range(n_nodes), key=lambda i: betweenness_dict[i], reverse=True)
                supporter_indices = sorted_nodes[:n_supporters]
                
            elif strategy_type == 'clustered':
                # Clustered seeding - pick top nodes from each community
                supporter_indices = []
                # Determine how many supporters to place in each community (proportional to size)
                for community in communities:
                    community = list(community)
                    n_community_supporters = max(1, int(len(community) * initial_supporter_percent))
                    # Sort community by degree
                    sorted_community = sorted(community, key=lambda i: degree_dict[i], reverse=True)
                    # Take top nodes from this community
                    supporter_indices.extend(sorted_community[:n_community_supporters])
                
                # If we've selected too many, trim the list
                if len(supporter_indices) > n_supporters:
                    supporter_indices = supporter_indices[:n_supporters]
                # If we need more, add random nodes
                elif len(supporter_indices) < n_supporters:
                    remaining = n_supporters - len(supporter_indices)
                    eligible = [i for i in range(n_nodes) if i not in supporter_indices]
                    additional = np.random.choice(eligible, size=remaining, replace=False)
                    supporter_indices.extend(additional)
            
            # Set the selected nodes as supporters
            initial_states[supporter_indices] = SUPPORTER
            
            # Add some opposition nodes - randomly distributed
            n_opposition = int(n_nodes * initial_supporter_percent)  # Same percentage as supporters
            eligible = [i for i in range(n_nodes) if initial_states[i] == UNDECIDED]
            opposition_indices = np.random.choice(eligible, size=n_opposition, replace=False)
            initial_states[opposition_indices] = OPPOSITION
            
            # Create model
            model = OpinionDynamicsModel(
                network=network.copy(),  # Use same network but make copy
                initial_states=initial_states,
                lambda_s=lambda_s,
                lambda_o=lambda_o
            )
            
            # Run simulation without interventions
            model.run(steps=total_steps)
            
            # Store results
            final_props = model.get_opinion_proportions()
            results[strategy_name]['supporter_final'].append(final_props[SUPPORTER])
            results[strategy_name]['undecided_final'].append(final_props[UNDECIDED])
            results[strategy_name]['opposition_final'].append(final_props[OPPOSITION])
            
            # Store history
            history = model.get_history_proportions()
            results[strategy_name]['all_histories'].append(history)
            
            # Store representative history
            if trial == 0:
                results[strategy_name]['history'] = history
    
    return results

def run_opponent_composition_experiment(
    n_nodes=1000, 
    total_steps=100, 
    num_trials=5,
    lambda_s=0.12,
    lambda_o=0.12,
    initial_supporter_percent=0.3,
    undecided_ratio_range=np.linspace(0.0, 1.0, 11)  # 0 to 1 in steps of 0.1
):
    """
    Run experiment varying the composition of the non-supporter population.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in each network
    total_steps : int
        Total simulation duration
    num_trials : int
        Number of simulation trials to run for each condition
    lambda_s : float
        Base rate of movement toward supporter state
    lambda_o : float
        Base rate of movement toward opposition state
    initial_supporter_percent : float
        Fixed percentage of initial supporters
    undecided_ratio_range : array-like
        Range of ratios for undecided within the non-supporter population
        
    Returns:
    --------
    dict
        Dictionary containing results of all simulations
    """
    # Network types to test
    network_types = ['scale-free', 'small-world', 'random']
    
    # Store results
    results = {
        network_type: {
            'undecided_ratios': undecided_ratio_range,
            'supporter_final': [],
            'undecided_final': [],
            'opposition_final': [],
            'all_trials': {}  # Store individual trial results
        } for network_type in network_types
    }
    
    # Run simulations
    for network_type in network_types:
        print(f"Running simulations for {network_type} networks...")
        
        for undecided_ratio in tqdm(undecided_ratio_range):
            supporter_finals = []
            undecided_finals = []
            opposition_finals = []
            results[network_type]['all_trials'][undecided_ratio] = []
            
            # Calculate exact proportions
            # - Fixed initial supporters
            # - Remaining split between undecided and opposition based on ratio
            non_supporter_percent = 1.0 - initial_supporter_percent
            undecided_percent = non_supporter_percent * undecided_ratio
            opposition_percent = non_supporter_percent * (1 - undecided_ratio)
            
            for trial in range(num_trials):
                # Create network
                if network_type == 'scale-free':
                    network = create_scale_free_network(n=n_nodes, m=3)
                elif network_type == 'small-world':
                    network = create_small_world_network(n=n_nodes, k=6, p=0.1)
                else:  # random
                    network = create_random_network(n=n_nodes, k=6)
                
                # Create initial states
                initial_states = np.random.choice(
                    [SUPPORTER, UNDECIDED, OPPOSITION], 
                    size=n_nodes, 
                    p=[initial_supporter_percent, undecided_percent, opposition_percent]
                )
                
                # Create and run model
                model = OpinionDynamicsModel(
                    network=network,
                    initial_states=initial_states,
                    lambda_s=lambda_s,
                    lambda_o=lambda_o
                )
                
                # Run simulation
                model.run(steps=total_steps)
                
                # Store results
                final_props = model.get_opinion_proportions()
                supporter_finals.append(final_props[SUPPORTER])
                undecided_finals.append(final_props[UNDECIDED])
                opposition_finals.append(final_props[OPPOSITION])
                
                # Store full trial data
                results[network_type]['all_trials'][undecided_ratio].append({
                    'final_supporters': final_props[SUPPORTER],
                    'final_undecided': final_props[UNDECIDED],
                    'final_opposition': final_props[OPPOSITION],
                    'history': model.get_history_proportions() if trial == 0 else None
                })
            
            # Calculate statistics across trials
            results[network_type]['supporter_final'].append(np.mean(supporter_finals))
            results[network_type]['undecided_final'].append(np.mean(undecided_finals))
            results[network_type]['opposition_final'].append(np.mean(opposition_finals))
    
    return results

def run_transition_rate_asymmetry_experiment(
    n_nodes=1000,
    total_steps=100,
    num_trials=5,
    lambda_base=0.12,
    lambda_ratios=np.logspace(-1, 1, 11),  # 0.1 to 10 in log space
    initial_proportions=[0.3, 0.4, 0.3]  # [supporter, undecided, opposition]
):
    """
    Run experiment varying the asymmetry between supporter and opposition transition rates.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in each network
    total_steps : int
        Total simulation duration
    num_trials : int
        Number of simulation trials to run for each condition
    lambda_base : float
        Base transition rate
    lambda_ratios : array-like
        Range of λs/λo ratios to test
    initial_proportions : list
        Initial distribution of [supporter, undecided, opposition]
        
    Returns:
    --------
    dict
        Dictionary containing results of all simulations
    """
    # Network types to test
    network_types = ['scale-free', 'small-world', 'random']
    
    # Store results
    results = {
        network_type: {
            'lambda_ratios': lambda_ratios,
            'supporter_final': [],
            'undecided_final': [],
            'opposition_final': [],
            'supporter_advantage': [],  # Difference between final supporter and opposition proportions
            'all_trials': {}  # Store individual trial results
        } for network_type in network_types
    }
    
    # Run simulations
    for network_type in network_types:
        print(f"Running simulations for {network_type} networks...")
        
        for ratio in tqdm(lambda_ratios):
            # Calculate transition rates based on ratio
            lambda_s = lambda_base * np.sqrt(ratio)  # Increases with ratio
            lambda_o = lambda_base / np.sqrt(ratio)  # Decreases with ratio
            # This ensures λs * λo = constant to maintain overall transition rate
            
            supporter_finals = []
            undecided_finals = []
            opposition_finals = []
            results[network_type]['all_trials'][ratio] = []
            
            for trial in range(num_trials):
                # Create network
                if network_type == 'scale-free':
                    network = create_scale_free_network(n=n_nodes, m=3)
                elif network_type == 'small-world':
                    network = create_small_world_network(n=n_nodes, k=6, p=0.1)
                else:  # random
                    network = create_random_network(n=n_nodes, k=6)
                
                # Create initial states
                initial_states = np.random.choice(
                    [SUPPORTER, UNDECIDED, OPPOSITION], 
                    size=n_nodes, 
                    p=initial_proportions
                )
                
                # Create and run model
                model = OpinionDynamicsModel(
                    network=network,
                    initial_states=initial_states,
                    lambda_s=lambda_s,
                    lambda_o=lambda_o
                )
                
                # Run simulation
                model.run(steps=total_steps)
                
                # Store results
                final_props = model.get_opinion_proportions()
                supporter_finals.append(final_props[SUPPORTER])
                undecided_finals.append(final_props[UNDECIDED])
                opposition_finals.append(final_props[OPPOSITION])
                
                # Store full trial data
                results[network_type]['all_trials'][ratio].append({
                    'final_supporters': final_props[SUPPORTER],
                    'final_undecided': final_props[UNDECIDED],
                    'final_opposition': final_props[OPPOSITION],
                    'history': model.get_history_proportions() if trial == 0 else None
                })
            
            # Calculate statistics across trials
            results[network_type]['supporter_final'].append(np.mean(supporter_finals))
            results[network_type]['undecided_final'].append(np.mean(undecided_finals))
            results[network_type]['opposition_final'].append(np.mean(opposition_finals))
            results[network_type]['supporter_advantage'].append(
                np.mean(supporter_finals) - np.mean(opposition_finals)
            )
    
    return results

def run_intervention_sensitivity_experiment(
    n_nodes=1000,
    shock_duration=20,
    total_steps=100,
    num_trials=5,
    lambda_s=0.12,
    lambda_o=0.12,
    initial_supporter_range=np.linspace(0.05, 0.95, 10),
    intervention_type='establishment'  # 'establishment' or 'grassroots'
):
    """
    Run experiment testing how intervention effectiveness varies with initial supporter levels.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in each network
    shock_duration : int
        Duration of shock in time steps
    total_steps : int
        Total simulation duration
    num_trials : int
        Number of simulation trials to run for each condition
    lambda_s : float
        Base rate of movement toward supporter state
    lambda_o : float
        Base rate of movement toward opposition state
    initial_supporter_range : array-like
        Range of initial supporter proportions to test
    intervention_type : str
        Type of intervention strategy to use ('establishment' or 'grassroots')
        
    Returns:
    --------
    dict
        Dictionary containing results of all simulations
    """
    # Network type - using scale-free for this experiment
    network_type = 'scale-free'
    
    # Set up intervention parameters based on type
    if intervention_type == 'establishment':
        intervention_func_name = 'apply_targeted_shock_high_degree'
        intervention_params = {'top_percent': 0.05, 'lambda_s_factor': 5.0}
    else:  # grassroots
        intervention_func_name = 'apply_targeted_shock_random'
        intervention_params = {'target_percent': 0.25, 'lambda_s_factor': 3.0}
    
    # Store results
    results = {
        'initial_proportions': initial_supporter_range,
        'intervention_type': intervention_type,
        'baseline_final_supporters': [],  # No intervention
        'baseline_final_supporters_std': [],
        'intervention_final_supporters': [],  # With intervention
        'intervention_final_supporters_std': [],
        'supporter_gain': [],  # Difference between intervention and baseline
        'supporter_gain_std': [],
        'relative_gain': [],  # Gain divided by initial proportion
        'relative_gain_std': [],
        'trials': {p: {'baseline': [], 'intervention': []} for p in initial_supporter_range}
    }
    
    print(f"Running intervention sensitivity experiment with {intervention_type} strategy...")
    
    # For each initial supporter proportion
    for init_proportion in tqdm(initial_supporter_range):
        baseline_supporters = []
        intervention_supporters = []
        
        # Run multiple trials
        for trial in range(num_trials):
            # Create network
            network = create_scale_free_network(n=n_nodes, m=3)
            
            # Create initial states according to specified proportion
            undecided_opposition_total = 1.0 - init_proportion
            # Equal split between undecided and opposition from remaining proportion
            undecided_prop = opposition_prop = undecided_opposition_total / 2
            
            initial_states = np.random.choice(
                [SUPPORTER, UNDECIDED, OPPOSITION], 
                size=n_nodes, 
                p=[init_proportion, undecided_prop, opposition_prop]
            )
            
            # BASELINE: Run without intervention
            baseline_model = OpinionDynamicsModel(
                network=network.copy(),
                initial_states=initial_states.copy(),
                lambda_s=lambda_s,
                lambda_o=lambda_o
            )
            baseline_model.run(steps=total_steps)
            baseline_final = baseline_model.get_opinion_proportions()
            baseline_supporters.append(baseline_final[SUPPORTER])
            
            # INTERVENTION: Run with specified intervention
            intervention_model = OpinionDynamicsModel(
                network=network.copy(),
                initial_states=initial_states.copy(),
                lambda_s=lambda_s,
                lambda_o=lambda_o
            )
            
            # Set up intervention
            shock_func = getattr(intervention_model, intervention_func_name)
            
            # Run with intervention
            intervention_model.run(
                steps=total_steps,
                shock_start=10,
                shock_end=10 + shock_duration,
                shock_func=shock_func,
                shock_params=intervention_params
            )
            
            intervention_final = intervention_model.get_opinion_proportions()
            intervention_supporters.append(intervention_final[SUPPORTER])
            
            # Store full trial data 
            results['trials'][init_proportion]['baseline'].append({
                'final_supporters': baseline_final[SUPPORTER],
                'final_undecided': baseline_final[UNDECIDED],
                'final_opposition': baseline_final[OPPOSITION],
                'history': baseline_model.get_history_proportions() if trial == 0 else None
            })
            
            results['trials'][init_proportion]['intervention'].append({
                'final_supporters': intervention_final[SUPPORTER],
                'final_undecided': intervention_final[UNDECIDED],
                'final_opposition': intervention_final[OPPOSITION],
                'history': intervention_model.get_history_proportions() if trial == 0 else None
            })
        
        # Calculate statistics across trials
        baseline_mean = np.mean(baseline_supporters)
        baseline_std = np.std(baseline_supporters)
        intervention_mean = np.mean(intervention_supporters)
        intervention_std = np.std(intervention_supporters)
        
        # Calculate absolute gain
        gain = intervention_mean - baseline_mean
        # Compute combined standard deviation using error propagation
        gain_std = np.sqrt(baseline_std**2 + intervention_std**2)
        
        # Calculate relative gain (normalized by initial proportion)
        relative_gain = gain / max(0.01, init_proportion)  # Avoid division by zero
        relative_gain_std = gain_std / max(0.01, init_proportion)
        
        # Store results
        results['baseline_final_supporters'].append(baseline_mean)
        results['baseline_final_supporters_std'].append(baseline_std)
        results['intervention_final_supporters'].append(intervention_mean)
        results['intervention_final_supporters_std'].append(intervention_std)
        results['supporter_gain'].append(gain)
        results['supporter_gain_std'].append(gain_std)
        results['relative_gain'].append(relative_gain)
        results['relative_gain_std'].append(relative_gain_std)
    
    return results

# fig2, axes2 = plt.subplots(len(network_types), len(strategies), figsize=(16, 12))

# # Add shock period indicator
# shock_start = 10
# shock_end = 10 + 20  # shock_duration

# for i, network_type in enumerate(network_types):
#     for j, strategy in enumerate(strategies):
#         ax = axes2[i, j]
        
#         if 'all_histories' in results[network_type][strategy] and len(results[network_type][strategy]['all_histories']) > 0:
#             # Get all histories for this network type and strategy
#             all_histories = results[network_type][strategy]['all_histories']
            
#             # Calculate mean and std for each time step
#             num_steps = len(all_histories[0])
#             supporters_data = np.zeros((len(all_histories), num_steps))
#             undecided_data = np.zeros_like(supporters_data)
#             opposition_data = np.zeros_like(supporters_data)
            
#             # Collect data from all trials
#             for k, hist in enumerate(all_histories):
#                 if len(hist) == num_steps:  # Ensure same length
#                     supporters_data[k] = [h[SUPPORTER] for h in hist]
#                     undecided_data[k] = [h[UNDECIDED] for h in hist]
#                     opposition_data[k] = [h[OPPOSITION] for h in hist]
            
#             # Calculate means and standard deviations
#             supporters_mean = np.mean(supporters_data, axis=0)
#             supporters_std = np.std(supporters_data, axis=0)
#             undecided_mean = np.mean(undecided_data, axis=0)
#             undecided_std = np.std(undecided_data, axis=0)
#             opposition_mean = np.mean(opposition_data, axis=0)
#             opposition_std = np.std(opposition_data, axis=0)
            
#             steps = range(num_steps)
            
#             # Plot means
#             ax.plot(steps, supporters_mean, '-', color=SUPPORTER_COLOR, linewidth=2.5, label='Supporters')
#             ax.plot(steps, undecided_mean, '-', color=UNDECIDED_COLOR, linewidth=2.5, label='Undecided')
#             ax.plot(steps, opposition_mean, '-', color=OPPOSITION_COLOR, linewidth=2.5, label='Opposition')
            
#             # Add confidence intervals
#             ax.fill_between(steps, 
#                            supporters_mean - supporters_std, 
#                            supporters_mean + supporters_std, 
#                            color=SUPPORTER_COLOR, alpha=0.2)
#             ax.fill_between(steps, 
#                            undecided_mean - undecided_std, 
#                            undecided_mean + undecided_std, 
#                            color=UNDECIDED_COLOR, alpha=0.2)
#             ax.fill_between(steps, 
#                            opposition_mean - opposition_std, 
#                            opposition_mean + opposition_std, 
#                            color=OPPOSITION_COLOR, alpha=0.2)
            
#             # Highlight the shock period with better styling
#             ax.axvspan(shock_start, shock_end, alpha=0.15, color='gray', edgecolor='none')
            
#             # Add styling as in the previous code...
#             # [Additional styling code here] 