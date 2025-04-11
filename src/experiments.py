import numpy as np
from tqdm import tqdm
from model import OpinionDynamicsModel, SUPPORTER, UNDECIDED, OPPOSITION
from networks import create_scale_free_network, create_small_world_network, create_random_network

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
                'history': []
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
                
                # Store full history for one representative trial
                if trial == 0:
                    results[network_type][strategy_name]['history'] = model.get_history_proportions()
    
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