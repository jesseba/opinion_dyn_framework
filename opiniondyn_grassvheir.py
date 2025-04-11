import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import pandas as pd
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import json
import pickle
import time
import datetime
from tqdm import tqdm

# Set better visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# Professional color scheme
SUPPORTER_COLOR = '#1f77b4'  # Blue
UNDECIDED_COLOR = '#2ca02c'  # Green
OPPOSITION_COLOR = '#d62728'  # Red
COLOR_PALETTE = [SUPPORTER_COLOR, UNDECIDED_COLOR, OPPOSITION_COLOR]

# Set random seed for reproducibility
np.random.seed(42)

# Define the states
SUPPORTER = 0
UNDECIDED = 1
OPPOSITION = 2

class OpinionDynamicsModel:
    def __init__(self, network, initial_states=None, lambda_s=0.12, lambda_o=0.12):
        """
        Initialize the opinion dynamics model.
        
        Parameters:
        -----------
        network : networkx.Graph
            The network on which the opinion dynamics will run
        initial_states : numpy.ndarray or None
            Initial states of the nodes. If None, will be randomly assigned.
        lambda_s : float
            Rate at which individuals move towards supporter state (O→U→S)
        lambda_o : float
            Rate at which individuals move towards opposition state (S→U→O)
        """
        self.network = network
        self.num_nodes = network.number_of_nodes()
        self.lambda_s = lambda_s
        self.lambda_o = lambda_o
        
        # Initialize states
        if initial_states is None:
            # Default: 30% supporters, 40% undecided, 30% opposition
            self.states = np.random.choice(
                [SUPPORTER, UNDECIDED, OPPOSITION], 
                size=self.num_nodes, 
                p=[0.3, 0.4, 0.3]
            )
        else:
            self.states = initial_states.copy()
        
        # Keep track of history
        self.history = [self.states.copy()]
        
        # Store node degrees for targeting strategies
        self.degrees = np.array([d for _, d in network.degree()])
        
        # Default: no individual-specific transition rates
        self.node_lambda_s = np.full(self.num_nodes, lambda_s)
        self.node_lambda_o = np.full(self.num_nodes, lambda_o)
        
        # Influence factors for transitions - adjusted for more dynamics
        self.s_influence = 0.15  # Influence of supporter neighbors
        self.u_influence = 0.15  # Increased influence of undecided neighbors
        self.o_influence = 0.15  # Influence of opposition neighbors
        
        # Stubbornness - reduces probability of changing opinion
        # Using a bimodal distribution with both persuadable and stubborn individuals
        self.stubbornness = np.random.beta(1.2, 3.0, size=self.num_nodes)
        
        # No special resistance for undecided voters anymore - they'll switch sides more readily
        # self.undecided_resistance = np.random.beta(2.5, 2.0, size=self.num_nodes)
        
        # Initialize opinion strength - how strongly each node holds its current opinion
        self.opinion_strength = np.random.beta(2, 2, size=self.num_nodes)
        
        # Track how long each node has held its current opinion
        # This increases stubbornness over time (opinions get entrenched)
        self.time_in_state = np.zeros(self.num_nodes)
    
    def apply_broadcast_shock(self, lambda_s_factor=1.0, lambda_o_factor=1.0):
        """
        Apply a broadcast shock that affects all nodes equally.
        
        Parameters:
        -----------
        lambda_s_factor : float
            Multiplicative factor for lambda_s
        lambda_o_factor : float
            Multiplicative factor for lambda_o
        """
        self.node_lambda_s = np.full(self.num_nodes, self.lambda_s * lambda_s_factor)
        self.node_lambda_o = np.full(self.num_nodes, self.lambda_o * lambda_o_factor)
    
    def apply_targeted_shock_high_degree(self, top_percent=0.05, lambda_s_factor=5.0, lambda_o_factor=1.0):
        """
        Apply a shock targeting the highest-degree nodes.
        
        Parameters:
        -----------
        top_percent : float
            Proportion of highest-degree nodes to target
        lambda_s_factor : float
            Multiplicative factor for lambda_s for targeted nodes
        lambda_o_factor : float
            Multiplicative factor for lambda_o for targeted nodes
        """
        # Reset to baseline
        self.node_lambda_s = np.full(self.num_nodes, self.lambda_s)
        self.node_lambda_o = np.full(self.num_nodes, self.lambda_o)
        
        # Determine threshold degree for top nodes
        k = int(self.num_nodes * top_percent)
        degree_threshold = np.sort(self.degrees)[-k]
        
        # Apply shock to high-degree nodes
        high_degree_nodes = np.where(self.degrees >= degree_threshold)[0]
        self.node_lambda_s[high_degree_nodes] *= lambda_s_factor
        self.node_lambda_o[high_degree_nodes] *= lambda_o_factor
        
        # Enhanced: reduce stubbornness of targeted high-degree nodes to make them more persuasive
        self.stubbornness[high_degree_nodes] *= 0.3  # More dramatic reduction
        
        # Enhanced: increase opinion strength of these influencers
        self.opinion_strength[high_degree_nodes] = np.minimum(self.opinion_strength[high_degree_nodes] * 2.0, 1.0)
        
        # Reset time in state for shocked nodes to make them more immediately influential
        self.time_in_state[high_degree_nodes] = 0
        
        # Convert some high-degree undecided nodes directly to supporters to create cascades
        undecided_high_degree = high_degree_nodes[self.states[high_degree_nodes] == UNDECIDED]
        if len(undecided_high_degree) > 0:
            convert_count = max(1, int(len(undecided_high_degree) * 0.5))
            convert_nodes = np.random.choice(undecided_high_degree, convert_count, replace=False)
            self.states[convert_nodes] = SUPPORTER
        
        # Critical for dynamics: also reduce stubbornness of nodes connected to high-degree nodes
        # This creates a ripple effect from the shock
        for node in high_degree_nodes:
            neighbors = list(self.network.neighbors(node))
            if neighbors:
                self.stubbornness[neighbors] *= 0.8
    
    def apply_targeted_shock_degree_proportional(self, lambda_s_factor=1.0, lambda_o_factor=1.0):
        """
        Apply a shock with strength proportional to node degree.
        
        Parameters:
        -----------
        lambda_s_factor : float
            Base multiplicative factor for lambda_s
        lambda_o_factor : float
            Base multiplicative factor for lambda_o
        """
        # Calculate degree-based multipliers
        degree_multipliers = 1 + self.degrees / np.mean(self.degrees)
        
        # Apply shock proportional to degree
        self.node_lambda_s = self.lambda_s * (1 + degree_multipliers * (lambda_s_factor - 1))
        self.node_lambda_o = self.lambda_o * (1 + degree_multipliers * (lambda_o_factor - 1))
    
    def apply_targeted_shock_random(self, target_percent=0.25, lambda_s_factor=3.0, lambda_o_factor=1.0):
        """
        Apply a shock targeting random nodes (simulating grassroots approach).
        
        Parameters:
        -----------
        target_percent : float
            Proportion of random nodes to target
        lambda_s_factor : float
            Multiplicative factor for lambda_s for targeted nodes
        lambda_o_factor : float
            Multiplicative factor for lambda_o for targeted nodes
        """
        # Reset to baseline
        self.node_lambda_s = np.full(self.num_nodes, self.lambda_s)
        self.node_lambda_o = np.full(self.num_nodes, self.lambda_o)
        
        # Randomly select nodes to target
        k = int(self.num_nodes * target_percent)
        targeted_nodes = np.random.choice(self.num_nodes, size=k, replace=False)
        
        # Apply shock to selected nodes
        self.node_lambda_s[targeted_nodes] *= lambda_s_factor
        self.node_lambda_o[targeted_nodes] *= lambda_o_factor
        
        # Enhanced: slightly reduce stubbornness of targeted random nodes
        self.stubbornness[targeted_nodes] *= 0.5  # More significant reduction
        
        # Enhanced: increase opinion strength of these grassroots campaigners
        self.opinion_strength[targeted_nodes] = np.minimum(self.opinion_strength[targeted_nodes] * 1.5, 1.0)
        
        # Reset time in state to make opinions more malleable
        self.time_in_state[targeted_nodes] = 0
        
        # Convert some targeted undecided nodes directly to supporters to seed cascades
        undecided_targeted = targeted_nodes[self.states[targeted_nodes] == UNDECIDED]
        if len(undecided_targeted) > 0:
            convert_count = max(1, int(len(undecided_targeted) * 0.4))
            convert_nodes = np.random.choice(undecided_targeted, convert_count, replace=False)
            self.states[convert_nodes] = SUPPORTER
    
    def reset_shocks(self):
        """Reset all shocks, returning to base transition rates."""
        self.node_lambda_s = np.full(self.num_nodes, self.lambda_s)
        self.node_lambda_o = np.full(self.num_nodes, self.lambda_o)
        
        # Reset other enhanced features that were modified during shocks
        # but don't reset opinion_strength as that should persist
        self.stubbornness = np.random.beta(1.2, 3.0, size=self.num_nodes)
    
    def step(self):
        """Perform one step of the opinion dynamics."""
        new_states = self.states.copy()
        
        # Increment time in current state for all nodes
        self.time_in_state += 1
        
        # Calculate social influence matrix (node x node) - precompute for efficiency
        social_influence = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            # Current state of the node
            current_state = self.states[i]
            
            # Get neighbors' states to calculate social influence
            neighbors = list(self.network.neighbors(i))
            if not neighbors:  # Skip if no neighbors
                continue
                
            neighbor_states = self.states[[n for n in neighbors]]
            neighbor_strengths = self.opinion_strength[[n for n in neighbors]]
            
            # Count weighted influence from neighbors in each state
            # Multiply by neighbor's opinion strength for more realistic dynamics
            n_supporter_influence = np.sum(
                (neighbor_states == SUPPORTER) * neighbor_strengths
            )
            n_undecided_influence = np.sum(
                (neighbor_states == UNDECIDED) * neighbor_strengths * 0.8  # Reduced influence from undecided
            )
            n_opposition_influence = np.sum(
                (neighbor_states == OPPOSITION) * neighbor_strengths  
            )
            
            # Total number of neighbors for normalizing influence
            total_neighbors = len(neighbors)
            
            # Calculate neighborhood composition ratios
            p_supporter = n_supporter_influence / total_neighbors if total_neighbors > 0 else 0
            p_undecided = n_undecided_influence / total_neighbors if total_neighbors > 0 else 0
            p_opposition = n_opposition_influence / total_neighbors if total_neighbors > 0 else 0
            
            # Entrenchment grows more slowly now
            entrenchment = min(self.time_in_state[i] / 50.0, 0.6)
            
            # Different stubbornness calculation based on current state
            if current_state == UNDECIDED:
                # Undecided voters now have much lower stubbornness - they'll switch sides more easily
                effective_stubbornness = 0.2 * entrenchment  # Very low resistance
            else:
                # Supporters and Opposition have regular stubbornness
                effective_stubbornness = self.stubbornness[i] + (1 - self.stubbornness[i]) * entrenchment
            
            # Cap stubbornness at a maximum value to maintain dynamics
            effective_stubbornness = min(effective_stubbornness, 0.85)
            stubbornness_factor = 1.0 - effective_stubbornness
            
            # Base transition rates (with time-dependent effects)
            base_s_prob = self.node_lambda_s[i]
            base_o_prob = self.node_lambda_o[i]
            
            # Critical point dynamics: make transitions more likely when opinions are balanced
            # This creates more interesting dynamics near 50-50 splits
            balance_effect = 1.0 + 0.5 * (4 * p_supporter * p_opposition)  # Maximum at 50/50 split
            
            # Transition probabilities based on current state and neighbor influence
            if current_state == SUPPORTER:
                # Supporter can transition to Undecided
                opposition_pressure = p_opposition * self.o_influence * balance_effect
                undecided_pressure = p_undecided * (self.u_influence/2)
                
                # Combined pressure from opposing views
                transition_prob = base_o_prob * stubbornness_factor * min(0.9, opposition_pressure + undecided_pressure)
                
                if np.random.random() < transition_prob:
                    new_states[i] = UNDECIDED
                    self.time_in_state[i] = 0  # Reset time in state
                    # Reduce opinion strength when changing opinion
                    self.opinion_strength[i] = max(self.opinion_strength[i] * 0.8, 0.2)
            
            elif current_state == UNDECIDED:
                # Undecided can transition to either Supporter or Opposition
                # More dynamics: strength of pull depends on neighbor composition
                
                # If neighborhood is polarized, undecided are more likely to pick a side
                polarization = abs(p_supporter - p_opposition)
                
                # Enhanced transition rates for undecided - they'll choose sides more quickly
                # Support pull - stronger with more supporter neighbors and higher polarization
                supporter_pull = p_supporter * self.s_influence * (1 + polarization) * 1.8  # Increased factor
                
                # Opposition pull - stronger with more opposition neighbors and higher polarization
                opposition_pull = p_opposition * self.o_influence * (1 + polarization) * 1.8  # Increased factor
                
                # No resistance factor for undecided anymore - they'll switch much more readily
                
                # Base probability plus social influence, with higher responsiveness
                prob_to_supporter = base_s_prob * supporter_pull * balance_effect * 1.5  # More responsive
                prob_to_opposition = base_o_prob * opposition_pull * balance_effect * 1.5  # More responsive
                
                # Normalize probabilities but with a higher cap to allow more transitions
                total_prob = prob_to_supporter + prob_to_opposition
                if total_prob > 0.9:  # Higher cap (was 0.75)
                    scaling = 0.9 / total_prob
                    prob_to_supporter *= scaling
                    prob_to_opposition *= scaling
                
                # Determine transition
                r = np.random.random()
                if r < prob_to_supporter:
                    new_states[i] = SUPPORTER
                    self.time_in_state[i] = 0
                    # Boost opinion strength when making a decision
                    self.opinion_strength[i] = min(self.opinion_strength[i] + 0.15, 1.0)
                elif r < prob_to_supporter + prob_to_opposition:
                    new_states[i] = OPPOSITION
                    self.time_in_state[i] = 0
                    # Boost opinion strength when making a decision
                    self.opinion_strength[i] = min(self.opinion_strength[i] + 0.15, 1.0)
            
            elif current_state == OPPOSITION:
                # Opposition can transition to Undecided
                supporter_pressure = p_supporter * self.s_influence * balance_effect
                undecided_pressure = p_undecided * (self.u_influence/2)
                
                transition_prob = base_s_prob * stubbornness_factor * min(0.9, supporter_pressure + undecided_pressure)
                
                if np.random.random() < transition_prob:
                    new_states[i] = UNDECIDED
                    self.time_in_state[i] = 0
                    # Reduce opinion strength when changing opinion
                    self.opinion_strength[i] = max(self.opinion_strength[i] * 0.8, 0.2)
        
        # Update states and history
        self.states = new_states
        self.history.append(self.states.copy())
    
    def run(self, steps=100, shock_start=None, shock_end=None, shock_func=None, shock_params=None):
        """
        Run the simulation for a number of steps.
        
        Parameters:
        -----------
        steps : int
            Number of steps to run
        shock_start : int or None
            Step at which to apply shock
        shock_end : int or None
            Step at which to remove shock
        shock_func : function or None
            Function to apply for shock
        shock_params : dict or None
            Parameters for shock function
        """
        for t in range(steps):
            # Apply shock if needed
            if shock_start is not None and t == shock_start and shock_func is not None:
                shock_func(**shock_params) if shock_params else shock_func()
            
            # Remove shock if needed
            if shock_end is not None and t == shock_end:
                self.reset_shocks()
            
            # Execute step
            self.step()
    
    def get_opinion_counts(self):
        """Get counts of each opinion."""
        unique, counts = np.unique(self.states, return_counts=True)
        result = {s: 0 for s in [SUPPORTER, UNDECIDED, OPPOSITION]}
        for s, c in zip(unique, counts):
            result[s] = c
        return result
    
    def get_opinion_proportions(self):
        """Get proportions of each opinion."""
        counts = self.get_opinion_counts()
        total = sum(counts.values())
        return {s: c/total for s, c in counts.items()}
    
    def get_history_proportions(self):
        """Get proportions of each opinion throughout history."""
        history_proportions = []
        for state in self.history:
            unique, counts = np.unique(state, return_counts=True)
            result = {s: 0 for s in [SUPPORTER, UNDECIDED, OPPOSITION]}
            for s, c in zip(unique, counts):
                result[s] = c / self.num_nodes
            history_proportions.append(result)
        return history_proportions
    
    def plot_opinion_evolution(self, title=None, shock_period=None):
        """Plot the evolution of opinions over time."""
        history_props = self.get_history_proportions()
        
        supporters = [h[SUPPORTER] for h in history_props]
        undecided = [h[UNDECIDED] for h in history_props]
        opposition = [h[OPPOSITION] for h in history_props]
        
        steps = range(len(self.history))
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, supporters, 'b-', label='Supporters')
        plt.plot(steps, undecided, 'g-', label='Undecided')
        plt.plot(steps, opposition, 'r-', label='Opposition')
        
        # Add shock period indicators if provided
        if shock_period:
            plt.axvspan(shock_period[0], shock_period[1], alpha=0.2, color='gray')
            plt.text((shock_period[0] + shock_period[1])/2, 0.95, 'Shock Period', 
                     horizontalalignment='center', fontsize=10)
        
        plt.xlabel('Time Step')
        plt.ylabel('Proportion')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if title:
            plt.title(title)
        else:
            plt.title('Evolution of Opinions Over Time')
        
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_network(self, ax=None, node_size=50, title=None):
        """Visualize the network with node colors representing opinions."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Define colors for each state
        colors = ['blue', 'green', 'red']
        node_colors = [colors[state] for state in self.states]
        
        # Scale node sizes based on degree and opinion strength
        scaled_sizes = [30 + (d * self.opinion_strength[i] * 20) for i, d in enumerate(self.degrees)]
        
        # Position nodes using a layout algorithm
        pos = nx.spring_layout(self.network, seed=42)
        
        # Draw the network
        nx.draw_networkx_nodes(self.network, pos, node_color=node_colors, 
                              node_size=scaled_sizes, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(self.network, pos, width=0.5, alpha=0.5, ax=ax)
        
        # Add a legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=c, markersize=10, label=l)
                          for c, l in zip(colors, ['Supporter', 'Undecided', 'Opposition'])]
        ax.legend(handles=legend_elements, loc='upper right')
        
        if title:
            ax.set_title(title)
        
        ax.set_axis_off()
        return ax


# Network generation functions
def create_scale_free_network(n=1000, m=3):
    """Create a scale-free network using Barabási-Albert model."""
    return nx.barabasi_albert_graph(n, m)

def create_small_world_network(n=1000, k=6, p=0.1):
    """Create a small-world network using Watts-Strogatz model."""
    return nx.watts_strogatz_graph(n, k, p)

def create_random_network(n=1000, p=None, k=6):
    """Create a random Erdős-Rényi network with similar average degree."""
    if p is None:
        # Calculate p to achieve target average degree k
        p = k / (n - 1)
    return nx.erdos_renyi_graph(n, p)


def run_grassroots_vs_establishment_experiment(
    n_nodes=1000, 
    shock_duration=20, 
    total_steps=100, 
    num_trials=5,
    lambda_s=0.12,  # Reduced base rates for more dynamics
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


def plot_experiment_results(results):
    """
    Plot the results from the grassroots vs. establishment experiment with publication-quality visualizations.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_grassroots_vs_establishment_experiment()
    """
    network_types = list(results.keys())
    strategies = list(results[network_types[0]].keys())
    
    # Setting up a more professional color scheme
    colors = [SUPPORTER_COLOR, UNDECIDED_COLOR, OPPOSITION_COLOR]
    opinion_labels = ['Supporters', 'Undecided', 'Opposition']
    
    # Create figure for barplots of final opinion distributions
    fig, axes = plt.subplots(1, len(network_types), figsize=(16, 6))
    
    for i, network_type in enumerate(network_types):
        ax = axes[i]
        
        # Data for plotting
        data = {
            'Strategy': [],
            'Opinion': [],
            'Proportion': []
        }
        
        for strategy in strategies:
            for opinion, idx in [('Supporters', SUPPORTER), ('Undecided', UNDECIDED), ('Opposition', OPPOSITION)]:
                # Fix: Use correct keys for each opinion
                opinion_key = 'supporter_final' if opinion == 'Supporters' else ('undecided_final' if opinion == 'Undecided' else 'opposition_final')
                values = results[network_type][strategy][opinion_key]
                mean_value = np.mean(values)
                
                # Add to data for seaborn plotting
                data['Strategy'].extend([strategy] * len(values))
                data['Opinion'].extend([opinion] * len(values))
                data['Proportion'].extend(values)
        
        # Plot with seaborn
        sns.barplot(x='Strategy', y='Proportion', hue='Opinion', 
                   data=data, ax=ax, 
                   palette=colors)
        
        # Enhanced styling
        network_type_titles = {
            'scale-free': 'Scale-Free Network\n(Urban Centers)',
            'small-world': 'Small-World Network\n(Suburban Areas)',
            'random': 'Random Network\n(Rural Communities)'
        }
        
        ax.set_title(network_type_titles.get(network_type, network_type.title()), fontsize=14, pad=10)
        ax.set_ylim(0, 1)
        ax.set_xlabel('')
        
        if i == 0:
            ax.set_ylabel('Proportion of Population', fontsize=12)
        else:
            ax.set_ylabel('')
        
        if i == len(network_types) - 1:
            ax.legend(title="Opinion", title_fontsize=12, fontsize=10, bbox_to_anchor=(1.05, 0.5), loc='center left')
        else:
            ax.legend([], [], frameon=False)
        
        # Enhanced x-axis labels
        strategy_labels = {
            'No shock': 'No Intervention',
            'Establishment (High-degree targets)': 'Establishment\nStrategy',
            'Grassroots (Random targets)': 'Grassroots\nStrategy'
        }
        ax.set_xticklabels([strategy_labels.get(s, s) for s in strategies])
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        
        # Add subtle grid
        ax.grid(axis='y', alpha=0.3)
        
        # Add data value labels on the bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', fontsize=8, color='black', 
                       xytext=(0, 1), textcoords='offset points')
    
    # Overall title
    plt.suptitle('Opinion Distribution Across Network Types and Intervention Strategies', 
                y=0.98, fontsize=16)

    
    plt.tight_layout()
    fig.subplots_adjust(top=0.85, bottom=0.15)
    
    # Create a second figure for opinion evolution over time
    fig2, axes2 = plt.subplots(len(network_types), len(strategies), figsize=(16, 12))
    
    # Add shock period indicator
    shock_start = 10
    shock_end = 10 + 20  # shock_duration
    
    for i, network_type in enumerate(network_types):
        for j, strategy in enumerate(strategies):
            ax = axes2[i, j]
            
            if 'history' in results[network_type][strategy]:
                history = results[network_type][strategy]['history']
                
                supporters = [h[SUPPORTER] for h in history]
                undecided = [h[UNDECIDED] for h in history]
                opposition = [h[OPPOSITION] for h in history]
                
                steps = range(len(history))
                
                # Plot with enhanced styling
                ax.plot(steps, supporters, '-', color=SUPPORTER_COLOR, linewidth=2.5, label='Supporters')
                ax.plot(steps, undecided, '-', color=UNDECIDED_COLOR, linewidth=2.5, label='Undecided')
                ax.plot(steps, opposition, '-', color=OPPOSITION_COLOR, linewidth=2.5, label='Opposition')
                
                # Highlight the shock period with better styling
                ax.axvspan(shock_start, shock_end, alpha=0.15, color='gray', edgecolor='none')
                
                # Add a vertical line at the shock start and end with labels
                if i == 0:  # Only add text labels on top row
                    ax.axvline(x=shock_start, color='gray', linestyle='--', alpha=0.7)
                    ax.axvline(x=shock_end, color='gray', linestyle='--', alpha=0.7)
                    ax.text(shock_start - 5, 0.95, 'Intervention\nStart', ha='right', va='top', 
                           color='black', fontsize=8, bbox=dict(facecolor='white', alpha=0.7, pad=2))
                    ax.text(shock_end + 5, 0.95, 'Intervention\nEnd', ha='left', va='top', 
                           color='black', fontsize=8, bbox=dict(facecolor='white', alpha=0.7, pad=2))
                
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                
                # Add titles
                if i == 0:
                    ax.set_title(strategy_labels.get(strategy, strategy), fontsize=14, pad=10)
                
                if j == 0:
                    ax.set_ylabel(f'{network_type_titles.get(network_type, network_type.title())}\nProportion', fontsize=12)
                
                # Add legend only on first plot
                if i == 0 and j == 0:
                    ax.legend(fontsize=10, loc='upper right')
                
                # Add x-label only on bottom row
                if i == len(network_types) - 1:
                    ax.set_xlabel('Time Step', fontsize=12)
                
                # Enhance tick labels
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Add annotations for final proportions
                final_s = supporters[-1]
                final_u = undecided[-1]
                final_o = opposition[-1]
                
                # Add final values annotation
                ax.annotate(f'Final: {final_s:.2f}', xy=(len(steps)-10, final_s), 
                           xytext=(5, 0), textcoords='offset points', 
                           color=SUPPORTER_COLOR, fontsize=8, ha='left', va='center')
                
                ax.annotate(f'Final: {final_o:.2f}', xy=(len(steps)-10, final_o), 
                           xytext=(5, 0), textcoords='offset points', 
                           color=OPPOSITION_COLOR, fontsize=8, ha='left', va='center')
    
    # Add a comprehensive title
    plt.suptitle('Opinion Evolution Over Time by Network Type and Intervention Strategy',
                y=0.98, fontsize=16)
    
    plt.tight_layout()
    fig2.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    return fig, fig2


def visualize_network_states(
    n_nodes=500, 
    shock_duration=20,
    network_type='scale-free'
):
    """
    Create network visualizations showing opinion distribution before, during, and after
    a targeted intervention.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in the network
    shock_duration : int
        Duration of the shock in time steps
    network_type : str
        Type of network to visualize ('scale-free', 'small-world', or 'random')
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the network visualizations
    """
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
        p=[0.3, 0.4, 0.3]
    )
    
    # Create model instances for each strategy
    establishment_model = OpinionDynamicsModel(
        network=network,
        initial_states=initial_states.copy(),
        lambda_s=0.12,
        lambda_o=0.12
    )
    
    grassroots_model = OpinionDynamicsModel(
        network=network,
        initial_states=initial_states.copy(),
        lambda_s=0.12,
        lambda_o=0.12
    )
    
    # Run models for 10 steps without shock (pre-intervention)
    establishment_model.run(steps=10)
    grassroots_model.run(steps=10)
    
    # Apply shocks
    establishment_model.apply_targeted_shock_high_degree(top_percent=0.05, lambda_s_factor=5.0)
    grassroots_model.apply_targeted_shock_random(target_percent=0.25, lambda_s_factor=3.0)
    
    # Run for shock duration
    establishment_model.run(steps=shock_duration)
    grassroots_model.run(steps=shock_duration)
    
    # Remove shocks
    establishment_model.reset_shocks()
    grassroots_model.reset_shocks()
    
    # Run for post-shock period
    establishment_model.run(steps=30)
    grassroots_model.run(steps=30)
    
    # Create visualization figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Visualize the network at 3 time points for each strategy
    # Establishment strategy - before shock
    establishment_model.visualize_network(
        ax=axes[0, 0], 
        title="Establishment Strategy\nBefore Intervention (t=10)"
    )
    
    # Establishment strategy - during shock
    establishment_model.visualize_network(
        ax=axes[0, 1], 
        title="Establishment Strategy\nDuring Intervention (t=30)"
    )
    
    # Establishment strategy - after shock
    establishment_model.visualize_network(
        ax=axes[0, 2], 
        title="Establishment Strategy\nAfter Intervention (t=60)"
    )
    
    # Grassroots strategy - before shock
    grassroots_model.visualize_network(
        ax=axes[1, 0], 
        title="Grassroots Strategy\nBefore Intervention (t=10)"
    )
    
    # Grassroots strategy - during shock
    grassroots_model.visualize_network(
        ax=axes[1, 1], 
        title="Grassroots Strategy\nDuring Intervention (t=30)"
    )
    
    # Grassroots strategy - after shock
    grassroots_model.visualize_network(
        ax=axes[1, 2], 
        title="Grassroots Strategy\nAfter Intervention (t=60)"
    )
    
    plt.tight_layout()
    return fig


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


def plot_battleground_results(results, save_path="figures/battleground_results.pdf"):
    """
    Plot the results from the network battleground experiment with enhanced visualizations.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_network_battleground_experiment()
    save_path : str
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plots
    """
    network_types = list(results.keys())
    
    # Create figure with GridSpec for more flexible layout
    fig = plt.figure(figsize=(16, 8))
    
    # Create a simple 1x2 grid instead of 2x2
    gs = GridSpec(1, 2, figure=fig)
    
    # Left: Supporter gain bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Format data for plotting
    gain_means = [np.mean(results[nt]['supporter_gain']) for nt in network_types]
    gain_std = [np.std(results[nt]['supporter_gain']) for nt in network_types]
    
    # Enhanced bar colors - use a blue gradient for supporter gain
    bar_colors = ['#6baed6', '#4292c6', '#2171b5']
    
    # Plot bars with error
    bars = ax1.bar(range(len(network_types)), gain_means, yerr=gain_std, 
                  capsize=10, color=bar_colors, edgecolor='black', linewidth=1)
    
    # Add data labels on top of bars
    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + gain_std[i] + 0.01,
                f'{gain_means[i]:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Enhanced styling
    ax1.set_title('Average Supporter Gain by Network Type', fontsize=16, pad=15)
    ax1.set_ylabel('Net Supporter Gain', fontsize=14)
    ax1.set_xticks(range(len(network_types)))
    
    # Better x-axis labels with descriptions
    network_labels = {
        'urban_center': 'Urban Centers\n(Scale-Free)',
        'suburban_area': 'Suburban Areas\n(Small-World)',
        'rural_community': 'Rural Communities\n(Random)'
    }
    ax1.set_xticklabels([network_labels.get(nt, nt) for nt in network_types], fontsize=12)
    
    # Add subtle grid
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Right: Campaign efficiency bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Format data for plotting
    efficiency_means = [np.mean(results[nt]['resource_efficiency']) for nt in network_types]
    efficiency_std = [np.std(results[nt]['resource_efficiency']) for nt in network_types]
    
    # Enhanced bar colors - use a green gradient for efficiency
    efficiency_colors = ['#74c476', '#41ab5d', '#238b45']
    
    # Plot bars with error
    bars = ax2.bar(range(len(network_types)), efficiency_means, yerr=efficiency_std, 
                  capsize=10, color=efficiency_colors, edgecolor='black', linewidth=1)
    
    # Add data labels on top of bars
    for i, bar in enumerate(bars):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + efficiency_std[i] + 0.2,
                f'{efficiency_means[i]:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Enhanced styling
    ax2.set_title('Campaign Efficiency by Network Type', fontsize=16, pad=15)
    ax2.set_ylabel('Gain per Resource Unit', fontsize=14)
    ax2.set_xticks(range(len(network_types)))
    ax2.set_xticklabels([network_labels.get(nt, nt) for nt in network_types], fontsize=12)
    
    # Add subtle grid
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    plt.suptitle('Strategic Resource Allocation for Opinion Campaigns', fontsize=18, y=0.98)
    fig.subplots_adjust(top=0.85)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {save_path}")
    
    return fig


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


def plot_timing_results(results, save_path="figures/timing_experiment.pdf"):
    """
    Plot the results from the timing experiment with publication-quality visualizations
    including confidence intervals and proper error bars.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_timing_experiment()
    save_path : str
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plots
    """
    # Set seaborn style for publication quality
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    
    timing_labels = list(results.keys())
    
    # Create a more comprehensive figure with GridSpec
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1])
    
    # Top: Time evolution with confidence intervals
    ax1 = fig.add_subplot(gs[0, :])
    
    # Define timing-specific colors and styles
    colors = {'Early Campaign': '#1f77b4', 'Late Campaign': '#d62728'}
    
    for i, timing in enumerate(timing_labels):
        if 'all_histories' in results[timing] and len(results[timing]['all_histories']) > 0:
            # Calculate confidence intervals from all trials
            num_steps = len(results[timing]['all_histories'][0])
            supporters_data = np.zeros((len(results[timing]['all_histories']), num_steps))
            
            # Collect data from all trials
            for k, hist in enumerate(results[timing]['all_histories']):
                if len(hist) == num_steps:  # Ensure same length
                    supporters_data[k] = [h[SUPPORTER] for h in hist]
            
            # Calculate means and standard deviations
            supporters_mean = np.mean(supporters_data, axis=0)
            supporters_std = np.std(supporters_data, axis=0)
            
            steps = range(num_steps)
            
            # Plot mean line
            ax1.plot(steps, supporters_mean, '-', color=colors[timing], 
                    linewidth=2.5, label=f'{timing}')
            
            # Add confidence interval (mean ± 1 std)
            ax1.fill_between(steps, 
                           supporters_mean - supporters_std, 
                           supporters_mean + supporters_std, 
                           color=colors[timing], alpha=0.2)
            
            # Mark intervention period if known
            shock_start = None
            shock_duration = None
            
            if timing == 'Early Campaign':
                shock_start = 10
                shock_duration = 20
            elif timing == 'Late Campaign':
                shock_start = 50
                shock_duration = 20
                
            if shock_start is not None and shock_duration is not None:
                ax1.axvspan(shock_start, shock_start + shock_duration, 
                          color=colors[timing], alpha=0.15, edgecolor=colors[timing],
                          linewidth=1, zorder=1)
                
                # Add label for intervention period
                ax1.text(shock_start + shock_duration/2, 0.98, timing,
                       ha='center', va='top', color=colors[timing], fontsize=10,
                       transform=ax1.get_xaxis_transform(),
                       bbox=dict(facecolor='white', alpha=0.7, pad=2))
        
        # Fallback to using history if all_histories is not available
        elif 'history' in results[timing]:
            history = results[timing]['history']
            supporters = np.array([h[SUPPORTER] for h in history])
            steps = range(len(history))
            
            # Plot with consistent colors
            ax1.plot(steps, supporters, '-', color=colors[timing], linewidth=2.5, label=timing)
    
    # Add styling
    ax1.set_title('Impact of Intervention Timing on Opinion Evolution', fontsize=16, pad=15)
    ax1.set_xlabel('Time Step', fontsize=14)
    ax1.set_ylabel('Supporter Proportion', fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Timing Strategy', fontsize=12, loc='upper left')
    
    # Bottom left: Bar chart of final supporter proportions
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Calculate means and standard errors for bar chart
    supporter_means = [np.mean(results[timing]['supporter_final']) for timing in timing_labels]
    supporter_stds = [np.std(results[timing]['supporter_final']) for timing in timing_labels]
    
    # Plot bars with error bars
    bars = ax2.bar(range(len(timing_labels)), supporter_means, yerr=supporter_stds, 
                  capsize=10, color=[colors[timing] for timing in timing_labels], 
                  edgecolor='black', linewidth=1)
    
    # Add data labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + supporter_stds[i] + 0.01,
                f'{supporter_means[i]:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add styling
    ax2.set_title('Final Supporter Proportion', fontsize=14)
    ax2.set_ylabel('Proportion', fontsize=12)
    ax2.set_xticks(range(len(timing_labels)))
    ax2.set_xticklabels(timing_labels, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Bottom right: Campaign effectiveness metrics
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Calculate gain metrics (final - initial)
    initial_supporter = 0.3  # From initial_states proportion in run_timing_experiment
    gain_means = [mean - initial_supporter for mean in supporter_means]
    gain_stds = supporter_stds  # Same std for the difference
    
    # Calculate efficiency metrics (gain per unit of intervention)
    efficiency_means = [gain / 20 for gain in gain_means]  # Assuming 20 time steps of intervention
    efficiency_stds = [std / 20 for std in gain_stds]
    
    # Plot bars with error bars - different metric
    ax3.bar([0, 1], gain_means, yerr=gain_stds, 
           capsize=10, color=['#aec7e8', '#ff9896'], 
           edgecolor='black', linewidth=1, label='Absolute Gain')
    
    # Add second set of bars for efficiency
    ax3_2 = ax3.twinx()
    ax3_2.bar([2, 3], efficiency_means, yerr=efficiency_stds,
             capsize=10, color=['#c5dbef', '#ffbcba'], 
             edgecolor='black', linewidth=1, label='Efficiency')
    
    # Add styling
    ax3.set_title('Campaign Effectiveness', fontsize=14)
    ax3.set_ylabel('Absolute Gain', fontsize=12)
    ax3_2.set_ylabel('Gain per Time Step', fontsize=12)
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xticklabels(['Early\nGain', 'Late\nGain', 'Early\nEfficiency', 'Late\nEfficiency'], 
                      fontsize=9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Create legend with both metrics
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_2.get_legend_handles_labels()
    ax3_2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # Add a network science-oriented header
    plt.suptitle('Temporal Dynamics of Opinion Spread in Scale-Free Networks', 
                fontsize=18, y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.35)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {save_path}")
    
    return fig

def run_cross_network_intervention_comparison(
    n_nodes=1000,
    shock_duration=20,
    total_steps=150,
    num_trials=5,
    lambda_s=0.12,
    lambda_o=0.12
):
    """
    Run comprehensive experiment comparing intervention strategies across different network topologies.
    This provides rigorous analysis for network science presentation.
    
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
    # Network types to compare
    network_types = {
        'Scale-free': {'create_func': create_scale_free_network, 'params': {'n': n_nodes, 'm': 3}},
        'Small-world': {'create_func': create_small_world_network, 'params': {'n': n_nodes, 'k': 6, 'p': 0.1}},
        'Random': {'create_func': create_random_network, 'params': {'n': n_nodes, 'k': 6}}
    }
    
    # Intervention strategies
    interventions = {
        'No Intervention': {
            'func': None,
            'params': {}
        },
        'Broadcast': {
            'func': 'apply_broadcast_shock',
            'params': {'lambda_s_factor': 3.0}
        },
        'High-Degree Targeting': {
            'func': 'apply_targeted_shock_high_degree',
            'params': {'top_percent': 0.05, 'lambda_s_factor': 5.0}
        },
        'Degree-Proportional': {
            'func': 'apply_targeted_shock_degree_proportional',
            'params': {'lambda_s_factor': 3.0}
        },
        'Random Targeting': {
            'func': 'apply_targeted_shock_random',
            'params': {'target_percent': 0.25, 'lambda_s_factor': 3.0}
        }
    }
    
    # Results storage - network_type -> intervention -> metrics
    results = {}
    
    for network_name, network_info in network_types.items():
        print(f"Running simulations for {network_name} network...")
        results[network_name] = {}
        
        for int_name, int_info in interventions.items():
            print(f"  Strategy: {int_name}")
            results[network_name][int_name] = {
                'supporter_final': [],
                'undecided_final': [],
                'opposition_final': [],
                'supporter_history': [],
                'undecided_history': [],
                'opposition_history': [],
                'all_histories': [],
                'network_metrics': []
            }
            
            for trial in range(num_trials):
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
                
                # Calculate initial network metrics
                initial_metrics = calculate_network_metrics(model)
                
                # Run simulation with appropriate intervention
                if int_name == 'No Intervention':
                    # Just run without intervention
                    model.run(steps=total_steps)
                else:
                    # Run until intervention start
                    shock_start = 30
                    model.run(steps=shock_start)
                    
                    # Apply intervention
                    func_name = int_info['func']
                    if hasattr(model, func_name):
                        getattr(model, func_name)(**int_info['params'])
                    
                    # Run during intervention
                    model.run(steps=shock_duration)
                    
                    # Reset after intervention
                    model.reset_shocks()
                    
                    # Run remaining steps
                    model.run(steps=total_steps - shock_start - shock_duration)
                
                # Store results
                final_props = model.get_opinion_proportions()
                results[network_name][int_name]['supporter_final'].append(final_props[SUPPORTER])
                results[network_name][int_name]['undecided_final'].append(final_props[UNDECIDED])
                results[network_name][int_name]['opposition_final'].append(final_props[OPPOSITION])
                
                # Store history
                history = model.get_history_proportions()
                results[network_name][int_name]['all_histories'].append(history)
                
                # Store first trial history as representative
                if trial == 0:
                    results[network_name][int_name]['supporter_history'] = [h[SUPPORTER] for h in history]
                    results[network_name][int_name]['undecided_history'] = [h[UNDECIDED] for h in history]
                    results[network_name][int_name]['opposition_history'] = [h[OPPOSITION] for h in history]
                
                # Store network metrics
                final_metrics = calculate_network_metrics(model)
                results[network_name][int_name]['network_metrics'].append({
                    'initial': initial_metrics,
                    'final': final_metrics
                })
    
    return results

def plot_cross_network_results(results, save_path="figures/cross_network_comparison.pdf"):
    """
    Plot comprehensive comparison of intervention strategies across network types.
    Creates publication-quality visualizations with proper statistical rigor.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_cross_network_intervention_comparison()
    save_path : str
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plots
    """
    # Set seaborn style for publication quality
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    
    network_types = list(results.keys())
    if not network_types:
        return None
        
    # Get intervention strategies from first network type
    interventions = list(results[network_types[0]].keys())
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Top left: Bar chart of final supporter proportions across networks and interventions
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Prepare data for grouped bar chart
    bar_positions = []
    bar_heights = []
    bar_errors = []
    bar_colors = []
    bar_labels = []
    
    # Color palette for networks
    network_palette = sns.color_palette("husl", len(network_types))
    network_colors = {network: network_palette[i] for i, network in enumerate(network_types)}
    
    # X positions for groups of bars
    group_width = 0.8
    bar_width = group_width / len(network_types)
    offsets = np.linspace(-group_width/2 + bar_width/2, group_width/2 - bar_width/2, len(network_types))
    
    for i, intervention in enumerate(interventions):
        for j, network in enumerate(network_types):
            # Calculate mean and std
            supporter_mean = np.mean(results[network][intervention]['supporter_final'])
            supporter_std = np.std(results[network][intervention]['supporter_final'])
            
            # Store position and value
            bar_positions.append(i + offsets[j])
            bar_heights.append(supporter_mean)
            bar_errors.append(supporter_std)
            bar_colors.append(network_colors[network])
            bar_labels.append(f"{network}-{intervention}")
    
    # Plot bars with error bars
    bars = ax1.bar(bar_positions, bar_heights, yerr=bar_errors, 
                  width=bar_width, color=bar_colors, 
                  edgecolor='black', linewidth=1, capsize=3)
    
    # Add styling
    ax1.set_title('Final Supporter Proportion by Intervention and Network', fontsize=14)
    ax1.set_ylabel('Supporter Proportion', fontsize=12)
    ax1.set_xticks(range(len(interventions)))
    ax1.set_xticklabels(interventions, fontsize=10, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    
    # Custom legend for network types
    network_handles = [plt.Rectangle((0,0),1,1, color=network_colors[nt]) for nt in network_types]
    ax1.legend(network_handles, network_types, title='Network Type', 
              fontsize=9, loc='upper left')
    
    # Top right: Line chart showing opinion evolution over time for different networks
    # Focus on one intervention strategy to keep it clear
    focus_intervention = 'High-Degree Targeting'
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, network in enumerate(network_types):
        # Check if we have multiple histories for confidence intervals
        if 'all_histories' in results[network][focus_intervention] and len(results[network][focus_intervention]['all_histories']) > 0:
            # Calculate confidence intervals
            histories = results[network][focus_intervention]['all_histories']
            num_steps = len(histories[0])
            supporters_data = np.zeros((len(histories), num_steps))
            
            # Collect data from all trials
            for k, hist in enumerate(histories):
                if len(hist) == num_steps:
                    supporters_data[k] = [h[SUPPORTER] for h in hist]
            
            # Calculate means and standard deviations
            supporters_mean = np.mean(supporters_data, axis=0)
            supporters_std = np.std(supporters_data, axis=0)
            
            steps = range(num_steps)
            
            # Plot mean line
            ax2.plot(steps, supporters_mean, '-', color=network_colors[network],
                    linewidth=2, label=network)
            
            # Add confidence interval
            ax2.fill_between(steps, 
                           supporters_mean - supporters_std, 
                           supporters_mean + supporters_std, 
                           color=network_colors[network], alpha=0.2)
        else:
            # Fallback to representative history
            history = results[network][focus_intervention]['supporter_history']
            steps = range(len(history))
            ax2.plot(steps, history, '-', color=network_colors[network],
                    linewidth=2, label=network)
    
    # Highlight intervention period
    ax2.axvspan(30, 30 + shock_duration, color='gray', alpha=0.2, 
               label='Intervention Period')
    
    # Add styling
    ax2.set_title(f'Opinion Evolution: {focus_intervention} Strategy', fontsize=14)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Supporter Proportion', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend(title='Network Type', fontsize=9, loc='upper left')
    
    # Bottom left: Network metrics comparison
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Prepare data for clustering coefficient and homophily change
    clustering_changes = []
    homophily_changes = []
    network_labels = []
    colors = []
    
    for network in network_types:
        for intervention in interventions:
            if len(results[network][intervention]['network_metrics']) > 0:
                metrics = results[network][intervention]['network_metrics']
                
                # Calculate average changes in metrics
                clustering_change = np.mean([m['final']['clustering'] - m['initial']['clustering'] for m in metrics])
                homophily_change = np.mean([m['final']['homophily'] - m['initial']['homophily'] for m in metrics])
                
                clustering_changes.append(clustering_change)
                homophily_changes.append(homophily_change)
                network_labels.append(f"{network}-{intervention}")
                colors.append(network_colors[network])
    
    # Create scatter plot
    ax3.scatter(homophily_changes, clustering_changes, c=colors, s=100, alpha=0.7, edgecolor='black')
    
    # Label points
    for i, label in enumerate(network_labels):
        ax3.annotate(label, (homophily_changes[i], clustering_changes[i]),
                    fontsize=8, ha='right', va='bottom',
                    xytext=(5, 5), textcoords='offset points')
    
    # Add styling
    ax3.set_title('Network Structural Changes by Intervention', fontsize=14)
    ax3.set_xlabel('Change in Opinion Homophily', fontsize=12)
    ax3.set_ylabel('Change in Clustering Coefficient', fontsize=12)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # Bottom right: Bar chart comparing intervention effectiveness
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate intervention effectiveness across network types
    effect_data = []
    
    for intervention in interventions:
        # Skip No Intervention as baseline
        if intervention == 'No Intervention':
            continue
            
        for network in network_types:
            # Calculate effectiveness as difference from No Intervention
            baseline = np.mean(results[network]['No Intervention']['supporter_final'])
            with_intervention = np.mean(results[network][intervention]['supporter_final'])
            effectiveness = with_intervention - baseline
            
            # Calculate standard error
            baseline_std = np.std(results[network]['No Intervention']['supporter_final'])
            intervention_std = np.std(results[network][intervention]['supporter_final'])
            # Combined standard error (approximation)
            std_error = np.sqrt(baseline_std**2 + intervention_std**2) / np.sqrt(num_trials)
            
            effect_data.append({
                'Network': network,
                'Intervention': intervention,
                'Effectiveness': effectiveness,
                'StdError': std_error
            })
    
    # Convert to DataFrame for seaborn
    import pandas as pd
    df = pd.DataFrame(effect_data)
    
    # Plot with seaborn
    sns.barplot(x='Intervention', y='Effectiveness', hue='Network', data=df, 
               palette=network_colors, errorbar='sd', ax=ax4)
    
    # Add styling
    ax4.set_title('Intervention Effectiveness by Network Type', fontsize=14)
    ax4.set_xlabel('Intervention Strategy', fontsize=12)
    ax4.set_ylabel('Effectiveness (vs. No Intervention)', fontsize=12)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # Add a network science-oriented header
    plt.suptitle('Network Topology Effects on Opinion Intervention Strategies', 
                fontsize=18, y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, wspace=0.3, hspace=0.3)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {save_path}")
    
    return fig


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
    
    # Opinion clusters: count connected components of same-opinion nodes
    supporter_subgraph = model.network.subgraph([n for n in range(model.num_nodes) if model.states[n] == SUPPORTER])
    undecided_subgraph = model.network.subgraph([n for n in range(model.num_nodes) if model.states[n] == UNDECIDED])
    opposition_subgraph = model.network.subgraph([n for n in range(model.num_nodes) if model.states[n] == OPPOSITION])
    
    metrics['supporter_clusters'] = nx.number_connected_components(supporter_subgraph)
    metrics['undecided_clusters'] = nx.number_connected_components(undecided_subgraph)
    metrics['opposition_clusters'] = nx.number_connected_components(opposition_subgraph)
    
    # Influential nodes analysis
    # Calculate weighted centrality based on both network structure and opinion strength
    centrality = nx.degree_centrality(model.network)
    weighted_centrality = {node: centrality[node] * model.opinion_strength[node] for node in range(model.num_nodes)}
    
    # Most influential nodes by opinion
    supporter_nodes = [n for n in range(model.num_nodes) if model.states[n] == SUPPORTER]
    undecided_nodes = [n for n in range(model.num_nodes) if model.states[n] == UNDECIDED]
    opposition_nodes = [n for n in range(model.num_nodes) if model.states[n] == OPPOSITION]
    
    if supporter_nodes:
        metrics['top_supporter_influence'] = max(weighted_centrality[n] for n in supporter_nodes)
    else:
        metrics['top_supporter_influence'] = 0
        
    if undecided_nodes:
        metrics['top_undecided_influence'] = max(weighted_centrality[n] for n in undecided_nodes)
    else:
        metrics['top_undecided_influence'] = 0
        
    if opposition_nodes:
        metrics['top_opposition_influence'] = max(weighted_centrality[n] for n in opposition_nodes)
    else:
        metrics['top_opposition_influence'] = 0
    
    return metrics

def track_network_metrics_over_time(model, steps=100, shock_start=None, shock_end=None, shock_func=None, shock_params=None):
    """
    Run a simulation and track network metrics over time.
    
    Parameters:
    -----------
    model : OpinionDynamicsModel
        The model to simulate
    steps : int
        Number of steps to run
    shock_start, shock_end, shock_func, shock_params : See model.run()
    
    Returns:
    --------
    dict
        Dictionary with metrics over time
    """
    # Store metrics over time
    metrics_history = {
        'homophily': [],
        'polarization': [],
        'segregation_index': [],
        'supporter_clusters': [],
        'undecided_clusters': [],
        'opposition_clusters': [],
        'top_supporter_influence': [],
        'top_undecided_influence': [],
        'top_opposition_influence': []
    }
    
    # Initialize with current metrics
    initial_metrics = calculate_network_metrics(model)
    for key in metrics_history:
        metrics_history[key].append(initial_metrics[key])
    
    # Run the simulation and track metrics
    for t in range(steps):
        # Apply shock if needed
        if shock_start is not None and t == shock_start and shock_func is not None:
            shock_func(**shock_params) if shock_params else shock_func()
        
        # Remove shock if needed
        if shock_end is not None and t == shock_end:
            model.reset_shocks()
        
        # Execute step
        model.step()
        
        # Calculate and store metrics
        step_metrics = calculate_network_metrics(model)
        for key in metrics_history:
            metrics_history[key].append(step_metrics[key])
    
    return metrics_history

def plot_enhanced_network_metrics(metrics_history, title=None, shock_period=None):
    """
    Create a publication-quality plot of network metrics over time.
    
    Parameters:
    -----------
    metrics_history : dict
        Dictionary with metrics over time from track_network_metrics_over_time()
    title : str or None
        Main title for the figure
    shock_period : tuple or None
        (start, end) of shock period to highlight
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plots
    """
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, figure=fig)
    
    # Plot 1: Homophily
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metrics_history['homophily'], color='#ff7f0e', linewidth=2.5)
    ax1.set_ylabel('Homophily')
    ax1.set_title('Opinion Similarity in Connections')
    
    # Plot 2: Polarization
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(metrics_history['polarization'], color='#d62728', linewidth=2.5)
    ax2.set_ylabel('Polarization')
    ax2.set_title('Supporter-Opposition Connections')
    
    # Plot 3: Segregation Index
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(metrics_history['segregation_index'], color='#9467bd', linewidth=2.5)
    ax3.set_ylabel('Segregation Index')
    ax3.set_title('Opinion Clustering vs Random')
    
    # Plot 4: Opinion Clusters
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(metrics_history['supporter_clusters'], color=SUPPORTER_COLOR, linewidth=2.5, label='Supporter Clusters')
    ax4.plot(metrics_history['undecided_clusters'], color=UNDECIDED_COLOR, linewidth=2.5, label='Undecided Clusters')
    ax4.plot(metrics_history['opposition_clusters'], color=OPPOSITION_COLOR, linewidth=2.5, label='Opposition Clusters')
    ax4.set_ylabel('Number of Clusters')
    ax4.set_title('Opinion Fragmentation Over Time')
    ax4.legend(loc='upper right')
    
    # Plot 5: Influence by Opinion
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(metrics_history['top_supporter_influence'], color=SUPPORTER_COLOR, linewidth=2.5, label='Supporter Influence')
    ax5.plot(metrics_history['top_undecided_influence'], color=UNDECIDED_COLOR, linewidth=2.5, label='Undecided Influence')
    ax5.plot(metrics_history['top_opposition_influence'], color=OPPOSITION_COLOR, linewidth=2.5, label='Opposition Influence')
    ax5.set_ylabel('Influence Score')
    ax5.set_xlabel('Time Step')
    ax5.set_title('Top Influencer Scores by Opinion')
    ax5.legend(loc='upper right')
    
    # Add shock period highlighting if provided
    if shock_period:
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.axvspan(shock_period[0], shock_period[1], alpha=0.2, color='gray')
            if ax == ax4:  # Only add label on one plot
                ax.text((shock_period[0] + shock_period[1])/2, ax.get_ylim()[1]*0.9, 'Intervention', 
                        horizontalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=18, y=0.98)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9 if title else 0.95)
    
    return fig

def analyze_influence_pathways(model, influenced_nodes, time_steps=3):
    """
    Analyze the pathways of influence from initial influenced nodes through the network.
    
    Parameters:
    -----------
    model : OpinionDynamicsModel
        The model to analyze
    influenced_nodes : list
        List of node indices that were directly influenced (e.g., by a shock)
    time_steps : int
        Number of time steps to trace influence
    
    Returns:
    --------
    dict
        Influence pathways and statistics
    """
    results = {
        'first_order': set(influenced_nodes),
        'second_order': set(),
        'higher_order': set(),
        'pathway_stats': {}
    }
    
    # First identify first-order connections (direct neighbors)
    first_order_neighbors = set()
    for node in influenced_nodes:
        neighbors = set(model.network.neighbors(node))
        first_order_neighbors.update(neighbors)
    
    # Remove already influenced nodes
    first_order_neighbors -= results['first_order']
    results['second_order'] = first_order_neighbors
    
    # Then identify higher-order connections
    current_set = first_order_neighbors
    higher_order = set()
    for _ in range(time_steps - 1):
        next_level = set()
        for node in current_set:
            neighbors = set(model.network.neighbors(node))
            next_level.update(neighbors)
        
        # Remove already tracked nodes
        next_level -= results['first_order']
        next_level -= results['second_order']
        next_level -= higher_order
        higher_order.update(next_level)
        current_set = next_level
    
    results['higher_order'] = higher_order
    
    # Calculate influence spread statistics
    total_influenced = len(results['first_order']) + len(results['second_order']) + len(results['higher_order'])
    results['pathway_stats'] = {
        'total_influenced': total_influenced,
        'percent_influenced': total_influenced / model.num_nodes * 100,
        'first_order_count': len(results['first_order']),
        'second_order_count': len(results['second_order']),
        'higher_order_count': len(results['higher_order']),
        'average_path_length': nx.average_shortest_path_length(model.network) if nx.is_connected(model.network) else 'N/A',
    }
    
    return results

def plot_influence_pathways(model, influence_results, ax=None, title=None):
    """
    Create a visualization of influence pathways through the network.
    
    Parameters:
    -----------
    model : OpinionDynamicsModel
        The model containing the network
    influence_results : dict
        Results from analyze_influence_pathways()
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates a new figure.
    title : str or None
        Title for the plot
    
    Returns:
    --------
    matplotlib.axes.Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create position layout
    pos = nx.spring_layout(model.network, seed=42)
    
    # Define node colors by opinion
    opinion_colors = [COLOR_PALETTE[state] for state in model.states]
    
    # Define node sizes by pathway level
    node_sizes = np.ones(model.num_nodes) * 50  # Default size
    node_sizes[[n for n in influence_results['first_order']]] = 200  # Source nodes
    node_sizes[[n for n in influence_results['second_order']]] = 120  # First-level influenced
    node_sizes[[n for n in influence_results['higher_order']]] = 80  # Higher-level influenced
    
    # Plot edges
    nx.draw_networkx_edges(model.network, pos, alpha=0.2, ax=ax)
    
    # Plot nodes
    nx.draw_networkx_nodes(model.network, pos, 
                           node_color=opinion_colors,
                           node_size=node_sizes,
                           alpha=0.8, ax=ax)
    
    # Highlight source nodes with a border
    nx.draw_networkx_nodes(model.network, pos,
                          nodelist=list(influence_results['first_order']),
                          node_color='white',
                          node_size=220,
                          alpha=1.0, ax=ax)
    nx.draw_networkx_nodes(model.network, pos,
                          nodelist=list(influence_results['first_order']),
                          node_color=[opinion_colors[n] for n in influence_results['first_order']],
                          node_size=200,
                          alpha=1.0, ax=ax)
    
    # Add legend for pathway levels
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', label='Source Nodes'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='First-Level Influence'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.4, label='Higher-Level Influence')
    ]
    legend1 = ax.legend(handles=legend_elements, title="Influence Pathways", 
                       loc='upper left', bbox_to_anchor=(0, 1))
    ax.add_artist(legend1)
    
    # Add legend for opinions
    legend_elements = [
        Patch(facecolor=SUPPORTER_COLOR, label='Supporter'),
        Patch(facecolor=UNDECIDED_COLOR, label='Undecided'),
        Patch(facecolor=OPPOSITION_COLOR, label='Opposition')
    ]
    ax.legend(handles=legend_elements, title="Opinions", 
             loc='upper right', bbox_to_anchor=(1, 1))
    
    # Add title
    if title:
        ax.set_title(title)
    
    # Add influence statistics as text
    stats_text = (
        f"Influence Coverage: {influence_results['pathway_stats']['percent_influenced']:.1f}%\n"
        f"Source Nodes: {influence_results['pathway_stats']['first_order_count']}\n"
        f"First-Level: {influence_results['pathway_stats']['second_order_count']}\n"
        f"Higher-Level: {influence_results['pathway_stats']['higher_order_count']}"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.set_axis_off()
    return ax

def create_network_analysis_dashboard(model, shock_func=None, shock_params=None, network_type="Unknown"):
    """
    Create a comprehensive dashboard of network analysis visualizations.
    
    Parameters:
    -----------
    model : OpinionDynamicsModel
        The model to analyze
    shock_func : function or None
        Shock function to apply
    shock_params : dict or None
        Parameters for shock function
    network_type : str
        Name of the network type for titles
    
    Returns:
    --------
    matplotlib.figure.Figure
        Dashboard figure
    """
    # Create a copy of the model to run simulations without affecting the original
    model_copy = OpinionDynamicsModel(
        network=model.network,
        initial_states=model.states.copy(),
        lambda_s=model.lambda_s,
        lambda_o=model.lambda_o
    )
    
    # Set up simulation parameters
    steps = 100
    shock_start = 20
    shock_end = 40
    
    # Store original metrics for comparison
    initial_metrics = calculate_network_metrics(model_copy)
    
    # Run simulation with shock
    if shock_func:
        metrics_history = track_network_metrics_over_time(
            model_copy, 
            steps=steps,
            shock_start=shock_start,
            shock_end=shock_end,
            shock_func=getattr(model_copy, shock_func) if isinstance(shock_func, str) else shock_func,
            shock_params=shock_params
        )
    else:
        # Run without shock if no shock function provided
        metrics_history = track_network_metrics_over_time(model_copy, steps=steps)
    
    # Create a list of nodes that were directly influenced by the shock
    if shock_func and shock_func == "apply_targeted_shock_high_degree" and shock_params:
        # For high-degree targeting, get top percent nodes by degree
        top_percent = shock_params.get("top_percent", 0.05)
        k = int(model.num_nodes * top_percent)
        degree_threshold = np.sort(model.degrees)[-k]
        influenced_nodes = np.where(model.degrees >= degree_threshold)[0]
    elif shock_func and shock_func == "apply_targeted_shock_random" and shock_params:
        # For random targeting, randomly select nodes
        target_percent = shock_params.get("target_percent", 0.25)
        k = int(model.num_nodes * target_percent)
        influenced_nodes = np.random.choice(model.num_nodes, size=k, replace=False)
    else:
        # Default: empty list
        influenced_nodes = []
    
    # Analyze influence pathways if there are influenced nodes
    if len(influenced_nodes) > 0:
        influence_results = analyze_influence_pathways(model, influenced_nodes)
    else:
        influence_results = None
    
    # Create dashboard figure
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2])
    
    # Top Left: Opinion Evolution
    ax1 = fig.add_subplot(gs[0, 0])
    history_props = model_copy.get_history_proportions()
    supporters = [h[SUPPORTER] for h in history_props]
    undecided = [h[UNDECIDED] for h in history_props]
    opposition = [h[OPPOSITION] for h in history_props]
    steps_range = range(len(history_props))
    
    ax1.plot(steps_range, supporters, '-', color=SUPPORTER_COLOR, linewidth=2.5, label='Supporters')
    ax1.plot(steps_range, undecided, '-', color=UNDECIDED_COLOR, linewidth=2.5, label='Undecided')
    ax1.plot(steps_range, opposition, '-', color=OPPOSITION_COLOR, linewidth=2.5, label='Opposition')
    
    if shock_func:
        ax1.axvspan(shock_start, shock_end, alpha=0.15, color='gray')
        ax1.axvline(x=shock_start, color='gray', linestyle='--', alpha=0.7)
        ax1.axvline(x=shock_end, color='gray', linestyle='--', alpha=0.7)
    
    ax1.set_ylabel('Opinion Proportion')
    ax1.set_title('Opinion Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top Middle: Homophily and Segregation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(metrics_history['homophily'], '-', color='#ff7f0e', linewidth=2.5, label='Homophily')
    ax2.plot(metrics_history['segregation_index'], '-', color='#9467bd', linewidth=2.5, label='Segregation')
    
    if shock_func:
        ax2.axvspan(shock_start, shock_end, alpha=0.15, color='gray')
    
    ax2.set_ylabel('Index Value')
    ax2.set_title('Opinion Segregation Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Top Right: Polarization
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(metrics_history['polarization'], '-', color='#d62728', linewidth=2.5)
    
    if shock_func:
        ax3.axvspan(shock_start, shock_end, alpha=0.15, color='gray')
    
    ax3.set_ylabel('Polarization Index')
    ax3.set_title('Network Polarization')
    ax3.grid(True, alpha=0.3)
    
    # Middle Left: Opinion Clusters
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(metrics_history['supporter_clusters'], '-', color=SUPPORTER_COLOR, linewidth=2.5, label='Supporter')
    ax4.plot(metrics_history['undecided_clusters'], '-', color=UNDECIDED_COLOR, linewidth=2.5, label='Undecided')
    ax4.plot(metrics_history['opposition_clusters'], '-', color=OPPOSITION_COLOR, linewidth=2.5, label='Opposition')
    
    if shock_func:
        ax4.axvspan(shock_start, shock_end, alpha=0.15, color='gray')
    
    ax4.set_ylabel('Number of Clusters')
    ax4.set_title('Opinion Fragmentation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Middle Center: Network Visualization
    ax5 = fig.add_subplot(gs[1, 1])
    model.visualize_network(ax=ax5, title="Network Structure")
    
    # Middle Right: Influence Pathways (if available)
    ax6 = fig.add_subplot(gs[1, 2])
    if influence_results:
        plot_influence_pathways(model, influence_results, ax=ax6, title="Influence Pathways")
    else:
        ax6.text(0.5, 0.5, "No influence pathway analysis available", 
                ha='center', va='center', fontsize=14)
        ax6.set_axis_off()
    
    # Bottom: Key Analysis and Insights
    ax_bottom = fig.add_subplot(gs[2, :])
    ax_bottom.axis('off')
    
    # Calculate key metrics for insights
    final_metrics = {key: metrics_history[key][-1] for key in metrics_history}
    initial_props = {
        'supporter': history_props[0][SUPPORTER], 
        'undecided': history_props[0][UNDECIDED], 
        'opposition': history_props[0][OPPOSITION]
    }
    final_props = {
        'supporter': history_props[-1][SUPPORTER], 
        'undecided': history_props[-1][UNDECIDED], 
        'opposition': history_props[-1][OPPOSITION]
    }
    
    # Prepare insights text
    network_type_info = {
        'scale-free': "Scale-free networks (urban centers) feature hub nodes with high degrees and many peripheral nodes with few connections.",
        'small-world': "Small-world networks (suburban areas) feature high clustering with occasional 'shortcut' connections that reduce the average path length.",
        'random': "Random networks (rural communities) have a more uniform degree distribution and less clustering.",
        'Unknown': "This network model represents a social structure with opinion dynamics."
    }
    
    insights = (
        f"Network Analysis: {network_type}\n\n"
        f"{network_type_info.get(network_type, network_type_info['Unknown'])}\n\n"
        f"Initial State: {initial_props['supporter']:.2f} Supporters, {initial_props['undecided']:.2f} Undecided, {initial_props['opposition']:.2f} Opposition\n"
        f"Final State: {final_props['supporter']:.2f} Supporters, {final_props['undecided']:.2f} Undecided, {final_props['opposition']:.2f} Opposition\n\n"
        f"Opinion Dynamics Metrics:\n"
        f"• Homophily: {final_metrics['homophily']:.2f} (initial: {initial_metrics['homophily']:.2f}) - Measures how much same-opinion nodes connect to each other\n"
        f"• Polarization: {final_metrics['polarization']:.2f} (initial: {initial_metrics['polarization']:.2f}) - Measures supporter-opposition connectivity\n"
        f"• Segregation Index: {final_metrics['segregation_index']:.2f} (initial: {initial_metrics['segregation_index']:.2f}) - Measures opinion clustering beyond random chance\n\n"
    )
    
    if shock_func:
        shock_type = "high-degree targeting" if shock_func == "apply_targeted_shock_high_degree" else "random targeting" if shock_func == "apply_targeted_shock_random" else "intervention"
        insights += (
            f"Intervention Analysis ({shock_type}):\n"
            f"• Net Supporter Gain: {final_props['supporter'] - initial_props['supporter']:.3f}\n"
            f"• Net Opposition Gain: {final_props['opposition'] - initial_props['opposition']:.3f}\n"
            f"• Net Advantage to {'Supporters' if final_props['supporter'] > final_props['opposition'] else 'Opposition'}: "
            f"{abs(final_props['supporter'] - final_props['opposition']):.3f}\n"
        )
        
        if influence_results:
            insights += (
                f"• Influence Reach: {influence_results['pathway_stats']['percent_influenced']:.1f}% of network\n"
                f"• First-order connections: {influence_results['pathway_stats']['first_order_count']}\n"
                f"• Second-order connections: {influence_results['pathway_stats']['second_order_count']}\n"
                f"• Higher-order connections: {influence_results['pathway_stats']['higher_order_count']}\n"
            )
    
    insights += (
        "\nNetwork Structure Insights:\n"
        f"• Number of opinion clusters - Supporters: {final_metrics['supporter_clusters']}, "
        f"Undecided: {final_metrics['undecided_clusters']}, Opposition: {final_metrics['opposition_clusters']}\n"
        f"• {'Opinion segregation increased' if final_metrics['segregation_index'] > initial_metrics['segregation_index'] else 'Opinion mixing increased'} "
        f"during the simulation\n"
        f"• Top influence from {'Supporters' if final_metrics['top_supporter_influence'] > final_metrics['top_opposition_influence'] else 'Opposition'}"
    )
    
    # Display the insights
    ax_bottom.text(0.5, 1.0, "Network Analysis Dashboard", ha='center', va='top', fontsize=18, fontweight='bold')
    ax_bottom.text(0.5, 0.9, insights, ha='center', va='top', fontsize=12, 
                  bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=1'))
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.4)
    return fig

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
            'all_histories': []  # New: store all histories for confidence intervals
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

def plot_intervention_pattern_results(results, save_path="figures/intervention_patterns.pdf"):
    """
    Plot the results from the intervention pattern experiment with improved visuals.
    Creates separate subplots for each intervention pattern compared with "No Intervention".
    Uses seaborn for publication-quality styling and includes confidence intervals.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_intervention_pattern_experiment()
    save_path : str
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plots
    """
    # Set seaborn style for publication quality
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    
    patterns = list(results.keys())
    if "No Intervention" not in patterns:
        print("Warning: 'No Intervention' pattern not found in results")
        return None
    
    # Separate "No Intervention" from other patterns
    intervention_patterns = [p for p in patterns if p != "No Intervention"]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(10, 3.5 * len(intervention_patterns)))
    
    # Define pattern-specific colors using seaborn color palette
    pattern_colors = sns.color_palette("husl", len(patterns))
    pattern_color_dict = {pattern: pattern_colors[i] for i, pattern in enumerate(patterns)}
    
    # Create one subplot for each intervention pattern compared with "No Intervention"
    for i, pattern in enumerate(intervention_patterns):
        ax = fig.add_subplot(len(intervention_patterns), 1, i+1)
        
        # Plot both "No Intervention" and the current intervention pattern
        comparison_patterns = ["No Intervention", pattern]
        
        for j, comp_pattern in enumerate(comparison_patterns):
            # Calculate means and confidence intervals if we have multiple histories
            if 'all_histories' in results[comp_pattern] and len(results[comp_pattern]['all_histories']) > 0:
                # For calculating confidence intervals
                num_steps = len(results[comp_pattern]['all_histories'][0])
                supporters_data = np.zeros((len(results[comp_pattern]['all_histories']), num_steps))
                undecided_data = np.zeros_like(supporters_data)
                opposition_data = np.zeros_like(supporters_data)
                
                # Collect data from all trials
                for k, hist in enumerate(results[comp_pattern]['all_histories']):
                    if len(hist) == num_steps:  # Ensure same length
                        supporters_data[k] = [h[SUPPORTER] for h in hist]
                        undecided_data[k] = [h[UNDECIDED] for h in hist]
                        opposition_data[k] = [h[OPPOSITION] for h in hist]
                
                # Calculate means and standard deviations
                supporters_mean = np.mean(supporters_data, axis=0)
                supporters_std = np.std(supporters_data, axis=0)
                undecided_mean = np.mean(undecided_data, axis=0)
                undecided_std = np.std(undecided_data, axis=0)
                opposition_mean = np.mean(opposition_data, axis=0)
                opposition_std = np.std(opposition_data, axis=0)
                
                steps = range(num_steps)
                
                # Line style based on pattern
                linestyle = '--' if comp_pattern == "No Intervention" else '-'
                alpha_level = 0.15  # Alpha level for confidence intervals
                
                # Plot means with confidence intervals (mean ± 1 std dev)
                ax.plot(steps, supporters_mean, 
                        linestyle=linestyle, color=SUPPORTER_COLOR, 
                        linewidth=2, label=f'{comp_pattern} - Supporters')
                ax.fill_between(steps, 
                               supporters_mean - supporters_std, 
                               supporters_mean + supporters_std, 
                               color=SUPPORTER_COLOR, alpha=alpha_level)
                
                ax.plot(steps, undecided_mean, 
                        linestyle=linestyle, color=UNDECIDED_COLOR, 
                        linewidth=2, label=f'{comp_pattern} - Undecided')
                ax.fill_between(steps, 
                               undecided_mean - undecided_std, 
                               undecided_mean + undecided_std, 
                               color=UNDECIDED_COLOR, alpha=alpha_level)
                
                ax.plot(steps, opposition_mean, 
                        linestyle=linestyle, color=OPPOSITION_COLOR, 
                        linewidth=2, label=f'{comp_pattern} - Opposition')
                ax.fill_between(steps, 
                               opposition_mean - opposition_std, 
                               opposition_mean + opposition_std, 
                               color=OPPOSITION_COLOR, alpha=alpha_level)
                
            # Fallback to using history if all_histories is not available
            elif 'history' in results[comp_pattern]:
                history = results[comp_pattern]['history']
                supporters = np.array([h[SUPPORTER] for h in history])
                undecided = np.array([h[UNDECIDED] for h in history])
                opposition = np.array([h[OPPOSITION] for h in history])
                
                steps = range(len(history))
                
                # Line style based on pattern
                linestyle = '--' if comp_pattern == "No Intervention" else '-'
                
                # Plot lines without confidence intervals
                ax.plot(steps, supporters, 
                        linestyle=linestyle, color=SUPPORTER_COLOR, 
                        linewidth=2, label=f'{comp_pattern} - Supporters')
                ax.plot(steps, undecided, 
                        linestyle=linestyle, color=UNDECIDED_COLOR, 
                        linewidth=2, label=f'{comp_pattern} - Undecided')
                ax.plot(steps, opposition, 
                        linestyle=linestyle, color=OPPOSITION_COLOR, 
                        linewidth=2, label=f'{comp_pattern} - Opposition')
        
        # Add intervention indicators for the non-baseline pattern
        if 'shock_schedule' in results[pattern]:
            for intervention in results[pattern]['shock_schedule']:
                # Create distinct colored regions with labels
                color = pattern_color_dict[pattern]
                
                # Add shaded region
                ax.axvspan(intervention['start'], intervention['end'], 
                          color=color, alpha=0.15, edgecolor=color, 
                          linewidth=1, zorder=1)
                
                # Add intensity label without arrow to reduce clutter
                arrow_mid = (intervention['start'] + intervention['end']) / 2
                if 'lambda_s_factor' in intervention:
                    ax.text(arrow_mid, 0.98, f"×{intervention.get('lambda_s_factor', 1.0):.1f}", 
                           ha='center', va='bottom', color='black', fontsize=9,
                           transform=ax.get_xaxis_transform(), 
                           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
        
        # Add styling
        ax.set_title(f'No Intervention vs. {pattern}', fontsize=12)
        ax.set_xlabel('Time Step', fontsize=10)
        if i == len(intervention_patterns) // 2:  # Only add y-label to middle plot
            ax.set_ylabel('Opinion Proportion', fontsize=10)
        ax.set_ylim(0, 1)
        
        # Add legend with minimal text
        if i == 0:  # Only add legend to top plot
            # Create custom handles and labels
            handles_opinions = [
                plt.Line2D([0], [0], color=SUPPORTER_COLOR, linewidth=2, label='Supporters'),
                plt.Line2D([0], [0], color=UNDECIDED_COLOR, linewidth=2, label='Undecided'),
                plt.Line2D([0], [0], color=OPPOSITION_COLOR, linewidth=2, label='Opposition')
            ]
            leg_opinion = ax.legend(handles=handles_opinions, title='Opinion', 
                             fontsize=9, loc='upper right')
            ax.add_artist(leg_opinion)
            
            # Create pattern legend
            handles_patterns = [
                plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='No Intervention'),
                plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Intervention')
            ]
            ax.legend(handles=handles_patterns, title='Pattern', 
                     fontsize=9, loc='upper left')
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {save_path}")
    
    return fig

def plot_intervention_final_distributions(results, save_path="figures/intervention_final_distributions.pdf"):
    """
    Plot the final opinion distributions for each intervention pattern.
    Uses seaborn for publication-quality styling.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_intervention_pattern_experiment()
    save_path : str
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the bar plot
    """
    # Set seaborn style for publication quality
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    
    patterns = list(results.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Data for plotting final distributions
    pattern_means = {
        'supporter': [np.mean(results[pattern]['supporter_final']) for pattern in patterns],
        'undecided': [np.mean(results[pattern]['undecided_final']) for pattern in patterns],
        'opposition': [np.mean(results[pattern]['opposition_final']) for pattern in patterns]
    }
    pattern_stds = {
        'supporter': [np.std(results[pattern]['supporter_final']) for pattern in patterns],
        'undecided': [np.std(results[pattern]['undecided_final']) for pattern in patterns],
        'opposition': [np.std(results[pattern]['opposition_final']) for pattern in patterns]
    }
    
    # Plot grouped bars for opinions - using seaborn
    bar_positions = np.arange(len(patterns))
    width = 0.25
    
    # Create DataFrame for seaborn
    data = []
    for i, pattern in enumerate(patterns):
        data.append({'Pattern': pattern, 'Opinion': 'Supporters', 
                    'Mean': pattern_means['supporter'][i], 
                    'Std': pattern_stds['supporter'][i]})
        data.append({'Pattern': pattern, 'Opinion': 'Undecided', 
                    'Mean': pattern_means['undecided'][i], 
                    'Std': pattern_stds['undecided'][i]})
        data.append({'Pattern': pattern, 'Opinion': 'Opposition', 
                    'Mean': pattern_means['opposition'][i], 
                    'Std': pattern_stds['opposition'][i]})
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Plot with seaborn
    sns.barplot(x='Pattern', y='Mean', hue='Opinion', data=df, 
               palette=[SUPPORTER_COLOR, UNDECIDED_COLOR, OPPOSITION_COLOR],
               errorbar='sd', ax=ax)
    
    # Add styling
    ax.set_title('Final Opinion Distribution by Intervention Pattern', fontsize=14)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_ylim(0, 0.7)  # Set consistent y-axis limits
    
    # Rotate x-tick labels if needed
    plt.xticks(rotation=30, ha='right')
    
    # Move legend to better position
    ax.legend(title='Opinion', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {save_path}")
    
    return fig

# Function to save all figures from experiments
def save_experiment_figures(experiment_name="intervention_patterns", results=None):
    """
    Save all figures from an experiment with standardized names.
    
    Parameters:
    -----------
    experiment_name : str
        Name of the experiment
    results : dict
        Results from the experiment
    """
    if results is None:
        return
        
    # Create figures directory
    import os
    os.makedirs("figures", exist_ok=True)
    
    if experiment_name == "intervention_patterns":
        # Save time evolution plots
        plot_intervention_pattern_results(results, f"figures/{experiment_name}_evolution.pdf")
        
        # Save final distribution plots
        plot_intervention_final_distributions(results, f"figures/{experiment_name}_final.pdf")
    
    elif experiment_name == "blitz_vs_sustained":
        # Implement additional experiment figure saving as needed
        pass

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



def run_network_blitz_comparison(
    n_nodes=1000,
    total_steps=500,
    num_trials=10,
    lambda_s=0.13,
    lambda_o=0.14
):
    """
    Run experiment comparing blitz vs sustained intervention effectiveness 
    across different network structures.
    
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
    # Network types to test
    network_types = {
        'Scale-free (Urban)': {'create_func': create_scale_free_network, 'params': {'n': n_nodes, 'm': 3}},
        'Small-world (Suburban)': {'create_func': create_small_world_network, 'params': {'n': n_nodes, 'k': 6, 'p': 0.1}},
        'Random (Rural)': {'create_func': create_random_network, 'params': {'n': n_nodes, 'k': 6}}
    }
    
    # Intervention patterns 
    patterns = {
        'None': {
            'schedule': []
        },
        'Blitz': {
            'schedule': [{'start': 30, 'end': 40, 'lambda_s_factor': 8.0}]
        },
        'Sustained': {
            'schedule': [{'start': 30, 'end': 70, 'lambda_s_factor': 2.0}]
        }
    }
    
    # Store results
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
                'supporter_history': [],
                'shock_schedule': pattern_info['schedule'],
                'opinion_evolution': []  # Will store complete history for each trial
            }
            
            for trial in tqdm(range(num_trials)):
                # Create network
                network = network_info['create_func'](**network_info['params'])
                
                # Create initial states - more evenly distributed
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
                
                # Store history for all trials to calculate variance bands
                history = model.get_history_proportions()
                supporter_history = [h[SUPPORTER] for h in history]
                results[network_name][pattern_name]['supporter_history'].append(supporter_history)
                
                # Store full opinion evolution for the first trial only
                if trial == 0:
                    results[network_name][pattern_name]['opinion_evolution'] = history
    
    return results

def plot_network_blitz_comparison(results):
    """
    Plot the results from the network blitz comparison experiment.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_network_blitz_comparison()
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plots
    """
    network_types = list(results.keys())
    if not network_types:
        return None
    
    # Get patterns from first network type
    patterns = list(results[network_types[0]].keys())
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], height_ratios=[1, 1])
    
    # Color mappings for patterns (consistent across plots)
    pattern_colors = {
        'None': 'gray',
        'Blitz': 'crimson',
        'Sustained': 'royalblue'
    }
    
    # 1. Time series comparison across network types (top row)
    for i, network_type in enumerate(network_types):
        ax = fig.add_subplot(gs[0, i])
        
        for pattern in patterns:
            # Get supporter history data for all trials
            all_supporter_histories = results[network_type][pattern]['supporter_history']
            
            if all_supporter_histories:
                # Calculate mean and standard deviation across trials
                # First ensure all histories are the same length
                min_length = min(len(history) for history in all_supporter_histories)
                standardized_histories = [history[:min_length] for history in all_supporter_histories]
                
                # Convert to numpy array for calculations
                supporter_data = np.array(standardized_histories)
                mean_values = np.mean(supporter_data, axis=0)
                std_values = np.std(supporter_data, axis=0)
                
                # Time steps for x-axis
                time_steps = np.arange(min_length)
                
                # Plot mean line
                ax.plot(time_steps, mean_values, 
                        color=pattern_colors.get(pattern, f'C{patterns.index(pattern)}'),
                        linewidth=2, label=pattern)
                
                # Add confidence interval (shaded area)
                ax.fill_between(time_steps, 
                                mean_values - std_values, 
                                mean_values + std_values,
                                alpha=0.2, 
                                color=pattern_colors.get(pattern, f'C{patterns.index(pattern)}'))
                
                # Highlight intervention periods if any
                for intervention in results[network_type][pattern]['shock_schedule']:
                    ax.axvspan(intervention['start'], intervention['end'], 
                              alpha=0.1, 
                              color=pattern_colors.get(pattern, f'C{patterns.index(pattern)}'),
                              edgecolor='none')
        
        # Set title and labels
        ax.set_title(f'{network_type}', fontsize=14)
        if i == 0:  # Only add y-label to the first plot
            ax.set_ylabel('Supporter Proportion', fontsize=12)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add legend to the last plot
        if i == len(network_types) - 1:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title='Intervention', fontsize=10, loc='upper right')
    
    # 2. Final supporter percentage bar chart across networks (bottom row)
    for i, network_type in enumerate(network_types):
        ax = fig.add_subplot(gs[1, i])
        
        # Collect final proportion data
        pattern_means = {
            'supporter': [np.mean(results[network_type][p]['supporter_final']) for p in patterns],
            'undecided': [np.mean(results[network_type][p]['undecided_final']) for p in patterns],
            'opposition': [np.mean(results[network_type][p]['opposition_final']) for p in patterns]
        }
        
        pattern_stds = {
            'supporter': [np.std(results[network_type][p]['supporter_final']) for p in patterns],
            'undecided': [np.std(results[network_type][p]['undecided_final']) for p in patterns],
            'opposition': [np.std(results[network_type][p]['opposition_final']) for p in patterns]
        }
        
        # Set bar positions
        x = np.arange(len(patterns))
        width = 0.3
        
        # Plot grouped bars
        ax.bar(x - width/2, pattern_means['supporter'], width, 
              yerr=pattern_stds['supporter'], capsize=3,
              color=[pattern_colors.get(p, f'C{patterns.index(p)}') for p in patterns],
              edgecolor='black', linewidth=1, label='Supporters')
        
        # Add hatched bars for opposition
        ax.bar(x + width/2, pattern_means['opposition'], width, 
              yerr=pattern_stds['opposition'], capsize=3,
              color='white', edgecolor='black', linewidth=1, 
              hatch='///', label='Opposition')
        
        # Add styling
        if i == 0:  # Only add y-label to the first plot
            ax.set_ylabel('Final Proportion', fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(patterns)
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend to the last plot
        if i == len(network_types) - 1:
            ax.legend(fontsize=10)
            
        # Add net supporter advantage as text
        for j, pattern in enumerate(patterns):
            supporter = pattern_means['supporter'][j]
            opposition = pattern_means['opposition'][j]
            advantage = supporter - opposition
            
            # Position text above the higher bar
            y_pos = max(supporter, opposition) + 0.05
            
            if advantage > 0:
                text_color = 'green'
                sign = '+'
            else:
                text_color = 'red'
                sign = ''
                
            ax.text(j, y_pos, f"{sign}{advantage:.2f}", 
                   ha='center', va='bottom', 
                   color=text_color, fontweight='bold')
    
    # Add colorbar legend
    cax = fig.add_subplot(gs[:, 3])
    cax.axis('off')
    
    # Add a suptitle
    plt.suptitle('Impact of Blitz vs Sustained Interventions Across Network Types', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, wspace=0.3)
    return fig


# Update main execution script to include the new experiment
if __name__ == "__main__":
    # Set default parameters
    n_nodes = 1000  # Using smaller network for faster demonstration
    shock_duration = 25  # Longer shock duration for more impact
    total_steps = 500  # Longer simulation to see dynamics
    num_trials = 10  # Increase for more reliable results
    lambda_s = 0.13  # Lower base rates for more dynamics
    lambda_o = 0.14
    
    # Set professional plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Uncomment to run the new blitz vs sustained experiment
    # results1 = run_blitz_vs_sustained_experiment(
    #     n_nodes=n_nodes,
    #     total_steps=total_steps,
    #     num_trials=num_trials,
    #     lambda_s=lambda_s,
    #     lambda_o=lambda_o,
    #     vary_intensity=True
    # )
    # fig1 = plot_blitz_vs_sustained_results(results1, vary_intensity=True)
    # plt.savefig('blitz_vs_sustained_variable_intensity.png', dpi=300, bbox_inches='tight')
    # 
    # results2 = run_blitz_vs_sustained_experiment(
    #     n_nodes=n_nodes,
    #     total_steps=total_steps,
    #     num_trials=num_trials,
    #     lambda_s=lambda_s,
    #     lambda_o=lambda_o,
    #     vary_intensity=False
    # )
    # fig2 = plot_blitz_vs_sustained_results(results2, vary_intensity=False)
    # plt.savefig('blitz_vs_sustained_constant_intensity.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "="*80)
    print("OPINION DYNAMICS ANALYSIS: GRASSROOTS VS ESTABLISHMENT STRATEGIES")
    print("="*80)
    
    # Ask user which experiment to run
    print("\nAvailable experiments:")
    print("1. Grassroots vs. Establishment Strategy Comparison")
    print("2. Network Battleground Analysis")
    print("3. Campaign Timing Analysis")
    print("4. Intervention Pattern Analysis (Blitz vs. Sustained)")
    print("5. All experiments")
    
    try:
        choice = int(input("\nEnter experiment number (1-5): "))
    except:
        print("Invalid input, defaulting to experiment 4 (Intervention Pattern Analysis)")
        choice = 4
    
    run_experiment_1 = choice in [1, 5]
    run_experiment_2 = choice in [2, 5]
    run_experiment_3 = choice in [3, 5]
    run_experiment_4 = choice in [4, 5]
    
    # Run selected experiments
    figures = []
    
    # Experiment 1: Grassroots vs. Establishment Strategy
    if run_experiment_1:
        print("\n* EXPERIMENT 1: Grassroots vs. Establishment Strategy *")
        print("Running simulations to compare targeting approaches...")
        
        results = run_grassroots_vs_establishment_experiment(
            n_nodes=n_nodes,
            shock_duration=shock_duration,
            total_steps=total_steps,
            num_trials=num_trials,
            lambda_s=lambda_s,
            lambda_o=lambda_o
        )
        
        print("Creating enhanced visualizations...")
        fig1, fig2 = plot_experiment_results(results)
        figures.extend([fig1, fig2])
    
    # Experiment 2: Network Battleground Selection
    if run_experiment_2:
        print("\n* EXPERIMENT 2: Network Battleground Selection *")
        print("Running simulations to compare network types...")
        
        battleground_results = run_network_battleground_experiment(
            n_nodes=n_nodes,
            shock_duration=shock_duration,
            total_steps=total_steps,
            num_trials=num_trials,
            lambda_s=lambda_s,
            lambda_o=lambda_o
        )
        
        print("Creating enhanced battleground visualizations...")
        battleground_viz = plot_battleground_results(battleground_results)
        figures.append(battleground_viz)
    
    # Experiment 3: Campaign Timing Strategy
    if run_experiment_3:
        print("\n* EXPERIMENT 3: Campaign Timing Strategy *")
        print("Running simulations to compare timing approaches...")
        
        timing_results = run_timing_experiment(
            n_nodes=n_nodes,
            shock_duration=shock_duration,
            total_steps=total_steps,
            num_trials=num_trials,
            lambda_s=lambda_s,
            lambda_o=lambda_o
        )
        
        print("Creating enhanced timing visualizations...")
        timing_viz = plot_timing_results(timing_results)
        figures.append(timing_viz)
    
    # Experiment 4: Intervention Pattern Analysis
    if run_experiment_4:
        print("\n* EXPERIMENT 4: Intervention Pattern Analysis *")
        print("Running simulations to compare intervention patterns (blitz vs. sustained)...")
        
        pattern_results = run_intervention_pattern_experiment(
            n_nodes=n_nodes,
            total_steps=total_steps,
            num_trials=num_trials,
            lambda_s=lambda_s,
            lambda_o=lambda_o
        )
        
        print("Creating intervention pattern visualizations...")
        pattern_viz = plot_intervention_pattern_results(pattern_results)
        pattern_viz_final_distributions = plot_intervention_final_distributions(pattern_results)
        figures.extend([pattern_viz, pattern_viz_final_distributions])
    
    # Show all figures
    print("\nDisplaying all visualizations. Close windows to continue...")
    
    for fig in figures:
        plt.figure(fig.number)
        plt.savefig(f'{fig.number}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\nExperiment complete!")

# Function to run all experiments and save all figures
def run_and_save_all_figures(n_nodes=1000, num_trials=5):
    """
    Run all experiments and save all figures with proper confidence intervals and error bars.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes for experiments
    num_trials : int
        Number of trials to run for statistical rigor
    """
    print("Running all experiments and generating publication-quality figures...")
    
    # Create figures directory
    os.makedirs("figures", exist_ok=True)
    
    # Run and save timing experiment
    print("\n1. Running timing experiment...")
    timing_results = run_timing_experiment(n_nodes=n_nodes, num_trials=num_trials)
    plot_timing_results(timing_results, "figures/timing_experiment.pdf")
    
    # Run and save intervention pattern experiment
    print("\n2. Running intervention pattern experiment...")
    pattern_results = run_intervention_pattern_experiment(n_nodes=n_nodes, num_trials=num_trials)
    plot_intervention_pattern_results(pattern_results, "figures/intervention_patterns.pdf")
    plot_intervention_final_distributions(pattern_results, "figures/intervention_final_distributions.pdf")
    
    # Run and save cross-network comparison
    print("\n3. Running cross-network intervention comparison...")
    cross_network_results = run_cross_network_intervention_comparison(n_nodes=n_nodes, num_trials=num_trials)
    plot_cross_network_results(cross_network_results, "figures/cross_network_comparison.pdf")
    
    print("\nAll experiments completed and figures saved to the 'figures' directory.")
    
    return {
        "timing": timing_results,
        "patterns": pattern_results,
        "cross_network": cross_network_results
    }

def run_all_experiments_and_save(
    output_dir="experiment_results",
    experiment_configs=None,
    overwrite=False
):
    """
    Run all experiments with multiple parameter configurations and save results to disk.
    This allows for loading results in notebooks for custom visualizations.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save experiment results
    experiment_configs : dict or None
        Dictionary of experiment configurations. If None, uses default settings.
        Format: {experiment_name: [list of parameter dicts]}
    overwrite : bool
        Whether to overwrite existing results
        
    Returns:
    --------
    dict
        Dictionary with paths to all saved result files
    """
    import os
    import json
    import pickle
    import time
    import datetime
    from tqdm import tqdm
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default configurations if none provided
    if experiment_configs is None:
        experiment_configs = {
            "grassroots_vs_establishment": [
                {
                    "n_nodes": 1000,
                    "shock_duration": 20,
                    "total_steps": 100,
                    "num_trials": 5,
                    "lambda_s": 0.12,
                    "lambda_o": 0.12
                },
                {
                    "n_nodes": 1000,
                    "shock_duration": 30,
                    "total_steps": 150,
                    "num_trials": 10,
                    "lambda_s": 0.08,
                    "lambda_o": 0.08
                }
            ],
            "network_battleground": [
                {
                    "n_nodes": 1000,
                    "shock_duration": 20,
                    "total_steps": 100,
                    "num_trials": 5,
                    "lambda_s": 0.12,
                    "lambda_o": 0.12
                }
            ],
            "timing": [
                {
                    "n_nodes": 1000,
                    "shock_duration": 20,
                    "total_steps": 100,
                    "num_trials": 5,
                    "lambda_s": 0.12,
                    "lambda_o": 0.12
                },
                {
                    "n_nodes": 1000,
                    "shock_duration": 30,
                    "total_steps": 150,
                    "num_trials": 10,
                    "lambda_s": 0.08,
                    "lambda_o": 0.08
                }
            ],
            "intervention_pattern": [
                {
                    "n_nodes": 1000,
                    "total_steps": 150,
                    "num_trials": 5,
                    "lambda_s": 0.12,
                    "lambda_o": 0.12
                },
                {
                    "n_nodes": 1000,
                    "total_steps": 200,
                    "num_trials": 10,
                    "lambda_s": 0.08,
                    "lambda_o": 0.08
                }
            ],
            "blitz_vs_sustained": [
                {
                    "n_nodes": 1000,
                    "total_steps": 150,
                    "num_trials": 5,
                    "lambda_s": 0.12,
                    "lambda_o": 0.12,
                    "vary_intensity": True
                },
                {
                    "n_nodes": 1000,
                    "total_steps": 150,
                    "num_trials": 5,
                    "lambda_s": 0.12,
                    "lambda_o": 0.12,
                    "vary_intensity": False
                }
            ],
            "cross_network_intervention": [
                {
                    "n_nodes": 1000,
                    "shock_duration": 20,
                    "total_steps": 150,
                    "num_trials": 5,
                    "lambda_s": 0.12,
                    "lambda_o": 0.12
                }
            ]
        }
    
    # Dictionary to store paths to all saved results
    result_paths = {}
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save the experiment configurations
    config_path = os.path.join(run_dir, "experiment_configs.json")
    with open(config_path, 'w') as f:
        json.dump(experiment_configs, f, indent=2)
    
    # Function to save experiment results
    def save_experiment_results(exp_name, config_idx, config, results):
        # Create directory for this experiment
        exp_dir = os.path.join(run_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create a readable name for the config
        config_str_parts = []
        for key, value in config.items():
            if key != "num_trials":  # Exclude num_trials from filename
                config_str_parts.append(f"{key}_{value}")
        config_str = "__".join(config_str_parts)
        
        # Filename: experiment_configidx_params.pkl
        filename = f"{exp_name}_{config_idx}_{config_str}.pkl"
        filepath = os.path.join(exp_dir, filename)
        
        # Check if file exists
        if os.path.exists(filepath) and not overwrite:
            print(f"Skipping {filepath} (already exists)")
            return filepath
        
        # Save results with pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'experiment': exp_name,
                'config': config,
                'results': results,
                'timestamp': timestamp
            }, f)
        
        return filepath
    
    # Run each experiment with each configuration
    for exp_name, configs in experiment_configs.items():
        print(f"\nRunning experiment: {exp_name}")
        result_paths[exp_name] = []
        
        for i, config in enumerate(configs):
            print(f"  Configuration {i+1}/{len(configs)}: {config}")
            
            # Run the appropriate experiment based on name
            if exp_name == "grassroots_vs_establishment":
                results = run_grassroots_vs_establishment_experiment(**config)
            elif exp_name == "network_battleground":
                results = run_network_battleground_experiment(**config)
            elif exp_name == "timing":
                results = run_timing_experiment(**config)
            elif exp_name == "intervention_pattern":
                results = run_intervention_pattern_experiment(**config)
            elif exp_name == "blitz_vs_sustained":
                results = run_blitz_vs_sustained_experiment(**config)
            elif exp_name == "cross_network_intervention":
                results = run_cross_network_intervention_comparison(**config)
            else:
                print(f"Unknown experiment: {exp_name}")
                continue
            
            # Save results
            filepath = save_experiment_results(exp_name, i, config, results)
            result_paths[exp_name].append(filepath)
            
            print(f"  Results saved to: {filepath}")
    
    # Create a summary file with all paths
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'result_paths': result_paths
        }, f, indent=2)
    
    print(f"\nAll experiments completed. Results saved to: {run_dir}")
    print(f"Summary file: {summary_path}")
    
    return result_paths

def load_experiment_results(filepath):
    """
    Load experiment results from a saved file.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved results file (.pkl)
        
    Returns:
    --------
    dict
        Dictionary with experiment results
    """
    import pickle
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data

def find_experiment_results(output_dir="experiment_results", experiment=None, config_pattern=None):
    """
    Find and list experiment result files matching the given criteria.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing experiment results
    experiment : str or None
        Name of the experiment to filter by (or None for all)
    config_pattern : str or None
        Pattern to match in the configuration string (or None for all)
        
    Returns:
    --------
    list
        List of paths to matching result files
    """
    import os
    import glob
    
    # Get all run directories
    run_dirs = sorted(glob.glob(os.path.join(output_dir, "run_*")))
    
    # For each run, get all matching experiment files
    matching_files = []
    
    for run_dir in run_dirs:
        if experiment:
            # Look in specific experiment directory
            exp_dir = os.path.join(run_dir, experiment)
            if os.path.isdir(exp_dir):
                patterns = [os.path.join(exp_dir, f"{experiment}_*.pkl")]
            else:
                continue
        else:
            # Look in all experiment directories
            patterns = [os.path.join(run_dir, "*", "*.pkl")]
        
        # Get all files matching the patterns
        for pattern in patterns:
            files = glob.glob(pattern)
            
            # Filter by config pattern if provided
            if config_pattern:
                files = [f for f in files if config_pattern in os.path.basename(f)]
            
            matching_files.extend(files)
    
    return matching_files