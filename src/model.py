import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# Define the states
SUPPORTER = 0
UNDECIDED = 1
OPPOSITION = 2

# Professional color scheme
SUPPORTER_COLOR = '#1f77b4'  # Blue
UNDECIDED_COLOR = '#2ca02c'  # Green
OPPOSITION_COLOR = '#d62728'  # Red
COLOR_PALETTE = [SUPPORTER_COLOR, UNDECIDED_COLOR, OPPOSITION_COLOR]

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
            # Default: 20% supporters, 60% undecided, 20% opposition
            self.states = np.random.choice(
                [SUPPORTER, UNDECIDED, OPPOSITION], 
                size=self.num_nodes, 
                p=[0.2, 0.6, 0.2]
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
        
        # reduce stubbornness of targeted high-degree nodes to make them more persuasive
        self.stubbornness[high_degree_nodes] *= 0.3  # More dramatic reduction
        
        # increase opinion strength of these influencers
        self.opinion_strength[high_degree_nodes] = np.minimum(self.opinion_strength[high_degree_nodes] * 2.0, 1.0)
        
        # Reset time in state for shocked nodes to make them more immediately influential
        self.time_in_state[high_degree_nodes] = 0
        
        # Convert some high-degree undecided nodes directly to supporters to create cascades
        undecided_high_degree = high_degree_nodes[self.states[high_degree_nodes] == UNDECIDED]
        if len(undecided_high_degree) > 0:
            convert_count = max(1, int(len(undecided_high_degree) * 0.5))
            convert_nodes = np.random.choice(undecided_high_degree, convert_count, replace=False)
            self.states[convert_nodes] = SUPPORTER
        
        # also reduce stubbornness of nodes connected to high-degree nodes
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
        
        # slightly reduce stubbornness of targeted random nodes
        self.stubbornness[targeted_nodes] *= 0.5  # More significant reduction
        
        # increase opinion strength of these grassroots campaigners
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
            
            # make transitions more likely when opinions are balanced
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