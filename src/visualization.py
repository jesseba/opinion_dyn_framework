import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
from matplotlib.patches import Patch
from model import SUPPORTER, UNDECIDED, OPPOSITION, SUPPORTER_COLOR, UNDECIDED_COLOR, OPPOSITION_COLOR, COLOR_PALETTE

# Set professional style for plots
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

def plot_grassroots_vs_establishment(results, save_path=None):
    """
    Plot the results from the grassroots vs. establishment experiment.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_grassroots_vs_establishment_experiment()
    save_path : str or None
        Path to save the figure, or None to not save
        
    Returns:
    --------
    tuple
        (fig1, fig2) containing the figures
    """
    network_types = list(results.keys())
    strategies = list(results[network_types[0]].keys())
    
    # Setting up a more professional color scheme
    colors = [SUPPORTER_COLOR, UNDECIDED_COLOR, OPPOSITION_COLOR]
    opinion_labels = ['Supporters', 'Undecided', 'Opposition']
    
    # Create figure for barplots of final opinion distributions
    fig1, axes = plt.subplots(1, len(network_types), figsize=(16, 6))
    
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
                # Use correct keys for each opinion
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
    fig1.subplots_adjust(top=0.85, bottom=0.15)
    
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
    
    # Save figures if a path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        base_path = os.path.splitext(save_path)[0]
        fig1.savefig(f"{base_path}_distribution.pdf", bbox_inches='tight', dpi=300)
        fig2.savefig(f"{base_path}_evolution.pdf", bbox_inches='tight', dpi=300)
        
        print(f"Figures saved to {base_path}_distribution.pdf and {base_path}_evolution.pdf")
    
    return fig1, fig2

def plot_network_battleground(results, save_path=None):
    """
    Plot the results from the network battleground experiment.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_network_battleground_experiment()
    save_path : str or None
        Path to save the figure, or None to not save
        
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
    
    # Save figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    return fig

def plot_timing_results(results, save_path=None):
    """
    Plot the results from the timing experiment with publication-quality visualizations
    including confidence intervals and proper error bars.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_timing_experiment()
    save_path : str or None
        Path to save the figure, or None to not save
        
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
    
    # Save figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    return fig

def plot_intervention_pattern_results(results, save_path=None):
    """
    Plot the results from the intervention pattern experiment with improved visuals.
    Creates separate subplots for each intervention pattern compared with "No Intervention".
    Uses seaborn for publication-quality styling and includes confidence intervals.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_intervention_pattern_experiment()
    save_path : str or None
        Path to save the figure, or None to not save
        
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
    
    # Save figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    return fig

def plot_intervention_final_distributions(results, save_path=None):
    """
    Plot the final opinion distributions for each intervention pattern.
    Uses seaborn for publication-quality styling.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_intervention_pattern_experiment()
    save_path : str or None
        Path to save the figure, or None to not save
        
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
    
    # Save figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    return fig

def plot_blitz_vs_sustained_results(results, vary_intensity=True, save_path=None):
    """
    Plot the results from the blitz vs sustained experiment.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_blitz_vs_sustained_experiment()
    vary_intensity : bool
        Whether the experiment varied intensity (affects title)
    save_path : str or None
        Path to save the figure, or None to not save
        
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
    
    # Get patterns from first network type
    pattern_names = list(results[network_types[0]].keys())
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, len(network_types), figure=fig)
    
    # Color palette for patterns
    pattern_colors = sns.color_palette("husl", len(pattern_names))
    
    # Row 1: Evolution of supporter proportion over time
    for i, network_type in enumerate(network_types):
        ax = fig.add_subplot(gs[0, i])
        
        for j, pattern in enumerate(pattern_names):
            if 'history' in results[network_type][pattern]:
                history = results[network_type][pattern]['history']
                supporters = [h[SUPPORTER] for h in history]
                steps = range(len(history))
                
                # Plot line
                ax.plot(steps, supporters, '-', color=pattern_colors[j], linewidth=2.0, label=pattern)
                
                # Add intervention period indicators
                for intervention in results[network_type][pattern]['shock_schedule']:
                    ax.axvspan(intervention['start'], intervention['end'], 
                              alpha=0.1, color=pattern_colors[j], edgecolor='none')
        
        # Add styling
        if i == 0:
            ax.set_ylabel('Supporter Proportion', fontsize=12)
        ax.set_title(f'{network_type}', fontsize=14)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to the first plot
        if i == 0:
            ax.legend(title='Intervention Pattern', fontsize=10, loc='upper right')
    
    # Row 2: Barplots of final proportions
    for i, network_type in enumerate(network_types):
        ax = fig.add_subplot(gs[1, i])
        
        # Data for plotting
        pattern_labels = []
        supporter_means = []
        supporter_stds = []
        
        for pattern in pattern_names:
            if pattern == 'No Intervention':
                continue  # Skip for cleaner comparison
            pattern_labels.append(pattern)
            supporter_means.append(np.mean(results[network_type][pattern]['supporter_final']))
            supporter_stds.append(np.std(results[network_type][pattern]['supporter_final']))
        
        # Plot bars with error
        bars = ax.bar(range(len(pattern_labels)), supporter_means, yerr=supporter_stds, 
                      capsize=5, color=pattern_colors[1:], edgecolor='black', linewidth=1)
        
        # Add data labels on top of bars
        for j, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + supporter_stds[j] + 0.01,
                    f'{supporter_means[j]:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Add styling
        if i == 0:
            ax.set_ylabel('Final Supporter Proportion', fontsize=12)
        ax.set_xticks(range(len(pattern_labels)))
        ax.set_xticklabels(pattern_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
    
    # Add comprehensive title
    if vary_intensity:
        title = 'Impact of Intervention Strategy: Blitz vs. Sustained (Equalized Total Influence Power)'
    else:
        title = 'Impact of Intervention Strategy: Blitz vs. Sustained (Equal Intensity)'
    plt.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.3)
    
    # Save figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    return fig 

def plot_critical_mass_results(results, output_dir=None, file_prefix='critical_mass'):
    """
    Plot results from the critical mass experiment.
    
    Parameters:
    -----------
    results : dict
        Results from run_critical_mass_experiment
    output_dir : str, optional
        Directory to save figures
    file_prefix : str, optional
        Prefix for saved figure filenames
    """
    network_types = list(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.8, 3))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # First plot: Line plot showing supporter proportion by initial level
    ax = axes[0]
    for i, network_type in enumerate(network_types):
        initial_proportions = results[network_type]['initial_proportions']
        final_supporters = results[network_type]['final_supporters']
        final_std = results[network_type]['final_supporters_std']
        
        ax.plot(initial_proportions, final_supporters, 'o-', label=network_type, 
                color=colors[i], linewidth=2, markersize=8)
        ax.fill_between(initial_proportions, 
                        np.array(final_supporters) - np.array(final_std),
                        np.array(final_supporters) + np.array(final_std),
                        alpha=0.2, color=colors[i])
    
    # Add reference line showing y=x
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No change')
    
    ax.set_xlabel('Initial supporter proportion', fontsize=14)
    ax.set_ylabel('Final supporter proportion', fontsize=14)
    ax.set_title('Critical Mass Effects on Final Opinion State', fontsize=16)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Second plot: Change in supporter proportion
    ax = axes[1]
    for i, network_type in enumerate(network_types):
        initial_proportions = results[network_type]['initial_proportions']
        final_supporters = results[network_type]['final_supporters']
        final_std = results[network_type]['final_supporters_std']
        
        # Calculate change in proportion
        change = np.array(final_supporters) - np.array(initial_proportions)
        
        ax.plot(initial_proportions, change, 'o-', label=network_type, 
                color=colors[i], linewidth=2, markersize=8)
        ax.fill_between(initial_proportions, 
                        change - np.array(final_std),
                        change + np.array(final_std),
                        alpha=0.2, color=colors[i])
    
    # Add reference line at y=0
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Initial supporter proportion', fontsize=14)
    ax.set_ylabel('Change in supporter proportion', fontsize=14)
    ax.set_title('Net Change in Supporter Proportion', fontsize=16)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    
    # Third plot: Final distributions stacked
    ax = axes[2]
    bar_width = 0.25
    x = np.arange(len(initial_proportions))
    
    # Choose one network type for demonstration (scale-free)
    network_type = 'scale-free'
    
    # Get data
    supporters = results[network_type]['final_supporters']
    undecided = results[network_type]['final_undecided']
    opposition = results[network_type]['final_opposition']
    initial_props = results[network_type]['initial_proportions']
    
    # Create bars
    bottom_vals = np.zeros(len(supporters))
    
    # Plot stacked bars
    ax.bar(x, supporters, bar_width, label='Supporters', color='green', alpha=0.7)
    
    bottom_vals = np.array(supporters)
    ax.bar(x, undecided, bar_width, bottom=bottom_vals, label='Undecided', color='gray', alpha=0.7)
    
    bottom_vals = bottom_vals + np.array(undecided)
    ax.bar(x, opposition, bar_width, bottom=bottom_vals, label='Opposition', color='red', alpha=0.7)
    
    # Set labels and title
    ax.set_ylabel('Final Proportion', fontsize=14)
    ax.set_xlabel('Initial Supporter Proportion', fontsize=14)
    ax.set_title(f'Final Opinion Distribution ({network_type} network)', fontsize=16)
    ax.legend(fontsize=12)
    
    # Set x-tick labels to show initial proportions
    ax.set_xticks(x)
    # Format tick labels to show percentages
    tick_labels = [f'{prop:.0%}' for prop in initial_props]
    # Only show some tick labels to avoid crowding
    for i in range(len(tick_labels)):
        if i % 3 != 0:  # Show every 3rd label
            tick_labels[i] = ''
    ax.set_xticklabels(tick_labels)
    
    # Layout and save
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{file_prefix}_overview.png'), dpi=300, bbox_inches='tight')
    
    # Optional additional plots - trajectories for key initial proportions
    
    # Choose one network type and a few key initial proportions
    network_type = 'scale-free'
    key_proportions = [0.15, 0.3, 0.5, 0.7]
    
    # Find the closest values in the actual initial_supporter_range
    initial_proportions = results[network_type]['initial_proportions']
    key_indices = [np.abs(initial_proportions - prop).argmin() for prop in key_proportions]
    key_actual_props = [initial_proportions[idx] for idx in key_indices]
    
    # Create figure for trajectories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (prop_idx, prop) in enumerate(zip(key_indices, key_actual_props)):
        ax = axes[i]
        
        # Get the history from the first trial for this proportion
        history = results[network_type]['trials'][prop]['history']
        
        if history:  # Check if history exists
            timesteps = range(len(history[1]))  # Supporters history
            
            # Plot trajectories
            ax.plot(timesteps, history[1], '-', color='green', linewidth=2, label='Supporters')
            ax.plot(timesteps, history[0], '-', color='gray', linewidth=2, label='Undecided')
            ax.plot(timesteps, history[2], '-', color='red', linewidth=2, label='Opposition')
            
            # Add horizontal line at initial proportion
            ax.axhline(y=prop, color='green', linestyle='--', alpha=0.5, 
                      label=f'Initial supporters: {prop:.2f}')
            
            ax.set_xlabel('Time step', fontsize=12)
            ax.set_ylabel('Proportion', fontsize=12)
            ax.set_title(f'Opinion Dynamics (Initial S: {prop:.2f})', fontsize=14)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=10, loc='best')
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'{file_prefix}_trajectories.png'), dpi=300, bbox_inches='tight')
    
    return fig, axes

def plot_intervention_sensitivity(results, output_dir=None, file_prefix='intervention_sensitivity'):
    """
    Plot results from the intervention sensitivity experiment.
    
    Parameters:
    -----------
    results : dict
        Results from run_intervention_sensitivity_experiment
    output_dir : str, optional
        Directory to save figures
    file_prefix : str, optional
        Prefix for saved figure filenames
    """
    # Extract data
    initial_proportions = results['initial_proportions']
    intervention_type = results['intervention_type']
    
    baseline = results['baseline_final_supporters']
    baseline_std = results['baseline_final_supporters_std']
    
    intervention = results['intervention_final_supporters']
    intervention_std = results['intervention_final_supporters_std']
    
    gain = results['supporter_gain']
    gain_std = results['supporter_gain_std']
    
    rel_gain = results['relative_gain']
    rel_gain_std = results['relative_gain_std']
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Baseline vs Intervention final state
    ax = axes[0, 0]
    
    ax.plot(initial_proportions, baseline, 'o-', color='blue', label='No intervention', 
            linewidth=2, markersize=8)
    ax.fill_between(initial_proportions, 
                    np.array(baseline) - np.array(baseline_std),
                    np.array(baseline) + np.array(baseline_std),
                    alpha=0.2, color='blue')
    
    ax.plot(initial_proportions, intervention, 'o-', color='green', label=f'{intervention_type.title()} intervention', 
            linewidth=2, markersize=8)
    ax.fill_between(initial_proportions, 
                    np.array(intervention) - np.array(intervention_std),
                    np.array(intervention) + np.array(intervention_std),
                    alpha=0.2, color='green')
    
    # Add reference line showing y=x
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No change')
    
    ax.set_xlabel('Initial supporter proportion', fontsize=14)
    ax.set_ylabel('Final supporter proportion', fontsize=14)
    ax.set_title('Intervention Effect by Initial Support Level', fontsize=16)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Plot 2: Absolute gain
    ax = axes[0, 1]
    
    ax.plot(initial_proportions, gain, 'o-', color='purple', linewidth=2, markersize=8)
    ax.fill_between(initial_proportions, 
                    np.array(gain) - np.array(gain_std),
                    np.array(gain) + np.array(gain_std),
                    alpha=0.2, color='purple')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Initial supporter proportion', fontsize=14)
    ax.set_ylabel('Absolute gain in supporter proportion', fontsize=14)
    ax.set_title('Intervention Effectiveness (Absolute Gain)', fontsize=16)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    
    # Plot 3: Relative gain (normalized by initial proportion)
    ax = axes[1, 0]
    
    ax.plot(initial_proportions, rel_gain, 'o-', color='orange', linewidth=2, markersize=8)
    ax.fill_between(initial_proportions, 
                    np.array(rel_gain) - np.array(rel_gain_std),
                    np.array(rel_gain) + np.array(rel_gain_std),
                    alpha=0.2, color='orange')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Initial supporter proportion', fontsize=14)
    ax.set_ylabel('Relative gain factor', fontsize=14)
    ax.set_title('Intervention Effectiveness (Normalized by Initial Support)', fontsize=16)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    
    # Plot 4: Trajectory examples at key initial proportions
    ax = axes[1, 1]
    
    # Choose a few key proportions
    key_proportions = [0.1, 0.3, 0.5, 0.7]
    
    # Find indices of closest values in our data
    key_indices = [np.abs(initial_proportions - prop).argmin() for prop in key_proportions]
    
    # Different line styles for each proportion
    linestyles = ['-', '--', '-.', ':']
    
    for i, idx in enumerate(key_indices):
        prop = initial_proportions[idx]
        
        # Get history data for baseline and intervention
        baseline_history = results['trials'][prop]['baseline'][0]['history']
        intervention_history = results['trials'][prop]['intervention'][0]['history']
        
        if baseline_history and intervention_history:
            timesteps = range(len(baseline_history[1]))
            
            # Plot supporter trajectories
            ax.plot(timesteps, baseline_history[1], linestyle=linestyles[i], color='blue', 
                   alpha=0.7, linewidth=2)
            ax.plot(timesteps, intervention_history[1], linestyle=linestyles[i], color='green', 
                   alpha=0.7, linewidth=2, label=f'Initial: {prop:.2f}')
    
    # Add legend for initial proportions
    ax.legend(fontsize=12, title='Initial support', loc='upper left')
    
    # Add legend for intervention vs baseline
    baseline_patch = mpatches.Patch(color='blue', label='No intervention')
    interv_patch = mpatches.Patch(color='green', label=f'{intervention_type.title()} intervention')
    ax.legend(handles=[baseline_patch, interv_patch], loc='lower right', fontsize=12)
    
    ax.set_xlabel('Time step', fontsize=14)
    ax.set_ylabel('Supporter proportion', fontsize=14)
    ax.set_title('Opinion Trajectories by Initial Support Level', fontsize=16)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Layout and save
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{file_prefix}_{intervention_type}.png'), dpi=300, bbox_inches='tight')
    
    # Additional plot: Example comparison at most interesting proportion
    # Find proportion with highest absolute gain
    max_gain_idx = np.argmax(gain)
    max_gain_prop = initial_proportions[max_gain_idx]
    
    # Create figure for detailed comparison
    fig2, axes2 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Get histories
    baseline_history = results['trials'][max_gain_prop]['baseline'][0]['history']
    intervention_history = results['trials'][max_gain_prop]['intervention'][0]['history']
    
    if baseline_history and intervention_history:
        timesteps = range(len(baseline_history[1]))
        
        # Plot all opinion states
        axes2.plot(timesteps, baseline_history[1], '-', color='blue', linewidth=2, label='Supporters (No intervention)')
        axes2.plot(timesteps, baseline_history[0], '-', color='lightblue', linewidth=2, label='Undecided (No intervention)')
        axes2.plot(timesteps, baseline_history[2], '-', color='darkblue', linewidth=2, label='Opposition (No intervention)')
        
        axes2.plot(timesteps, intervention_history[1], '-', color='green', linewidth=2, label='Supporters (Intervention)')
        axes2.plot(timesteps, intervention_history[0], '-', color='lightgreen', linewidth=2, label='Undecided (Intervention)')
        axes2.plot(timesteps, intervention_history[2], '-', color='darkgreen', linewidth=2, label='Opposition (Intervention)')
        
        # Highlight intervention period
        axes2.axvspan(10, 10 + 20, alpha=0.2, color='yellow', label='Intervention period')
        
        axes2.set_xlabel('Time step', fontsize=14)
        axes2.set_ylabel('Proportion', fontsize=14)
        axes2.set_title(f'Detailed Opinion Dynamics (Initial S: {max_gain_prop:.2f})', fontsize=16)
        axes2.grid(alpha=0.3)
        axes2.legend(fontsize=10, loc='best')
        axes2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'{file_prefix}_{intervention_type}_detail.png'), 
                    dpi=300, bbox_inches='tight')
    
    return fig, axes 