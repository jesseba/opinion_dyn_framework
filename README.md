# Opinion Dynamics Simulation Framework

This project provides a modular framework for simulating opinion dynamics on complex networks. It includes several experiments that simulate different intervention strategies for influencing opinions across various network topologies.

## Project Structure

The codebase is organized into several modules:

- `model.py` - Contains the `OpinionDynamicsModel` class that implements the simulation logic
- `networks.py` - Functions for generating different network types (scale-free, small-world, random)
- `experiments.py` - Implementation of various experiments on the model
- `visualization.py` - Functions for creating visualizations of the results
- `utils.py` - Utility functions for saving and loading experiment results
- `run_experiments.py` - Main script for running experiments from the command line

## Getting Started

### Prerequisites

This project requires Python 3.6+ and the following packages:
- numpy
- networkx
- matplotlib
- seaborn
- tqdm
- pandas

You can install the required packages using pip:

```bash
pip install numpy networkx matplotlib seaborn tqdm pandas
```

### Running an Experiment

To run an experiment, use the `run_experiments.py` script with the desired parameters:

```bash
python run_experiments.py --experiment intervention_pattern --n_nodes 1000 --total_steps 150 --num_trials 5
```

Available experiments:
- `grassroots_vs_establishment` - Compare establishment (high-degree) vs. grassroots (random) targeting
- `network_battleground` - Compare effectiveness across network types (urban, suburban, rural)
- `timing` - Compare early vs late intervention timing
- `intervention_pattern` - Compare various intervention patterns (blitz, sustained, pulsed)
- `blitz_vs_sustained` - Compare short intense vs. longer moderate interventions
- `all` - Run all experiments

Common parameters:
- `--n_nodes` - Number of nodes in the network (default: 1000)
- `--shock_duration` - Duration of the intervention in time steps (default: 20)
- `--total_steps` - Total duration of the simulation (default: 150)
- `--num_trials` - Number of simulation trials to run (default: 5)
- `--lambda_s` - Base rate toward supporter state (default: 0.12)
- `--lambda_o` - Base rate toward opposition state (default: 0.12)
- `--vary_intensity` - Whether to vary intensity for blitz vs sustained experiment
- `--output_dir` - Directory to save results (default: "results")
- `--no_save` - Do not save results
- `--no_plots` - Do not display plots

## Example Usage

### Running a Single Experiment

To run the intervention pattern experiment:

```bash
python run_experiments.py --experiment intervention_pattern --n_nodes 1000 --total_steps 150 --num_trials 10
```

### Running All Experiments

To run all experiments:

```bash
python run_experiments.py --experiment all --n_nodes 500 --total_steps 100 --num_trials 3
```

### Running with Custom Parameters

To run the timing experiment with custom parameters:

```bash
python run_experiments.py --experiment timing --n_nodes 2000 --shock_duration 30 --total_steps 200 --num_trials 5 --lambda_s 0.10 --lambda_o 0.15
```

## Model Description

The opinion dynamics model simulates the spread of opinions in a social network. Each node in the network represents an individual with one of three possible opinions:

- Supporter (0)
- Undecided (1)
- Opposition (2)

Opinions evolve over time based on:
1. Social influence from neighbors
2. Individual stubbornness
3. Opinion entrenchment over time
4. External interventions (shocks)

The model supports different intervention strategies:
- Broadcast (affecting all nodes equally)
- High-degree targeting (influential nodes)
- Random targeting (grassroots approach)
- Degree-proportional targeting

## Extending the Framework

To add a new experiment:

1. Define a new experiment function in `experiments.py`
2. Add visualization functions in `visualization.py`
3. Update the command-line parser in `run_experiments.py`

For more detailed information, refer to the docstrings in each module.
