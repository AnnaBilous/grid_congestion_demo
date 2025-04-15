# Grid Congestion Analysis

This repository contains a suite of tools for analyzing electricity grid congestion risk under different technology adoption scenarios. The analysis focuses on how electric vehicles (EVs) and solar photovoltaic (PV) systems affect transformer loading in residential low-voltage grids.

## Overview

The project includes three levels of analysis:

1. **Simple Congestion Simulation**: Basic simulation of household electricity demand and transformer loading
2. **Monte Carlo Congestion Simulation**: Probabilistic assessment of congestion risk across different EV and PV adoption scenarios
3. **Bayesian Congestion Analysis**: Statistical modeling to quantify uncertainty and infer the relative impact of different technologies

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: numpy, pandas, matplotlib, networkx, pymc, arviz, tqdm

### Installation

```bash
git clone https://github.com/yourusername/grid-congestion-analysis.git
cd grid-congestion-analysis
pip install -r requirements.txt
```

### Usage

Run Python scripts:

```bash
python python_scripts/simple_congestion_simulation.py
python python_scripts/monte_carlo_congestion_simulation.py
python python_scripts/bayesian_congestion_simulation.py
```

Or use Jupyter notebooks:

```bash
jupyter notebook ipython_notebooks/
```

## Project Structure

```
├── python_scripts/              # Main simulation scripts
│   ├── simple_congestion_simulation.py
│   ├── monte_carlo_congestion_simulation.py
│   └── bayesian_congestion_simulation.py
├── ipython_notebooks/           # Jupyter notebooks
│   ├── simple_congestion_simulation.ipynb
│   ├── monte_carlo_congestion_simulation.ipynb
│   └── bayesian_congestion_simulation.ipynb
├── src/                         # Core functionality
│   ├── grid_model.py            # Grid topology generation
│   ├── simulate.py              # Load profile simulation
│   └── visualize.py             # Visualization utilities
├── data/
│   ├── generated/               # Simulation outputs
│   └── input/                   # Input data
└── README.md
```

## Analysis Methods

### Simple Simulation
Generates a synthetic grid and realistic household load profiles to visualize transformer loading.

### Monte Carlo Simulation
Runs multiple simulations with random variations to assess congestion probability across different EV and PV adoption scenarios.

### Bayesian Analysis
Uses probabilistic modeling to quantify uncertainty and infer the relative impact of different technologies on grid congestion.

## License

[MIT License](LICENSE)

