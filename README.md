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
- Required packages: numpy, pandas, matplotlib, networkx, pymc, arviz

### Installation

```bash
git clone https://github.com/yourusername/grid-congestion-analysis.git
cd grid-congestion-analysis
pip install -r requirements.txt
```

### Usage

Run Python scripts:

```bash
python simple_congestion_simulation.py
python monte_carlo_congestion_simulation.py
python bayesian_congestion_simulation.py
```

Or use Jupyter notebooks:
- simple_congestion_simulation.ipynb
- monte_carlo_congestion_simulation.ipynb
- bayesian_congestion_simulation.ipynb -- to be added

## Files

- `simple_congestion_simulation.py`: Basic grid congestion model
- `monte_carlo_congestion_simulation.py`: Probabilistic simulation across scenarios
- `bayesian_congestion_simulation.py`: Statistical inference model
- `data/generated/monte_carlo_results.csv`: Simulation output data

## License

[MIT License](LICENSE)

