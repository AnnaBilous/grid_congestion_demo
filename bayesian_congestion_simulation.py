# Add to imports
import pymc as pm
import arviz as az
import xarray as xr
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')  # Add parent directory to path

# Parameters
N_HOUSES = 50
HOURS = 24
TRANSFORMER_CAPACITY = 125  # kW
TRANSFORMER_ID = "TR_001"
DATA_DIR = "data/generated"
RESULTS_FILE = f"{DATA_DIR}/monte_carlo_results.csv"
N_SIMULATIONS = 10



def run_bayesian_congestion_model(results_df, n_samples=1000):
    """
    Build a Bayesian model to predict congestion probability based on EV and PV counts
    
    Args:
        results_df: DataFrame with simulation results
        n_samples: Number of samples for MCMC
    
    Returns:
        Inference data object with posterior samples and standardization parameters
    """
    # Extract data
    n_ev = results_df['n_ev'].values
    n_pv = results_df['n_pv'].values
    congestion = results_df['congestion_probability'].values
    
    # Standardize predictors
    ev_mean, ev_std = n_ev.mean(), n_ev.std()
    pv_mean, pv_std = n_pv.mean(), n_pv.std()
    n_ev_std = (n_ev - ev_mean) / ev_std
    n_pv_std = (n_pv - pv_mean) / pv_std
    
    # Build model
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta_ev = pm.Normal('beta_ev', mu=0, sigma=1)
        beta_pv = pm.Normal('beta_pv', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=0.1)
        
        # Linear predictor (logit scale)
        logit_p = alpha + beta_ev * n_ev_std + beta_pv * n_pv_std
        
        # Transform to probability scale
        p = pm.Deterministic('p', pm.math.invlogit(logit_p))
        
        # Likelihood
        likelihood = pm.Normal('likelihood', mu=p, sigma=sigma, observed=congestion)
        
        # Sample posterior
        idata = pm.sample(n_samples, tune=1000, return_inferencedata=True)
    
    # Return both the inference data and standardization parameters
    return idata, ev_mean, ev_std, pv_mean, pv_std

def plot_bayesian_contour(idata, results_df, ev_mean, ev_std, pv_mean, pv_std, output_dir=DATA_DIR):
    """Plot contour map of congestion probability from Bayesian model"""
    # Create prediction grid for smooth surface
    ev_range = np.linspace(0, results_df['n_ev'].max(), 100)
    pv_range = np.linspace(0, results_df['n_pv'].max(), 100)
    ev_grid, pv_grid = np.meshgrid(ev_range, pv_range)
    
    # Standardize grid values
    ev_grid_std = (ev_grid - ev_mean) / ev_std
    pv_grid_std = (pv_grid - pv_mean) / pv_std
    
    # Get posterior samples
    posterior = idata.posterior
    alpha = posterior['alpha'].values.flatten()
    beta_ev = posterior['beta_ev'].values.flatten()
    beta_pv = posterior['beta_pv'].values.flatten()
    
    # Compute predicted probabilities (mean of posterior)
    logit_p = (alpha.mean() + 
              beta_ev.mean() * ev_grid_std + 
              beta_pv.mean() * pv_grid_std)
    
    prob = 1 / (1 + np.exp(-logit_p))
    
    # Create detailed contour plot with data points
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(ev_grid, pv_grid, prob, cmap='YlOrRd', levels=20)
    plt.colorbar(label='Congestion Probability')
    
    # Add contour lines
    contour_lines = plt.contour(ev_grid, pv_grid, prob, colors='black', alpha=0.5, 
                               levels=[0.1, 0.3, 0.7, 0.9])
    plt.clabel(contour_lines, inline=True, fontsize=8)
    
    # Add the 50% probability contour line in blue
    contour_50 = plt.contour(ev_grid, pv_grid, prob, colors='blue', 
                            levels=[0.5], linewidths=2)
    plt.clabel(contour_50, inline=True, fontsize=10, fmt='%.2f')
    
    # Add a legend entry for the 50% threshold
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='blue', lw=2)]
    plt.legend(custom_lines, ['50% Risk Threshold'], loc='lower right')
    
    # Overlay actual data points
    sc = plt.scatter(results_df['n_ev'], results_df['n_pv'], 
                    c=results_df['congestion_probability'], 
                    cmap='YlOrRd', edgecolor='black', s=50)
    
    plt.title('Grid Congestion Risk by Technology Adoption')
    plt.xlabel('Number of EVs')
    plt.ylabel('Number of Solar PV')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bayesian_contour.png", dpi=300)
    plt.show()

def main():
    # Check if results file exists
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file {RESULTS_FILE} not found. Run monte_carlo_congestion_simulation.py first.")
        return
    
    # Load results
    results = pd.read_csv(RESULTS_FILE)
    print(f"Loaded results with {len(results)} scenarios")
    
    # Run Bayesian model
    print("\nRunning Bayesian model...")
    idata, ev_mean, ev_std, pv_mean, pv_std = run_bayesian_congestion_model(results)
    
    # Summarize posterior
    print("\nBayesian Model Summary:")
    print(az.summary(idata))
    
    # Plot posterior distributions for specific parameters
    az.plot_posterior(idata, var_names=['alpha', 'beta_ev', 'beta_pv', 'sigma'])
    plt.suptitle("Posterior Distributions")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/bayesian_posterior.png")
    plt.show()
    
    # Plot contour map
    plot_bayesian_contour(idata, results, ev_mean, ev_std, pv_mean, pv_std)
    
    # Calculate and print effect sizes
    posterior = idata.posterior
    beta_ev = posterior['beta_ev'].values.flatten()
    beta_pv = posterior['beta_pv'].values.flatten()
    
    print("\nEffect Sizes:")
    print(f"EV effect (mean): {beta_ev.mean():.4f}")
    print(f"EV effect (95% CI): [{np.percentile(beta_ev, 2.5):.4f}, {np.percentile(beta_ev, 97.5):.4f}]")
    print(f"PV effect (mean): {beta_pv.mean():.4f}")
    print(f"PV effect (95% CI): [{np.percentile(beta_pv, 2.5):.4f}, {np.percentile(beta_pv, 97.5):.4f}]")
    
    # Calculate probability that EV effect is greater than PV effect (in absolute terms)
    prob_ev_greater = np.mean(np.abs(beta_ev) > np.abs(beta_pv))
    print(f"Probability |EV effect| > |PV effect|: {prob_ev_greater:.4f}")

if __name__ == "__main__":
    main()