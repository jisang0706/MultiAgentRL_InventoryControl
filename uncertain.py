from tkinter import font
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from scipy.stats import poisson

plt.rcParams['text.usetex'] = True

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

# Data from your table
data = {
    "Lambda Demand": [3, 4, 5, 6, 7],
    "Profit": [-1260, 2168, 4606, 7661, 8144],
    "Std Profit": [134, 170, 160, 179, 190],
    "Avg Backlog": [3, 6, 15, 17, 38],
    "Std Backlog": [3, 5, 14, 11, 29],
    "Inventory on Hold": [390, 340, 300, 265, 217],
    "Std Inventory": [80, 90, 100, 105, 114]
}

df = pd.DataFrame(data)

# Define lambda values and sample size
lambda_values = [3, 4, 5, 6, 7]
num_samples = 1000  # Number of demand samples per lambda
max_demand = 20  # X-axis limit

fig, ax = plt.subplots(figsize=(12, 8), layout = 'constrained')

# Plot empirical histograms & theoretical PMF for each λ
for lam, color in zip(lambda_values, colors):
    sampled_demand = np.random.poisson(lam, num_samples)
    
    # Histogram (empirical demand distribution)
    sns.histplot(sampled_demand, bins=range(0, max_demand), kde=False, 
                 stat="probability", label=f"Empirical $\lambda$ ={lam}", alpha=0.5, color=color)
    
    # Overlay theoretical PMF
    x = np.arange(0, max_demand)
    pmf = poisson.pmf(x, lam)
    ax.plot(x, pmf, marker='o', linestyle='-', label=f"Theoretical PMF $\lambda$={lam}", color=color)

plt.xlabel("Demand ($k$)", fontsize = 18)
plt.ylabel("Probability", fontsize = 18)
plt.legend(frameon=False, fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig.savefig('figures/uncertain_demand.png', dpi=1100)  # Save as PNG
fig.savefig('figures/uncertain_demand.pdf', bbox_inches='tight')  # Save as PDF with tight bounding box
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 8))

# First Y-axis (Avg Backlog)
ax1.set_xlabel("Poisson Demand Rate ($\lambda$)", fontsize=18)
ax1.tick_params(axis='x', labelsize=16)
ax1.set_ylabel("Average Backlog", color=colors[0], fontsize=18)
ax1.errorbar(df["Lambda Demand"], df["Avg Backlog"], yerr=df["Std Backlog"], fmt='o-', color=colors[0], capsize=4, label="Avg Backlog")
ax1.tick_params(axis='y', labelcolor=colors[0], labelsize=14)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.set_xticks(df["Lambda Demand"])  # Ensures only 3, 4, 5, 6, 7 appear
# Second Y-axis (Inventory on Hold)
ax2 = ax1.twinx()
ax2.set_ylabel("Average On-hold Inventory", color=colors[1], fontsize=18)
ax2.errorbar(df["Lambda Demand"], df["Inventory on Hold"], yerr=df["Std Inventory"], fmt='s--', color=colors[1], capsize=4, label="Inventory on Hold")
ax2.tick_params(axis='y', labelcolor=colors[1], labelsize=14)

# Add title and legend
fig.tight_layout()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig.savefig('figures/uncertain_inv_b.png', dpi=1100)  # Save as PNG
fig.savefig('figures/uncertain_inv_b.pdf', bbox_inches='tight')  # Save as PDF with tight bounding box
plt.show()

fig, ax = plt.subplots(figsize=(12, 8), layout = 'constrained')
x = df["Lambda Demand"]
y = df["Profit"]

# Fill between lower and upper bounds of std profit
ax.fill_between(x, y - df["Std Profit"], y + df["Std Profit"], alpha=0.2, color=colors[0], label = "Std Profit")
ax.plot(x, y, marker='o', color=colors[0], linewidth=2, label="Mean Cumulative Profit")
ax.set_xticks(df["Lambda Demand"])  # Ensures only 3, 4, 5, 6, 7 appear
ax.legend(frameon=False, fontsize=18, loc='lower right')
ax.set_xlabel("Poisson Demand Rate ($\lambda$)", fontsize = 18)
ax.set_ylabel("Profit", fontsize = 18)
fig.tight_layout()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig.savefig('figures/uncertain_profit.png', dpi=1100)  # Save as PNG
fig.savefig('figures/uncertain_profit.pdf', bbox_inches='tight')  # Save as PDF with tight bounding box
plt.show()

