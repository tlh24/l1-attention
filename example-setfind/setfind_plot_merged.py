import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
import pdb
import argparse

# Define the set of heads and layers configurations
heads_layers_set = [
    (1, 1),
    (2, 1),
    (2, 2),
    (4, 4), 
    (6, 6),
    (8, 8)
]

# Create a figure and a single set of axes
fig, ax = plt.subplots(figsize=(12, 8))

parser = argparse.ArgumentParser()
parser.add_argument('-r', type=int, default=3, help='which run to plot')
cmd_args = parser.parse_args()
run = cmd_args.r

# Define a color map
colors = plt.get_cmap('tab10')(np.linspace(0,1,len(heads_layers_set)))

# Function to create a scatter plot for a given heads and layers configuration
def create_plot(ax, layers, heads, color):
    # Read the data from the file and store it in a DataFrame
    data = pd.read_csv(f"vallog{run}_l{layers}_h{heads}.txt", sep="\t", header=None, names=["data_size", "validation"])

    # Remove outliers in the median calc (as you would anyway)
    filtered_data = data[data["validation"] < 3]
    
    # Group by data size and calculate median
    grouped_data = filtered_data.groupby("data_size")["validation"].median().reset_index()

    # Create the scatter plot with a logarithmic y-axis
    ax.scatter(data["data_size"], data["validation"], color=color, alpha=0.45, edgecolors='w', s=100, label=f'{layers} layers, {heads} heads')

    # Plot median for each data size
    ax.plot(grouped_data["data_size"], grouped_data["validation"], color=color, alpha=0.7)

# Create a plot for each heads and layers configuration with different colors
for i, (heads, layers) in enumerate(heads_layers_set):
    create_plot(ax, layers, heads, colors[i])

ax.set_title(f'Run {run}: Validation vs Training Data Size for Different Model Configurations')
ax.set_xlabel('Training data size')
ax.set_ylabel('Validation')
ax.set_ylim(10**-5, 10**1) # FIXME clips data! 
ax.set_yscale('log')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.savefig(f'generalization_{run}_model-size_vs_data_size.pdf')
plt.show()
