import matplotlib.pyplot as plt
import pandas as pd

# Define the set of heads and layers configurations
heads_layers_set = [
    (1, 1),
    (2, 1),
    (2, 2),
    (4, 4), 
    (6, 6),
    (8, 8)
]

# Create a figure and axes for subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 15))
axes = axes.flatten()

# Function to create a scatter plot for a given heads and layers configuration
def create_plot(ax, layers, heads):
	# Read the data from the file and store it in a DataFrame
	data = pd.read_csv(f"vallog5_l{layers}_h{heads}.txt", sep="\t", header=None, names=["data_size", "validation"])

	# remove outliers in the median calc (as you would anyway)
	filtered_data = data[data["validation"] < 5]
	# Group by data size and calculate median
	grouped_data = data.groupby("data_size")["validation"].median().reset_index()

	# Create the scatter plot with a logarithmic y-axis
	ax.scatter(data["data_size"], data["validation"], c='blue', alpha=0.5, edgecolors='w', s=100, label='Individual runs')

	# Plot median for each data size
	ax.scatter(grouped_data["data_size"], grouped_data["validation"], c='black', s=100, label='Median', edgecolors='w')
	
	ax.plot(grouped_data["data_size"], grouped_data["validation"], color=color, alpha=0.7, label=f'{layers}l, {heads}h - median')

	ax.set_title(f'{layers} layers, {heads} heads, 15k iters')
	ax.set_xlabel('Training data size')
	ax.set_ylabel('Validation')
	ax.set_ylim(10**-5, 10**2)
	ax.set_yscale('log')
	ax.grid(True)
	ax.legend()

# Create a plot for each heads and layers configuration
for i, (heads, layers) in enumerate(heads_layers_set):
    create_plot(axes[i], layers, heads)

# Hide any unused subplots
for j in range(len(heads_layers_set), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


'''
1 = AdamW, dual loss. 
2 = PSGD, 5 min, dual loss.
3 = PSGD, dual loss, distractors. (-x flag)
4 = AdamW, dual loss, distractors. (-a, -x)
5 = PSGD, update liklihood 0.5, rank 20 approx, distractors. 
6 = PSGD, update liklihood 0.1, rank 10 approx, lr_params= 0.03, distractors.  
'''
