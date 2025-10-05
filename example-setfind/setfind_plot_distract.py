import matplotlib.pyplot as plt
import pandas as pd

# Define the set of heads and layers configurations
# layers_heads_npos_set = [
#     (1, 1, 10),
#     (1, 2, 10),
#     (2, 2, 10),
#     (4, 4, 10),
#     (6, 6, 10),
#     (8, 8, 10)
# ]
# layers_heads_npos_set = [
#     (1, 2, 8),
#     (1, 2, 12),
#     (1, 2, 16),
#     (1, 2, 24),
#     (1, 2, 32),
#     (1, 2, 48),
# ]

layers_heads_npos_distract_set = [
	(1, 2, 16, 0),
	(1, 2, 16, 8),
	(1, 2, 16, 16),
	(1, 2, 16, 24),
	(1, 2, 16, 32),
	(1, 2, 16, 48),
	(1, 2, 16, 64),
	(1, 2, 16, 96)
]

colors = [
    '#d62728',  # Brick Red
    '#ff7f0e',  # Safety Orange
    '#bcbd22',  # Tumeric Yellow-Green
    '#2ca02c',  # Cooked Asparagus Green
    '#17becf',  # Blue-Muted Cyan
    '#1f77b4',  # Muted Blue
    '#9467bd',  # Muted Purple
    '#e377c2',  # Raspberry Pink
    '#8c564b',  # Chestnut Brown
    '#7f7f7f'   # Middle Gray
] # gemini came up with this

# Create a figure and axes for subplots
plt.rcParams['font.size'] = 20
plt.rcParams['figure.dpi'] = 120
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
# axes = axes.flatten()

# Function to create a scatter plot for a given heads and layers configuration
def create_plot(ax, layers, heads, npos, distract, color):
	# Read the data from the file and store it in a DataFrame
	data = pd.read_csv(f"vallog_x4_psgd_l{layers}_h{heads}_npos{npos}_width{32+distract}.txt", sep="\t", header=None, names=["data_size", "validation", "train_loss"])

	# remove outliers in the median calc (as you would anyway)
	filtered_data = data[data["validation"] < 1000]
	# Group by data size and calculate median
	grouped_data = data.groupby("data_size")["validation"].median().reset_index()

	# Create the scatter plot with a logarithmic y-axis
	ax.scatter(data["data_size"], data["validation"], c=color, alpha=0.5, edgecolors='w', s=100)

	ax.scatter(data["data_size"], data["train_loss"], c=color, alpha=0.5, edgecolors='w', s=40)

	# Plot median for each data size
	ax.scatter(grouped_data["data_size"], grouped_data["validation"], c=color, s=120, edgecolors='w')
	
	ax.plot(grouped_data["data_size"], grouped_data["validation"], color=color, alpha=0.7, linewidth=3, label=f'Set size {npos}')

	ax.set_title(f'Validaton vs set size vs sample size')
	ax.set_xlabel('Training data size')
	ax.set_ylabel('Validation')
	ax.set_ylim(10**-3, 5e3)
	ax.set_yscale('log')
	ax.set_xscale('log', base=2)
	ax.grid(True)
	ax.legend()

# Create a plot for each heads and layers configuration
for i, (layers, heads, npos, distract) in enumerate(layers_heads_npos_distract_set):
    create_plot(axes, layers, heads, npos, distract, colors[i])

# # Hide any unused subplots
# for j in range(len(layers_heads_npos_set), len(axes)):
#     fig.delaxes(axes[j])

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
