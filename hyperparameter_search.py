columns = ['tau', 'llm_score', 'inverted_rbo']
records = [
    [1, 2.7267, .9855],
    [2, 2.9215, .9919],
    [3, 2.9482, .9922],
    [5, 2.9658, .9895],
    [10, 2.9763, .9800],
]

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# Extract data
tau_values = [record[0] for record in records]
llm_scores = [record[1] for record in records]
inverted_rbo_values = [record[2] for record in records]

# Create the scatter plot
plt.figure(figsize=(10, 6))

# Sort the points by tau value to ensure correct line connection
sorted_indices = np.argsort(tau_values)
sorted_tau = [tau_values[i] for i in sorted_indices]
sorted_llm = [llm_scores[i] for i in sorted_indices]
sorted_rbo = [inverted_rbo_values[i] for i in sorted_indices]

# Connect the dots with a dotted line with lower alpha
plt.plot(inverted_rbo_values, llm_scores, 'b:', alpha=0.3, zorder=1)

# Add larger scatter points
plt.scatter(inverted_rbo_values, llm_scores, s=200, c='blue', alpha=0.7, zorder=2)

# Add labels for each point (tau values) with larger font
for i, tau in enumerate(tau_values):
    plt.annotate(f'τ={tau}', 
                 (inverted_rbo_values[i], llm_scores[i]),
                 textcoords="offset points", 
                 xytext=(10,0), 
                 ha='left',
                 fontsize=16)  # Positioned to the right of dots

# Add axis labels with arrows indicating "better" direction
plt.xlabel('I-RBO →', fontsize=16, labelpad=15)
plt.ylabel('LLM Score →', fontsize=16, labelpad=15)

# Set axis limits to provide more space
plt.ylim(2.7, 3.05)  # Maintain margin but will adjust ticks below
plt.xlim(0.975, 0.995)  # Adjust as needed for x-axis

# Format x-axis to show .0000 without leading zero
def format_func(x, pos):
    return f'.{int(x*10000):04d}'

ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

# Set up y-axis ticks to stop at 3.0
yticks = [tick for tick in np.arange(2.7, 3.05, 0.05) if tick <= 3.0]
plt.yticks(yticks, fontsize=14)
plt.xticks(fontsize=14)

# Add a title
plt.title("")

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.3)

# Adjust the plot margins
plt.tight_layout()

# Save the plot
plt.savefig('tau_comparison_plot.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

