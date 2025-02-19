import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('data/manuscript_data/supplementary_table_1.csv')

# Create a single figure for both histograms
fig, ax = plt.subplots(figsize=(3, 2.5))

# First histogram
sns.histplot(x='dominant_wavelength', data=df, ax=ax, color='tab:orange', label=r'$L_{\lambda}$', alpha=0.6, binwidth=10)

# Second histogram
sns.histplot(x='variogram_range', data=df, ax=ax, color='tab:blue', label=r'$L_C$', alpha=0.6, binwidth=8)

# Set labels and ticks
ax.tick_params(labelsize=13)
ax.set_xlabel('Length scale (km)', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_xticks([0, 50, 110])
ymax = int(ax.get_yticks()[-1])
ax.set_yticks([0, ymax//2, ymax])

# Add legend
ax.legend(fontsize=11)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig('data/manuscript_data/plots/lengthscale_histograms.png', dpi=300)
plt.show()
