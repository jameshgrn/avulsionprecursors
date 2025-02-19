import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings

# Update global plot parameters
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 0.5,
    'axes.labelsize': 10,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5
})

scatter_color = '#FF9F1C'  # Bright orange for all data points

def calculate_normalized_distances(row):
    """
    Simply return the original values without sinuosity normalization
    """
    return row['L_A'], row['variogram_range'], row['L_A_error'], row['variogram_range_error']

def create_plots():
    # Load only the data needed for regression
    full_data = pd.read_csv('data/manuscript_data/supplementary_table_1.csv')
    
    # Prepare dataframe for regression
    df = full_data[['river_name', 'L_A', 'variogram_range', 'river_type', 'avg_width', 'variogram_range_error']].dropna()
    
    df['L_A'] = df['L_A'] * 1000 / df['avg_width']
    df['variogram_range'] = df['variogram_range'] * 1000 / df['avg_width']
    df['L_A_error'] = df['L_A'] * 0.1  # Keep 10% error for L_A
    df['variogram_range_error'] = df['variogram_range_error'] * 1000 / df['avg_width']
    
    # Apply sinuosity normalization
    df['L_A'], df['variogram_range'], df['L_A_error'], df['variogram_range_error'] = zip(*df.apply(calculate_normalized_distances, axis=1))

    # Create regression plot - note x and y are swapped
    fig, ax = plt.subplots(figsize=(5, 3.5))
    scatter = sns.scatterplot(data=df, x='variogram_range', y='L_A', hue='river_type', ax=ax, alpha=0.7, s=60, edgecolor='black')
    
    # Add error bars - note x and y errors are swapped
    ax.errorbar(df['variogram_range'], df['L_A'], 
                xerr=df['variogram_range_error'], yerr=df['L_A_error'], 
                fmt='none', ecolor='gray', alpha=0.75, capsize=3, lw=0.5)

    # Perform regression - note X and y are swapped
    X, y = sm.add_constant(df['variogram_range']), df['L_A']
    model = sm.OLS(y, X).fit()
    
    # Prediction line
    x_pred = np.linspace(df['variogram_range'].min(), df['variogram_range'].max(), 100)
    y_pred = model.predict(sm.add_constant(x_pred))

    y_pred_ci = model.get_prediction(sm.add_constant(x_pred)).conf_int()
    y_pred_ci = np.clip(y_pred_ci, 0, None)

    ax.plot(x_pred, y_pred, color='red', zorder=1)
    ax.fill_between(x_pred, y_pred_ci[:, 0], y_pred_ci[:, 1], color='red', alpha=0.1)

    turkwel_lilongwe_data = pd.DataFrame({
        'river_name': ['Turkwel', 'Lilongwe'],
        'L_A': [4197, 2115],  # Original values in meters
        'variogram_range': [4062, 4307],  # Original values in meters
        'river_type': ['Computed Delta'] * 2,
        'avg_width': [77.93, 99.8],  # Add average widths in meters
        'L_A_error': [419.7, 211.5],  # 10% error
        'variogram_range_error': [406.2, 430.7]  # 10% error
    })

    # Normalize by width
    turkwel_lilongwe_data['L_A'] = turkwel_lilongwe_data['L_A'] / turkwel_lilongwe_data['avg_width']
    turkwel_lilongwe_data['variogram_range'] = turkwel_lilongwe_data['variogram_range'] / turkwel_lilongwe_data['avg_width']
    turkwel_lilongwe_data['L_A_error'] = turkwel_lilongwe_data['L_A_error'] / turkwel_lilongwe_data['avg_width']
    turkwel_lilongwe_data['variogram_range_error'] = turkwel_lilongwe_data['variogram_range_error'] / turkwel_lilongwe_data['avg_width']

    # Update Turkwel and Lilongwe data plotting - note x and y are swapped
    ax.errorbar(turkwel_lilongwe_data['variogram_range'], turkwel_lilongwe_data['L_A'],
                 xerr=turkwel_lilongwe_data['variogram_range_error'], yerr=turkwel_lilongwe_data['L_A_error'],
                 fmt='s', color='#FF9F1C', markersize=6, capsize=3, 
                 markeredgecolor='black', markeredgewidth=0.5,
                 label='backwater length (not included in OLS)')

    ax.plot([0,400], [0, 400], color='black', linestyle='dashed')

    # Update axis labels
    ax.set(xlabel=r'$\Lambda_{c}/B$', ylabel=r'$L_A/B$')
    ax.legend(fontsize=7, loc='lower right')

    n = len(df)
    ax.text(0.05, 0.95, f'$R^2 = {model.rsquared:.2f}$\n$y = {model.params["const"]:.2f} + {model.params["variogram_range"]:.2f}x$\n$n = {n}$', 
             transform=ax.transAxes, verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.show()

    # Save regression plot

if __name__ == "__main__":
    create_plots()
