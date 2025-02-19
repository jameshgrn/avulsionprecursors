import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from adjustText import adjust_text
from typing import List, Dict, Optional
import json
from scipy.signal import savgol_filter

def load_data_dict(filename: str) -> Optional[Dict]:
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {filename} is not a valid JSON file.")
        return None

def load_river_data(river_name: str) -> gpd.GeoDataFrame:
    try:
        df = pd.read_csv(f'data/{river_name}_recalculated_edited.csv')
        
        # Load the river_spatial_stats_FINAL dataframe
        stats_df = pd.read_csv('data/manuscript_data/supplementary_table_1.csv')
        
        # Get the 'collected_during_or_after_tandem_start' value for this river
        collected_during_tandem = stats_df.loc[stats_df['river_name'] == river_name, 'collected_during_or_after_tandem_start'].iloc[0]
        
        river_data = df[df['river_name'] == river_name].dropna(subset=['dist_out', 'lambda'])
        river_data['dist_out'] = river_data['dist_out'].astype(float)
        river_data = river_data.sort_values('dist_out')
        river_data['lambda'] = river_data['lambda'].clip(lower=0.01)
        
        # Add the collected_during_tandem information to the river_data
        river_data['collected_during_tandem'] = collected_during_tandem
        
        gdf = gpd.GeoDataFrame(
            river_data,
            geometry=gpd.points_from_xy(river_data['dist_out'], np.zeros(len(river_data))),
            crs="EPSG:3857"
        )
        return gdf
    except Exception as e:
        print(f"Error loading river data: {e}")
        return None

def plot_lambda(data: pd.DataFrame, 
                avulsion_sites: List[Dict], crevasse_splay_sites: List[Dict], 
                river_name: str, 
                avulsion_belt: Optional[List[Dict]] = None) -> None:
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
    
    # Plot data points with reduced emphasis
    ax.scatter(data['dist_out'] / 1000, data['lambda'], 
              color='blue', s=15, zorder=2, alpha=0.3, 
              edgecolor='none')
    
    # Load filter lengths from supplementary table
    stats_df = pd.read_csv('data/manuscript_data/supplementary_table_1.csv')
    river_stats = stats_df.loc[stats_df['river_name'] == river_name].iloc[0]
    
    # Get variogram range and dominant wavelength in km
    small_filter_km = river_stats['variogram_range']
    large_filter_km = river_stats['dominant_wavelength']
    
    # Calculate window sizes based on data density
    points_per_km = len(data) / (data['dist_out'].max() - data['dist_out'].min()) * 1000
    window_small = int(small_filter_km * points_per_km)
    window_large = int(large_filter_km * points_per_km)
    
    # Ensure windows are odd numbers
    window_small = window_small + 1 if window_small % 2 == 0 else window_small
    window_large = window_large + 1 if window_large % 2 == 0 else window_large
    
    # Calculate Savgol filters and clip to prevent negative values
    savgol_small = np.clip(savgol_filter(data['lambda'], window_small, 3), 0.1, None)
    savgol_large = np.clip(savgol_filter(data['lambda'], window_large, 3), 0.1, None)
    
    # Plot filtered lines with increased emphasis
    # ax.plot(data['dist_out'] / 1000, savgol_large, 
    #        color='red', alpha=1.0, zorder=4, 
    #        linewidth=2.5, label=f'{large_filter_km:.0f} km filter')
    # ax.plot(data['dist_out'] / 1000, savgol_small, 
    #        color='blue', alpha=1.0, zorder=3, 
    #        linewidth=2.5, label=f'{small_filter_km:.0f} km filter')
    
    # Add lambda = 2 line
    ax.axhline(y=2, color='darkred', linestyle='--', label=r'$\Lambda$ = 2')
    
    # Styling
    ax.set_xlabel('Distance from outlet (km)', fontsize=13, fontname='Helvetica')
    ax.set_ylabel(r'Avulsion Potential $\Lambda$', fontsize=13, rotation=90)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.legend(loc='best', fontsize=10, frameon=False)
    ax.set_yscale('log')
    
    ax.invert_xaxis()

    # Load tandem date information
    stats_df = pd.read_csv('data/manuscript_data/supplementary_table_1.csv')
    tandem_date_start = pd.to_datetime(stats_df.loc[stats_df['river_name'] == river_name, 'tandem_date_start'].iloc[0])

    # Add vertical lines for avulsions and crevasse splays
    texts = []
    for site in avulsion_sites:
        dist_out = site['position']
        site_date = pd.to_datetime(site['date'])
        if site_date >= tandem_date_start:
            ax.axvline(dist_out, color='black', lw=2, linestyle='--', alpha=0.9)
            year = int(site['date'].split('-')[0])
            texts.append(ax.text(dist_out, ax.get_ylim()[1], str(year), 
                               rotation=90, va='bottom', ha='right', 
                               fontsize=10, fontweight='bold', color='black'))
        else:
            ax.axvline(dist_out, color='gray', lw=1, linestyle='--', alpha=0.5)
            year = int(site['date'].split('-')[0])
            texts.append(ax.text(dist_out, ax.get_ylim()[1], str(year), 
                               rotation=90, va='bottom', ha='right', 
                               fontsize=10, color='gray'))

    for site in crevasse_splay_sites:
        dist_out = site['position']
        site_date = pd.to_datetime(site['date'])
        if not any(abs(site['position'] - avulsion['position']) < 1e-6 for avulsion in avulsion_sites):
            if site_date >= tandem_date_start:
                ax.axvline(dist_out, color='black', lw=2, linestyle=':', alpha=0.9)
                year = int(site['date'].split('-')[0])
                texts.append(ax.text(dist_out, ax.get_ylim()[1], str(year), 
                                   rotation=90, va='bottom', ha='right', 
                                   fontsize=10, fontweight='bold', color='black'))
            else:
                ax.axvline(dist_out, color='gray', lw=1, linestyle=':', alpha=0.5)
                year = int(site['date'].split('-')[0])
                texts.append(ax.text(dist_out, ax.get_ylim()[1], str(year), 
                                   rotation=90, va='bottom', ha='right', 
                                   fontsize=10, color='gray'))
    
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    
    plt.tight_layout()
    # plt.savefig(f'/Users/jakegearon/staging/supplementary_figures/lambda_{river_name}.png', 
    #             dpi=300, bbox_inches='tight')
    # plt.savefig(f'/Users/jakegearon/projects/NSF_PostDoc/AGU_2024_lambda_{river_name}_w_savgol.png', 
    #             dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main(river_names: Optional[List[str]] = None) -> None:
    data_dict = load_data_dict('data/data_dict.json')
    
    if data_dict is None:
        print("Error: Unable to load data dictionary.")
        return
    
    if river_names is None:
        river_names = ["VENEZ_2023"]
    
    for river_name in river_names:
        print(f"\nProcessing {river_name}")

        if river_name not in data_dict:
            print(f"Warning: {river_name} not found in data_dict. Skipping.")
            continue

        try:
            data = load_river_data(river_name)
            
            if data is not None and len(data) > 0:
                plot_lambda(
                    data,
                    data_dict[river_name].get('avulsion_lines', []),
                    data_dict[river_name].get('crevasse_splay_lines', []),
                    river_name,
                    data_dict[river_name].get('avulsion_belt', None)
                )
        except Exception as e:
            print(f"Error processing {river_name}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 