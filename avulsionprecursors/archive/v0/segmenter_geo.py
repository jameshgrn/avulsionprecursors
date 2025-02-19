#%%
import uuid
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from gee_sampler import perform_cross_section_sampling


def initialize_result_gdf(df):
    """
    Initialize an empty GeoDataFrame with the same columns as the input DataFrame.

    Parameters:
    df (GeoDataFrame): Input GeoDataFrame with columns to be copied.

    Returns:
    GeoDataFrame: An empty GeoDataFrame with the same columns and CRS as the input.
    """
    result = gpd.GeoDataFrame(columns=df.columns, geometry='geometry', crs='EPSG:4326')
    result.set_crs(epsg=4326, inplace=True)
    return result

def process_neighbours(current_reach_id, direction, df, result, visited):
    """
    Recursively process and add neighboring reaches to the result GeoDataFrame.

    Parameters:
    current_reach_id (int): The current reach ID to process.
    direction (str): The direction to process ('up' or 'down').
    df (GeoDataFrame): The input GeoDataFrame containing reach data.
    result (GeoDataFrame): The result GeoDataFrame to store processed reaches.
    visited (set): A set to keep track of visited reach IDs.
    """
    if current_reach_id in visited:
        return

    visited.add(current_reach_id)
    row = df[df['reach_id'] == current_reach_id]

    if not row.empty:
        row = row.iloc[0]
        result.loc[current_reach_id, df.columns.difference(['geometry'])] = row[df.columns.difference(['geometry'])]
        result.loc[current_reach_id, 'geometry'] = row['geometry']

        if direction == 'up':
            upstream_neighbours = str(row['rch_id_up']).split()
            if len(upstream_neighbours) > 1:
                facc_values = df[df['reach_id'].isin(map(int, upstream_neighbours))][['reach_id', 'facc']]
                next_reach_id = facc_values.loc[facc_values['facc'].idxmax()]['reach_id']
                process_neighbours(next_reach_id, direction, df, result, visited)
            elif len(upstream_neighbours) == 1:
                process_neighbours(int(upstream_neighbours[0]), direction, df, result, visited)
        elif direction == 'down':
            downstream_neighbours = str(row['rch_id_dn']).split()
            if len(downstream_neighbours) > 1:
                facc_values = df[df['reach_id'].isin(map(int, downstream_neighbours))][['reach_id', 'facc']]
                next_reach_id = facc_values.loc[facc_values['facc'].idxmax()]['reach_id']
                process_neighbours(next_reach_id, direction, df, result, visited)
            elif len(downstream_neighbours) == 1:
                process_neighbours(int(downstream_neighbours[0]), direction, df, result, visited)

def crop_node_gdf(node_gdf, start_dist_out, end_dist_out, result):
    print(f"Original node_gdf shape: {node_gdf.shape}")
    print(f"Unique (node_id, reach_id) pairs in original: {node_gdf.drop_duplicates(subset=['node_id', 'reach_id']).shape[0]}")

    # Check for duplicates in the original data
    duplicates = node_gdf[node_gdf.duplicated(subset=['node_id', 'reach_id'], keep=False)]
    if not duplicates.empty:
        print("Warning: Duplicates found in original node_gdf:")
        print(duplicates[['node_id', 'reach_id', 'dist_out']].sort_values(['node_id', 'reach_id']))

    node_gdf_cropped = node_gdf[(node_gdf['dist_out'] <= start_dist_out) & (node_gdf['dist_out'] >= end_dist_out)]
    print(f"After distance filtering: {node_gdf_cropped.shape}")
    print(f"Unique (node_id, reach_id) pairs after filtering: {node_gdf_cropped.drop_duplicates(subset=['node_id', 'reach_id']).shape[0]}")

    # Check for duplicates after filtering
    duplicates = node_gdf_cropped[node_gdf_cropped.duplicated(subset=['node_id', 'reach_id'], keep=False)]
    if not duplicates.empty:
        print("Warning: Duplicates found after distance filtering:")
        print(duplicates[['node_id', 'reach_id', 'dist_out']].sort_values(['node_id', 'reach_id']))

    # Join with result DataFrame
    node_gdf_cropped = node_gdf_cropped.join(result[['slope']], on='reach_id')
    print(f"After joining with result: {node_gdf_cropped.shape}")

    node_gdf_cropped.rename(columns={'geometry': 'original_geom'}, inplace=True)
    node_gdf_cropped.set_geometry('original_geom', inplace=True)
    node_gdf_cropped.to_crs('EPSG:3857', inplace=True)

    return node_gdf_cropped

def plot_nodes(node_gdf_cropped):
    """
    Plot the nodes on a map with a satellite background.

    Parameters:
    node_gdf_cropped (GeoDataFrame): The cropped node GeoDataFrame to plot.
    """
    osm_background = cimgt.GoogleTiles(style='satellite')
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    ax.add_image(osm_background, 14)
    plot_gdf = node_gdf_cropped.copy().to_crs('EPSG:4326')
    plot_gdf.plot(ax=ax, column='width', cmap='jet', legend=True, markersize=1)
    bounds = plot_gdf.total_bounds
    ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]])
    plt.show()

def save_elevations_gdf(cross_section_points, name):
    """
    Save the elevations GeoDataFrame to a parquet file.

    Parameters:
    cross_section_points (GeoDataFrame): The cross-section points GeoDataFrame.
    name (str): The name to use for the saved file.

    Returns:
    GeoDataFrame: The elevations GeoDataFrame.
    """
    unique_id = uuid.uuid4()
    unique_id_str = str(unique_id)
    all_elevations_gdf = perform_cross_section_sampling(cross_section_points, unique_id_str)
    all_elevations_gdf['edit_flag'] = all_elevations_gdf['edit_flag'].astype(str)
    all_elevations_gdf.to_parquet(f'/Users/jakegearon/projects/river_cross_section/data/all_elevations_gdf_{name}.parquet')
    return all_elevations_gdf

def calculate_relief(group):
    """
    Calculate the relief for a group of points.

    Parameters:
    group (DataFrame): The input group of points.

    Returns:
    float: The calculated relief.
    """
    center_third = group[group['dist_along'].between(group['dist_along'].quantile(0.33), group['dist_along'].quantile(0.66))]
    outer_two_thirds = group[~group.index.isin(center_third.index)]
    relief = center_third['elevation'].quantile(0.95) - outer_two_thirds['elevation'].quantile(0.15)
    return relief

def aggregate_cross_section_stats(all_elevations_gdf):
    """
    Aggregate statistics for cross-sections.

    Parameters:
    all_elevations_gdf (GeoDataFrame): The input elevations GeoDataFrame.

    Returns:
    GeoDataFrame: The aggregated cross-section statistics GeoDataFrame.
    """
    if 'cross_id' in all_elevations_gdf.columns:
        all_elevations_gdf.set_index('cross_id', inplace=True)
    elif 'cross_id' != all_elevations_gdf.index.name:
        raise KeyError("'cross_id' is neither in the columns nor the index.")

    cross_section_stats = all_elevations_gdf.groupby('cross_id').agg({
        'elevation': ['mean', 'var', 'skew', lambda x: kurtosis(x), 'median', 'std'],
        'slope': ['mean', 'std', 'skew', lambda x: kurtosis(x)]
    }).ffill()

    cross_section_stats.columns = ['_'.join(col).strip() for col in cross_section_stats.columns.values]
    cross_section_stats.rename(columns={'elevation_<lambda_0>': 'elevation_kurtosis', 'slope_<lambda_0>': 'slope_kurtosis'}, inplace=True)

    cross_section_stats['relief'] = all_elevations_gdf.groupby('cross_id').apply(calculate_relief)
    cross_section_stats['azimuth_range'] = all_elevations_gdf.groupby('cross_id')['azimuth'].apply(lambda x: x.max() - x.min())
    cross_section_stats['mean_slope'] = all_elevations_gdf.groupby('cross_id')['slope'].mean()
    cross_section_stats['std_slope'] = all_elevations_gdf.groupby('cross_id')['slope'].std()
    cross_section_stats['skew_slope'] = all_elevations_gdf.groupby('cross_id')['slope'].skew()
    cross_section_stats['kurtosis_slope'] = all_elevations_gdf.groupby('cross_id')['slope'].apply(kurtosis)

    all_elevations_gdf.reset_index(inplace=True)
    cross_section_stats = all_elevations_gdf.merge(cross_section_stats, on='cross_id', how='left')
    cross_section_stats = gpd.GeoDataFrame(cross_section_stats, geometry='geometry')
    return cross_section_stats

def save_cross_section_stats(cross_section_stats, name):
    """
    Save the cross-section statistics GeoDataFrame to a parquet file.

    Parameters:
    cross_section_stats (GeoDataFrame): The cross-section statistics GeoDataFrame.
    name (str): The name to use for the saved file.
    """
    cross_section_stats.to_parquet(f'/Users/jakegearon/projects/river_cross_section/data/cross_section_stats_{name}.parquet')
    


# %%