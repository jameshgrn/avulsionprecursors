import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

def calculate_azimuth(node_gdf):
    # Ensure the GeoDataFrame is sorted by 'dist_out'
    node_gdf = node_gdf.sort_values('dist_out')
    
    # Create previous and next geometry columns without grouping
    node_gdf['prev_geom'] = node_gdf['original_geom'].shift(1)
    node_gdf['next_geom'] = node_gdf['original_geom'].shift(-1)
    
    # Drop rows with NaN values in prev_geom or next_geom
    node_gdf.dropna(subset=['prev_geom', 'next_geom'], inplace=True)
    
    # Calculate differences in coordinates
    node_gdf['dx'] = (node_gdf['next_geom'].x - node_gdf['prev_geom'].x)
    node_gdf['dy'] = (node_gdf['next_geom'].y - node_gdf['prev_geom'].y)
    
    # Calculate azimuth
    node_gdf['azimuth'] = np.arctan2(node_gdf['dy'], node_gdf['dx'])

    return node_gdf

def make_cross_section(row):
    start = (
        row['original_geom'].x + 100 * row['width'] * np.cos(row['azimuth'] + np.pi / 2),
        row['original_geom'].y + 100 * row['width'] * np.sin(row['azimuth'] + np.pi / 2)
    )
    end = (
        row['original_geom'].x + 100 * row['width'] * np.cos(row['azimuth'] - np.pi / 2),
        row['original_geom'].y + 100 * row['width'] * np.sin(row['azimuth'] - np.pi / 2)
    )
    line = LineString([start, end])
    # print(f"Created LineString for node {row['node_id']}: {line}")

    return line

def create_cross_sections(node_gdf_w_azimuth):
    node_gdf_w_azimuth['perp_geometry'] = node_gdf_w_azimuth.apply(lambda row: make_cross_section(row), axis=1)
    node_gdf_w_azimuth = node_gdf_w_azimuth.set_geometry('perp_geometry')
    
    return node_gdf_w_azimuth

def create_points(row):
    """
    Create points along the cross-section LineString for a given row.

    This function calculates the number of points to create based on the length of the cross-section
    and interpolates points at regular intervals.

    Parameters:
    row (Series): A row from the GeoDataFrame containing cross-section data.

    Returns:
    list: A list of points along the cross-section.
    """
    length = row['perp_geometry'].length
    num_points = int(length / 2)
    distances = [i*2 for i in range(num_points+1)]
    points = [row['perp_geometry'].interpolate(dist) for dist in distances]
    return points

def create_cross_section_points(sword_cross_sections):
    sword_cross_sections = create_cross_sections(sword_cross_sections)
    sword_cross_sections['points'] = sword_cross_sections.apply(create_points, axis=1)
    cross_section_points = sword_cross_sections.explode('points').reset_index(drop=True)
    cross_section_points.rename(columns={'points': 'geometry'}, inplace=True)
    cross_section_points.set_geometry('geometry', inplace=True, crs='EPSG:3857')
    cross_section_points['cross_id'] = cross_section_points.groupby(['node_id', 'reach_id', 'dist_out']).ngroup()
    cross_section_points = cross_section_points.drop(columns=[col for col in cross_section_points.columns if isinstance(cross_section_points[col].dtype, gpd.array.GeometryDtype) and col != 'geometry'])
    # Debugging: Log cross section points
    # print("Cross section points created:")
    # print(cross_section_points[['node_id', 'reach_id', 'cross_id', 'dist_out']])
    return cross_section_points

def calculate_distance_along_cross_section(gdf):
    """
    Calculate the cumulative distance along each cross-section for each point in the GeoDataFrame.

    This function iterates over each unique cross_id, calculates the distance between consecutive points,
    and assigns the cumulative distance to the 'dist_along' column.

    Parameters:
    gdf (GeoDataFrame): The input GeoDataFrame with points along cross-sections.

    Returns:
    GeoDataFrame: The GeoDataFrame with the calculated cumulative distance along each cross-section.
    """
    # Iterate over each unique cross_id
    for cross_id in gdf['cross_id'].unique():
        # Select all points belonging to the current cross-section
        cross_section = gdf[gdf['cross_id'] == cross_id]
        distances = [0]
        
        # Calculate the cumulative distance for each point in the cross-section
        for i in range(1, len(cross_section)):
            # Calculate the distance between this point and the previous point
            dist = cross_section.iloc[i].geometry.distance(cross_section.iloc[i-1].geometry)
            distances.append(distances[-1] + dist)
        
        # Assign the calculated distances to the 'dist_along' column for the current cross-section
        gdf.loc[gdf['cross_id'] == cross_id, 'dist_along'] = distances

    return gdf

def calculate_mean_direction(gdf):
    """
    Calculate the mean direction (angle) of the LineStrings in the GeoDataFrame.

    This function calculates the angle of each LineString and computes the mean angle.

    Parameters:
    gdf (GeoDataFrame): The input GeoDataFrame with LineString geometries.

    Returns:
    float: The mean direction (angle) of the LineStrings.
    """
    angles = []
    for line in gdf.geometry:
        if line and len(line.coords) > 1:
            start, end = line.coords[:2]
            angle = np.arctan2(end[1] - start[1], end[0] - start[0])
            angles.append(angle)
    mean_angle = np.mean(angles)
    return mean_angle


def adjust_slope_per_segment(slope, azimuth_difference):
    """
    Adjust the slope based on the azimuth difference using trigonometric functions.
    This function assumes the azimuth difference is in degrees.
    """
    azimuth_difference_rad = np.radians(azimuth_difference)
    correction_factor = np.cos(azimuth_difference_rad)
    adjusted_slope = slope * correction_factor
    return adjusted_slope



def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two points.
    """
    bearing = np.arctan2(lat2 - lat1, lon2 - lon1)
    bearing_degrees = np.degrees(bearing) % 360
    return bearing_degrees

def calculate_bearing_for_df(df):
    """
    Calculate the bearing between consecutive points in a dataframe.
    """
    bearings = []
    for i in range(len(df) - 1):
        lat1, lon1 = df.iloc[i]['transformed_lat'], df.iloc[i]['transformed_lon']
        lat2, lon2 = df.iloc[i + 1]['transformed_lat'], df.iloc[i + 1]['transformed_lon']
        bearing = calculate_bearing(lat1, lon1, lat2, lon2)
        bearings.append(bearing)
    bearings.append(np.nan)
    return bearings


def perform_geometric_operations(node_gdf):
    # Check for duplicate nodes
    if node_gdf.duplicated(subset=['node_id', 'reach_id', 'dist_out']).any():
        print("Warning: Duplicate nodes found in node_gdf.")
    
    node_gdf_w_azimuth = calculate_azimuth(node_gdf)
    sword_cross_sections = create_cross_sections(node_gdf_w_azimuth)
    cross_section_points = create_cross_section_points(sword_cross_sections)
    cross_section_points = calculate_distance_along_cross_section(cross_section_points)
    
    # # Debugging: Log final cross section points
    # print("Final cross section points:")
    # print(cross_section_points[['node_id', 'cross_id', 'geometry']])
    
    return cross_section_points


