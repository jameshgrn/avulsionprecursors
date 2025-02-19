"""Core geometric operations for river analysis."""
from typing import List, Optional
import numpy as np
from shapely.geometry import Point, LineString
import geopandas as gpd
from ..sword.base import SwordNode, SwordReach

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the bearing between two points."""
    bearing = np.arctan2(lat2 - lat1, lon2 - lon1)
    return np.degrees(bearing) % 360

def adjust_slope_per_segment(slope: float, azimuth_difference: float) -> float:
    """
    Adjust slope based on azimuth difference using trigonometric functions.
    
    Args:
        slope: Original slope value
        azimuth_difference: Difference in azimuth (degrees)
        
    Returns:
        float: Adjusted slope value
    """
    azimuth_difference_rad = np.radians(azimuth_difference)
    correction_factor = np.cos(azimuth_difference_rad)
    return slope * correction_factor

def calculate_mean_direction(lines: List[LineString]) -> float:
    """
    Calculate mean direction of multiple LineStrings.
    
    Args:
        lines: List of LineString geometries
        
    Returns:
        float: Mean direction in radians
    """
    angles = []
    for line in lines:
        if line and len(line.coords) > 1:
            start, end = line.coords[:2]
            angle = np.arctan2(end[1] - start[1], end[0] - start[0])
            angles.append(angle)
    return np.mean(angles) if angles else 0.0 