"""Metrics calculation for river analysis."""
from typing import Dict, Any
import numpy as np
import pandas as pd

def calculate_ridge_metrics(
    ridge_elevation: float,
    floodplain_elevation: float,
    channel_elevation: float,
    depth: float,
    min_superelevation: float = 0.01
) -> Dict[str, float]:
    """
    Calculate ridge-based metrics.
    
    Args:
        ridge_elevation: Elevation of ridge
        floodplain_elevation: Elevation of floodplain
        channel_elevation: Elevation of channel
        depth: Channel depth
        min_superelevation: Minimum superelevation value
        
    Returns:
        Dictionary of calculated metrics
    """
    # Calculate basic differences
    ridge_height = ridge_elevation - channel_elevation
    floodplain_height = ridge_elevation - floodplain_elevation
    
    # Calculate a/b ratio
    a_b = ridge_height / depth
    
    # Determine denominator for superelevation
    if a_b <= 1.25:
        denominator = depth
    else:
        denominator = ridge_height
    
    # Calculate superelevation
    superelevation = max(
        floodplain_height / denominator,
        min_superelevation
    )
    
    return {
        'ridge_height': ridge_height,
        'floodplain_height': floodplain_height,
        'a_b': a_b,
        'superelevation': superelevation
    } 