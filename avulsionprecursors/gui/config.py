"""Configuration for GUI components."""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

def get_default_features() -> List[str]:
    """Get default feature list."""
    return [
        # Original features
        'width', 'width_var', 'sinuosity', 'max_width', 'dist_out',
        'n_chan_mod', 'n_chan_max', 'facc', 'meand_len', 'slope',
        
        # Elevation-related features
        'elevation_min', 'elevation_max', 'elevation_range', 'elevation_mean',
        'relative_elevation_mean', 'relative_elevation_max',
        
        # Slope-related features
        'slope_r_mean', 'slope_r_max', 'local_slope_mean', 'local_slope_max',
        
        # Curvature-related features
        'curvature_mean', 'curvature_max', 'local_curvature_mean', 'local_curvature_max',
        
        # Distance-related features
        'dist_along_range'
    ]

def get_default_labels() -> List[str]:
    """Get default label list."""
    return ['channel', 'ridge1', 'floodplain1', 'ridge2', 'floodplain2']

def get_default_colors() -> Dict[str, str]:
    """Get default color mapping."""
    return {
        'channel': 'blue',
        'ridge1': 'green',
        'floodplain1': 'red',
        'ridge2': 'green',
        'floodplain2': 'red'
    }

@dataclass
class GUIConfig:
    """Configuration for cross-section labeling GUI."""
    features: List[str] = field(default_factory=get_default_features)
    labels: List[str] = field(default_factory=get_default_labels)
    colors: Dict[str, str] = field(default_factory=get_default_colors)
    
    @property
    def output_columns(self) -> List[str]:
        """Get columns for output DataFrame."""
        return (
            self.features + 
            [f'{label}_dist_along' for label in self.labels] +
            [f'{label}_elevation' for label in self.labels] +
            ['reach_id', 'node_id']
        )