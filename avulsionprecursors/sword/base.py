"""Base classes for SWORD data model."""
from dataclasses import dataclass
from typing import List, Optional, Any
from shapely.geometry import Point, LineString
import geopandas as gpd
import numpy as np

@dataclass
class SwordNode:
    """Represents a SWORD node with its attributes."""
    node_id: int
    reach_id: int
    dist_out: float  # distance from outlet
    width: float
    geometry: Point  # shapely Point geometry
    elevation: Optional[float] = None
    slope: Optional[float] = None
    azimuth: Optional[float] = None
    cross_section: Optional[LineString] = None
    
    def calculate_azimuth(self, prev_node: Optional['SwordNode'], next_node: Optional['SwordNode']) -> None:
        """Calculate azimuth based on neighboring nodes."""
        if prev_node and next_node:
            dx = next_node.geometry.x - prev_node.geometry.x
            dy = next_node.geometry.y - prev_node.geometry.y
            self.azimuth = np.arctan2(dy, dx)
    
    def create_cross_section(self) -> None:
        """Create perpendicular cross-section line."""
        if self.azimuth is None:
            raise ValueError("Azimuth must be calculated before creating cross-section")
            
        start = (
            self.geometry.x + 100 * self.width * np.cos(self.azimuth + np.pi / 2),
            self.geometry.y + 100 * self.width * np.sin(self.azimuth + np.pi / 2)
        )
        end = (
            self.geometry.x + 100 * self.width * np.cos(self.azimuth - np.pi / 2),
            self.geometry.y + 100 * self.width * np.sin(self.azimuth - np.pi / 2)
        )
        self.cross_section = LineString([start, end])
    
    @classmethod
    def from_gdf(cls, row: gpd.GeoSeries) -> 'SwordNode':
        """Create a node from a GeoDataFrame row."""
        if not isinstance(row.geometry, Point):
            if hasattr(row.geometry, 'centroid'):
                geometry = row.geometry.centroid
            else:
                raise ValueError("Geometry must be a Point or have a centroid")
        else:
            geometry = row.geometry
            
        return cls(
            node_id=row.node_id,
            reach_id=row.reach_id,
            dist_out=row.dist_out,
            width=row.width,
            geometry=geometry,
            slope=row.get('slope'),
            elevation=None,
            azimuth=None,
            cross_section=None
        )

@dataclass
class SwordReach:
    """Represents a SWORD reach with its attributes and nodes."""
    reach_id: int
    nodes: List[SwordNode]
    rch_id_up: List[int]  # upstream reach IDs
    rch_id_dn: List[int]  # downstream reach IDs
    facc: float  # flow accumulation
    geometry: Any  # shapely geometry
    
    @property
    def start_dist(self) -> float:
        """Get the starting distance of the reach."""
        return max(node.dist_out for node in self.nodes)
    
    @property
    def end_dist(self) -> float:
        """Get the ending distance of the reach."""
        return min(node.dist_out for node in self.nodes) 