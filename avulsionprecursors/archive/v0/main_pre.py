"""Pre-processing pipeline for river analysis."""
import os
from typing import Optional
from pathlib import Path
import geopandas as gpd
import numpy as np
import json
import pandas as pd

from .db.config import DBConfig
from .db.sword import SWORDDatabase
from .gee.base import GEEConfig, GEEInitializer
from .geometry.coordinator import GeometryCoordinator
from .icesat.config import ICESat2Config
from .icesat.processor import ICESat2Processor

class PreProcessor:
    """Handles pre-processing of river data."""
    
    def __init__(
        self,
        river_name: str,
        starting_reach_id: int,
        start_dist: float,
        end_dist: float,
        output_dir: Optional[Path] = None
    ):
        self.river_name = river_name
        self.starting_reach_id = starting_reach_id
        self.start_dist = start_dist
        self.end_dist = end_dist
        self.output_dir = output_dir or Path('data')
        self._initialize_components()
        
    def _initialize_components(self) -> None:
        """Initialize all required components."""
        # Database setup
        db_config = DBConfig.from_env()
        self.db = SWORDDatabase(db_config)
        
        # GEE setup
        gee_config = GEEConfig.from_env()
        gee_init = GEEInitializer(gee_config)
        gee_init.initialize()
        
        # ICESat-2 setup
        icesat_config = ICESat2Config()
        self.icesat = ICESat2Processor(icesat_config)
        self.icesat.initialize()
        
    def process(self) -> None:
        """Run the complete pre-processing pipeline."""
        # Get river reaches
        reaches = self.db.get_reaches_by_distance(
            self.starting_reach_id,
            self.start_dist,
            self.end_dist
        )
        
        if not reaches:
            raise ValueError(
                f"No reaches found for {self.river_name} "
                f"between {self.start_dist} and {self.end_dist}"
            )
        
        # Process geometry for each reach
        all_cross_sections = []
        for reach in reaches:
            # Create cross-sections
            coordinator = GeometryCoordinator(reach)
            cross_sections = coordinator.create_cross_section_points()
            
            # Get ICESat-2 data
            elevations = self.icesat.process_reach(reach)
            
            # Merge cross-sections with elevations
            cross_sections = self._merge_elevations(cross_sections, elevations)
            all_cross_sections.append(cross_sections)
        
        # Combine and save results
        combined = gpd.GeoDataFrame(
            pd.concat(all_cross_sections, ignore_index=True),
            crs="EPSG:4326"
        )
        self._save_results(combined)
        
        # Print statistics
        self._print_processing_stats(combined)
    
    def _merge_elevations(
        self, 
        cross_sections: gpd.GeoDataFrame,
        elevations: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Merge cross-sections with elevation data."""
        # Ensure same CRS
        cross_sections = cross_sections.to_crs(elevations.crs)
        
        # Find nearest elevation points
        merged = gpd.sjoin_nearest(
            cross_sections,
            elevations[['geometry', 'orthometric_height']],
            how='left',
            max_distance=50  # meters
        )
        
        # Add quality flags
        merged['elevation_source'] = np.where(
            merged['orthometric_height'].notna(),
            'icesat2',
            'missing'
        )
        
        return merged
    
    def _save_results(self, gdf: gpd.GeoDataFrame) -> None:
        """Save processing results."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        output_file = self.output_dir / f"{self.river_name}_cross_sections.parquet"
        gdf.to_parquet(output_file)
        
        # Save statistics
        stats_file = self.output_dir / f"{self.river_name}_stats.json"
        stats = {
            'total_cross_sections': len(gdf['cross_id'].unique()),
            'total_points': len(gdf),
            'points_with_elevation': gdf['orthometric_height'].notna().sum(),
            'mean_elevation': gdf['orthometric_height'].mean(),
            'std_elevation': gdf['orthometric_height'].std()
        }
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _print_processing_stats(self, gdf: gpd.GeoDataFrame) -> None:
        """Print processing statistics."""
        print(f"\nProcessing completed for {self.river_name}")
        print(f"Total cross-sections: {len(gdf['cross_id'].unique())}")
        print(f"Total points: {len(gdf)}")
        print(f"Points with elevation: {gdf['orthometric_height'].notna().sum()}")
        print(f"Mean elevation: {gdf['orthometric_height'].mean():.2f}")
        print(f"Std elevation: {gdf['orthometric_height'].std():.2f}")

def main():
    """Main entry point."""
    processor = PreProcessor(
        river_name="ANJOBONY",
        starting_reach_id=81170000241,
        start_dist=107018,
        end_dist=211,
        output_dir=Path("data")
    )
    processor.process()

if __name__ == "__main__":
    main()