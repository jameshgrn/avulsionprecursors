"""Pipeline for cross-section labeling."""
from pathlib import Path
from typing import Optional
import geopandas as gpd
from ..db.config import DBConfig
from ..db.sword import SWORDDatabase
from ..geometry.coordinator import GeometryCoordinator
from ..gui.config import GUIConfig
from ..gui.labeler import CrossSectionLabeler

class LabelingPipeline:
    """Pipeline for labeling river cross-sections."""
    
    def __init__(
        self,
        river_name: str,
        output_dir: Optional[Path] = None,
        db_config: Optional[DBConfig] = None
    ):
        self.river_name = river_name
        self.output_dir = output_dir or Path('data')
        self.db_config = db_config or DBConfig.from_env()
        self.db = SWORDDatabase(self.db_config)
        self.gui_config = GUIConfig()
        
    def run(
        self,
        start_reach_id: int,
        start_dist: float,
        end_dist: float,
        skip_n: int = 1,
        sample_only: bool = False
    ) -> None:
        """
        Run the labeling pipeline.
        
        Args:
            start_reach_id: ID of the starting reach
            start_dist: Starting distance from outlet
            end_dist: Ending distance from outlet
            skip_n: Number of cross-sections to skip
            sample_only: Whether to sample only without labeling
        """
        # Get reaches
        reaches = self.db.get_reaches_by_distance(
            start_reach_id,
            start_dist,
            end_dist
        )
        
        if not reaches:
            raise ValueError(
                f"No reaches found for {self.river_name} "
                f"between {start_dist} and {end_dist}"
            )
        
        # Process each reach
        for reach in reaches:
            print(f"\nProcessing reach {reach.reach_id}")
            
            # --- New Stage: ICEsat2 & BASED Processing ---
            # Sample the water surface elevation (WSE) profile via ICEsat2 routines.
            print("Sampling water surface elevation using ICEsat2 routines...")
            from avulsionprecursors.icesat.processor import process_reach_wse_profile
            reach.wse_profile = process_reach_wse_profile(reach)

            # Ensure discharge data exists. For Kanektok, use the manual value if missing.
            if not hasattr(reach, 'discharge') or reach.discharge is None:
                if self.river_name.lower() == "kanektok":
                    print("No discharge value found; setting manual discharge value 84.835 for Kanektok")
                    reach.discharge = 84.835
                else:
                    raise ValueError(f"Discharge not provided for reach {reach.reach_id}")

            # Run the BASED model using the WSE profile and discharge data.
            print("Running BASED model on discharge data...")
            from avulsionprecursors.analysis.based import run_based_model
            reach.based_results = run_based_model(reach.discharge, reach.wse_profile)
            # --- End New Stage ---

            # Create cross-sections based on the updated reach data.
            coordinator = GeometryCoordinator(reach)
            cross_sections = coordinator.create_cross_section_points()
            
            # Group by cross-section
            grouped = cross_sections.groupby('cross_id')
            cross_section_dfs = [group for _, group in grouped]
            
            # Labeling is disabled in this version.
            # Instead, just report the number of generated cross-sections.
            print(f"Generated {len(cross_section_dfs)} cross-sections for reach {reach.reach_id} (labeling disabled).")
    
    def _save_progress(self, reach_id: int, df: gpd.GeoDataFrame, labeled_points: dict) -> None:
        """Save intermediate labeling results."""
        progress_dir = self.output_dir / 'progress' / str(reach_id)
        progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cross-section data
        df.to_parquet(progress_dir / 'cross_section.parquet')
        
        # Save labeled points
        records = []
        for label, points in labeled_points.items():
            if points:
                x, y = points[0]
                records.append({
                    'label': label,
                    'dist_along': x,
                    'elevation': y,
                    'reach_id': reach_id,
                    'cross_id': df['cross_id'].iloc[0]
                })
        
        if records:
            gpd.GeoDataFrame(records).to_parquet(
                progress_dir / 'labeled_points.parquet'
            ) 