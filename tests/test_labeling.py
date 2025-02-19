"""Test the labeling pipeline."""
import os
from pathlib import Path
from avulsionprecursors.pipeline.labeling import LabelingPipeline

def test_labeling_pipeline():
    """Test the labeling pipeline with a small sample."""
    # Set up test parameters
    river_name = "ANJOBONY"
    start_reach_id = 81170000241
    start_dist = 107018
    end_dist = 106000  # Small section for testing
    output_dir = Path("test_data")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = LabelingPipeline(
        river_name=river_name,
        output_dir=output_dir
    )
    
    # Run pipeline
    try:
        pipeline.run(
            start_reach_id=start_reach_id,
            start_dist=start_dist,
            end_dist=end_dist,
            skip_n=2  # Label every other cross-section
        )
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise
    finally:
        # Check outputs
        progress_dir = output_dir / 'progress'
        if progress_dir.exists():
            print("\nGenerated files:")
            for path in progress_dir.rglob('*'):
                print(f"- {path.relative_to(output_dir)}")

if __name__ == "__main__":
    test_labeling_pipeline() 