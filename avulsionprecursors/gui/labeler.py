"""Cross-section labeling interface."""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib
matplotlib.use('Qt5Agg')
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from .config import GUIConfig
from ..analysis.metrics import calculate_ridge_metrics

class CrossSectionLabeler:
    """Interactive GUI for labeling river cross-sections."""
    
    def __init__(
        self,
        config: GUIConfig,
        output_dir: Path,
        river_name: str
    ):
        self.config = config
        self.output_dir = output_dir
        self.river_name = river_name
        self.current_label_idx = 0
        self.plotted_points: List[Any] = []
        self.labeled_points: Dict[str, List[Tuple[float, float]]] = {
            label: [] for label in config.labels
        }
        self.panning = False
        self.pan_start: Optional[Tuple[float, float]] = None
        self.zooming = False
        self.zoom_rect: Optional[Rectangle] = None
        
    def setup_figure(self) -> Tuple[Figure, Axes, Axes]:
        """Set up the figure and axes for plotting."""
        fig = plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(121)  # Cross-section plot
        ax2 = plt.subplot(122, projection=ccrs.PlateCarree())  # Map
        return fig, ax1, ax2
    
    def setup_event_handlers(self, fig: Figure, ax1: Axes, ax2: Axes) -> None:
        """Set up event handlers for interactive labeling."""
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        
        # Connect event handlers
        self.cid_click = fig.canvas.mpl_connect(
            'button_press_event', self._on_click
        )
        self.cid_move = fig.canvas.mpl_connect(
            'motion_notify_event', self._on_move
        )
        self.cid_release = fig.canvas.mpl_connect(
            'button_release_event', self._on_release
        )
        self.cid_scroll = fig.canvas.mpl_connect(
            'scroll_event', self._on_scroll
        )
        self.cid_key = fig.canvas.mpl_connect(
            'key_press_event', self._on_key
        )
    
    def _on_click(self, event: Any) -> None:
        """Handle mouse click events."""
        if event.inaxes != self.ax1:
            return
            
        if event.button == 2:  # Middle mouse button
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)
            return
            
        if self.panning or self.zooming:
            return
            
        if self.current_label_idx >= len(self.config.labels):
            return
            
        current_label = self.config.labels[self.current_label_idx]
        self.labeled_points[current_label].append((event.xdata, event.ydata))
        
        # Plot point
        point = self.ax1.plot(
            event.xdata,
            event.ydata,
            'X',
            markersize=10,
            color=self.config.colors[current_label]
        )[0]
        self.plotted_points.append(point)
        
        # Update label index and title
        self.current_label_idx += 1
        self._update_title()
        self.fig.canvas.draw()
    
    def _on_move(self, event: Any) -> None:
        """Handle mouse movement events."""
        if self.panning and event.inaxes == self.ax1:
            if self.pan_start is None:
                return
            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata
            self.ax1.set_xlim(self.ax1.get_xlim() + dx)
            self.ax1.set_ylim(self.ax1.get_ylim() + dy)
            self.fig.canvas.draw_idle()
    
    def _on_release(self, event: Any) -> None:
        """Handle mouse button release events."""
        if event.button == 2:
            self.panning = False
            self.pan_start = None
    
    def _on_scroll(self, event: Any) -> None:
        """Handle scroll wheel events."""
        if event.inaxes != self.ax1:
            return
            
        # Get current limits
        cur_xlim = self.ax1.get_xlim()
        cur_ylim = self.ax1.get_ylim()
        
        # Get the cursor position
        xdata, ydata = event.xdata, event.ydata
        
        # Calculate zoom factor
        base_scale = 1.1
        if event.button == 'up':
            scale_factor = 1 / base_scale
        else:
            scale_factor = base_scale
            
        # Calculate new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        # Set new limits
        self.ax1.set_xlim([
            xdata - new_width * (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0]),
            xdata + new_width * (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        ])
        self.ax1.set_ylim([
            ydata - new_height * (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0]),
            ydata + new_height * (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        ])
        
        self.fig.canvas.draw_idle()
    
    def plot_cross_section(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        predicted_points: Optional[Dict[str, float]] = None
    ) -> None:
        """Plot cross-section with elevation profile."""
        # Create a dummy 'elevation' column if not present, as DEM sampling hasn't been performed yet.
        if 'elevation' not in df.columns:
            df['elevation'] = float('nan')
        ax.plot(df['dist_along'], df['elevation'], '-o', markersize=2, color='blue')
        ax.set_xlabel('Along Track Distance')
        ax.set_ylabel('Elevation')
        
        # Plot predicted points if available
        if predicted_points:
            for label, dist in predicted_points.items():
                feature_name = label.split('_')[0]
                y = np.interp(dist, df['dist_along'], df['elevation'])
                ax.plot(
                    dist, y, 'X',
                    markersize=10,
                    color=self.config.colors[feature_name],
                    alpha=0.5
                )
    
    def plot_map(
        self,
        ax: plt.Axes,
        df: gpd.GeoDataFrame,
        zoom_level: int = 15
    ) -> None:
        """Plot satellite map with cross-section points."""
        # Add satellite imagery
        osm_background = cimgt.GoogleTiles(style='satellite', cache=True)
        ax.add_image(osm_background, zoom_level, interpolation='spline36')
        
        # Plot points
        scatter = ax.scatter(
            df.geometry.x,
            df.geometry.y,
            c=df['elevation'],
            cmap='terrain',
            marker=matplotlib.markers.MarkerStyle('o'),
            edgecolor='k',
            linewidth=0.5,
            s=10,
            transform=ccrs.PlateCarree()
        )
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, orientation='vertical', label='Elevation')
        
        # Set extent
        bounds = df.total_bounds
        buffer = 0.001
        if hasattr(ax, 'set_extent'):  # Check if it's a GeoAxes
            ax.set_extent([
                bounds[0] - buffer,
                bounds[2] + buffer,
                bounds[1] - buffer,
                bounds[3] + buffer
            ], crs=ccrs.PlateCarree())
        else:  # Regular Axes
            ax.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
            ax.set_ylim(bounds[1] - buffer, bounds[3] + buffer)
    
    def _on_key(self, event: Any) -> None:
        """Handle keyboard events."""
        if event.key == 'u':  # Undo
            self._undo_last_point()
        elif event.key == 'd':  # Done
            self._save_and_close()
        elif event.key == 'r':  # Reset view
            self._reset_view()
        elif event.key == 'h':  # Help
            self._show_help()
    
    def _update_title(self) -> None:
        """Update the plot title with current status."""
        if self.current_label_idx >= len(self.config.labels):
            title = "All points labeled. Press 'd' to save and continue."
        else:
            next_label = self.config.labels[self.current_label_idx]
            title = f"Pick {next_label} point"
        self.ax1.set_title(title)
    
    def _undo_last_point(self) -> None:
        """Remove the last labeled point."""
        if self.current_label_idx > 0 and self.plotted_points:
            self.current_label_idx -= 1
            current_label = self.config.labels[self.current_label_idx]
            if self.labeled_points[current_label]:
                self.labeled_points[current_label].pop()
            if self.plotted_points:
                point = self.plotted_points.pop()
                point.remove()
            self._update_title()
            self.fig.canvas.draw()
    
    def _save_and_close(self) -> None:
        """Save the labeled points and close the figure."""
        self._save_labels()
        plt.close(self.fig)
    
    def _reset_view(self) -> None:
        """Reset the view to show all data."""
        self.ax1.autoscale()
        self.fig.canvas.draw()
    
    def _show_help(self) -> None:
        """Display help text."""
        help_text = """
        Controls:
        - Left click: Place point
        - Middle click + drag: Pan
        - Scroll wheel: Zoom
        - 'u': Undo last point
        - 'd': Done/Save
        - 'r': Reset view
        - 'h': Show this help
        """
        print(help_text)
    
    def _save_labels(self) -> None:
        """Save labeled points to file."""
        output_file = self.output_dir / f"{self.river_name}_labels.csv"
        records = []
        
        for label, points in self.labeled_points.items():
            if points:
                x, y = points[0]
                records.append({
                    'label': label,
                    'dist_along': x,
                    'elevation': y
                })
        
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
    
    def label_cross_section(
        self,
        df: gpd.GeoDataFrame,
        predicted_points: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Main method to label a cross-section.
        
        Args:
            df: GeoDataFrame with cross-section data
            predicted_points: Optional dictionary of predicted point locations
            
        Returns:
            Dictionary of labeled points
        """
        fig, ax1, ax2 = self.setup_figure()
        self.setup_event_handlers(fig, ax1, ax2)
        
        # Plot data
        self.plot_cross_section(ax1, df, predicted_points)
        self.plot_map(ax2, df)
        
        # Set initial title
        self._update_title()
        
        plt.show()
        return self.labeled_points