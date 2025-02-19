import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QLabel, QVBoxLayout, QWidget, QMessageBox, QTextEdit, QSizePolicy, QGridLayout, QHBoxLayout, QComboBox, QSlider, QShortcut
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import PickEvent
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QLabel, QVBoxLayout, QWidget, QMessageBox, QTextEdit
from PyQt5.QtGui import QFont
import pyarrow as pa
import pyarrow.parquet as pq
from functools import partial
from PyQt5.QtCore import Qt
import json
from sklearn.linear_model import LinearRegression
from PyQt5.QtGui import QKeySequence
from matplotlib.lines import Line2D  # Add this import

# V7_2 avulsion was ~1991
# V11_2 avulsion was ~1997
# RIOPIRAI older avulsion was ~2010-2011
#SULENGGUOLE began in 2011 and has grown since.
# Set the river name once at the top of the script
RIVER_NAME = 'VENEZ_2023'

plt.style.use('dark_background')
plt.rcParams['grid.color'] = "white"
plt.rcParams['grid.linestyle'] = "--"
plt.rcParams['grid.linewidth'] = 0.5

class CrossSectionViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.name = RIVER_NAME  # Use the global RIVER_NAME
        self.old_lambda_value = None
        self.initializeUI()
        self.setupData()
        self.setupLayout()
        self.connectEvents()
        self.setupStatusBar()  # Add status bar setup

        # Load data_dict
        with open('src/data/data/data_dict.json', 'r') as f:
            self.data_dict = json.load(f)
        

    def initializeUI(self):
        # Window Configuration
        print("Initializing Cross Section Viewer")
        self.setWindowTitle(f"Cross Section Viewer - {self.name}")
        self.setGeometry(100, 100, 1600, 1200)  # Window size

    def setupData(self):
        # Data Initialization
        self.load_data()
        self.original_position = None  # For tracking original positions in edit mode
        self.current_edit_label = None  # Currently selected label for editing
        self.edit_mode = False  # Flag to indicate if edit mode is active
        self.actions_history = []  # To track actions for undo functionality

    def load_data_dict(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    
    def setupLayout(self):
        # Main Layout Configuration
        main_layout = QVBoxLayout()  # This will hold everything

        # Top Controls Layout
        top_controls_layout = QHBoxLayout()
        top_controls_layout.setSpacing(10)
        top_controls_layout.setContentsMargins(10, 10, 10, 10)

        # Node ID Entry and Plot Button
        top_controls_layout.addWidget(QLabel("Node ID:"))
        self.node_id_entry = QLineEdit()
        self.node_id_entry.setText("1")  # Set a default Node ID
        top_controls_layout.addWidget(self.node_id_entry)
        self.plot_button = self.createStyledButton("Plot", self.on_plot, "Plot the cross-section for the given Node ID")
        top_controls_layout.addWidget(self.plot_button)

        # Edit and Delete Buttons
        self.setupEditDeleteButtons(top_controls_layout)

        # Save and Undo Buttons
        self.setupSaveUndoButtons(top_controls_layout)

        # Center to Points Button
        self.center_button = self.createStyledButton("Center to Points", self.center_to_points, "Center the view on key points")
        top_controls_layout.addWidget(self.center_button)

        # Delete Cross Section Button
        self.delete_cross_section_button = self.createStyledButton("Delete Cross Section", self.delete_cross_section, "Delete the entire cross section for the current Node ID")
        top_controls_layout.addWidget(self.delete_cross_section_button)

        # Changes Log - Make the text box smaller and add to top controls
        self.changes_log = QTextEdit()
        self.changes_log.setReadOnly(True)
        self.changes_log.setMaximumHeight(100)  # Set maximum height to limit space usage
        top_controls_layout.addWidget(self.changes_log)

        # Add the top controls layout to the main layout
        main_layout.addLayout(top_controls_layout)

        # Create a horizontal layout for the plot and slider
        plot_slider_layout = QHBoxLayout()

        # Matplotlib Figure and Canvas
        self.setupFigureCanvas(plot_slider_layout)

        # Add slider for aspect ratio control
        self.aspect_ratio_slider = QSlider(Qt.Vertical)  # Vertical slider
        self.aspect_ratio_slider.setMinimum(1)
        self.aspect_ratio_slider.setMaximum(1000)
        self.aspect_ratio_slider.setValue(500)  # Default value
        self.aspect_ratio_slider.setTickPosition(QSlider.TicksLeft)
        self.aspect_ratio_slider.setTickInterval(100)
        self.aspect_ratio_slider.valueChanged.connect(self.update_aspect_ratio)

        # Add label for VE display
        self.ve_label = QLabel("VE: 500")

        # Create a vertical layout for the slider and its label
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(QLabel("Vertical Exaggeration:"))
        slider_layout.addWidget(self.aspect_ratio_slider)
        slider_layout.addWidget(self.ve_label)

        # Add the slider layout to the plot_slider_layout
        plot_slider_layout.addLayout(slider_layout)

        # Add the plot_slider_layout to the main layout
        main_layout.addLayout(plot_slider_layout)

        # Set the central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def createStyledButton(self, text, callback, tooltip=None):
        button = QPushButton(text)
        button.clicked.connect(callback)
        font = QFont("Arial", 10, QFont.Bold)  # Adjust font size as needed
        button.setFont(font)
        sizePolicy = button.sizePolicy()
        sizePolicy.setVerticalPolicy(QSizePolicy.Preferred)
        button.setSizePolicy(sizePolicy)
        # Convert the height to an integer
        button.setMaximumHeight(int(button.sizeHint().height()))  # Adjust the multiplier to find a good balance  # Ensure the height is an integer
        button.setStyleSheet("QPushButton { padding: 2px; }")  # Example: Reduce padding
        if tooltip:
            button.setToolTip(tooltip)
        return button

    def setupFigureCanvas(self, layout):
        # Matplotlib Figure and Canvas Configuration
        self.figure = Figure(figsize=(10, 10))  # Adjusted from (12, 8) to (10, 10)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(800, 600)  # Adjusted from (800, 600)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create a vertical layout to stack the canvas and toolbar
        canvas_toolbar_layout = QVBoxLayout()
        canvas_toolbar_layout.addWidget(self.canvas)
        canvas_toolbar_layout.addWidget(NavigationToolbar(self.canvas, self))
        
        layout.addLayout(canvas_toolbar_layout)
        layout.setAlignment(Qt.AlignTop)

    def setupEditDeleteButtons(self, layout):
        # Edit and Delete Buttons Configuration
        buttons_grid = QGridLayout()
        self.edit_buttons = {}
        labels = ['channel', 'ridge1', 'floodplain1', 'ridge2', 'floodplain2']
        for i, label in enumerate(labels):
            edit_btn = self.createStyledButton(f"Edit {label}", partial(self.enable_edit_mode, label))
            del_btn = self.createStyledButton(f"Delete {label}", partial(self.delete_label_data, label))
            buttons_grid.addWidget(edit_btn, i, 0)
            buttons_grid.addWidget(del_btn, i, 1)
            self.edit_buttons[label] = edit_btn

        # Wrap buttons_grid in a QHBoxLayout to utilize side spaces
        buttons_hbox = QHBoxLayout()
        buttons_hbox.addStretch()  # Add stretch to push the grid to the center
        buttons_hbox.addLayout(buttons_grid)
        buttons_hbox.addStretch()  # Add stretch to ensure even spacing

        layout.addLayout(buttons_hbox)  # Add the horizontal layout to the main layout

    def setupSaveUndoButtons(self, layout):
        # Save and Undo Buttons Configuration
        save_undo_hbox = QHBoxLayout()  # Use QHBoxLayout to place buttons horizontally
        self.save_button = self.createStyledButton("Save", self.save_data)
        save_undo_hbox.addWidget(self.save_button)
        self.undo_button = self.createStyledButton("Undo", self.undo_last_action)
        save_undo_hbox.addWidget(self.undo_button)

        save_undo_hbox.addStretch(1)  # Add stretch to align buttons to the left

        layout.addLayout(save_undo_hbox)  # Add the horizontal layout to the main layout

    def setupStatusBar(self):
        self.statusBar = self.statusBar()
        self.statusBar.showMessage("Ready")

    def connectEvents(self):
        # Event Connections
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.tooltip = self.canvas.figure.text(0, 0, "", va="bottom", ha="left")
        self.save_button.clicked.connect(lambda: self.save_data(verbose=True))
        
        # New event connections for hotkeys
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # Existing shortcuts
        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(self.save_data)
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo_last_action)

        # New attributes for panning and zooming
        self.panning = False
        self.zooming = False
        self.pan_start = None

        self.plot_initial_bottom_plot()  # Initial plot

    def on_key_press(self, event):
        if event.key == 'u':
            self.undo_last_action()
        elif event.key == 'd':
            self.save_data(verbose=True)
            self.load_next_cross_section()  # You'll need to implement this method
        elif event.key == 'o':
            self.activate_zoom_mode()
        elif event.key == 'p':
            self.activate_pan_mode()
        elif event.key == 'h':
            self.reset_view()

    def on_scroll(self, event):
        if event.button == 'up':
            self.zoom(1.1, event)
        elif event.button == 'down':
            self.zoom(0.9, event)

    def on_mouse_press(self, event):
        if event.button == 2:  # Middle mouse button
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)
            self.canvas.setCursor(Qt.ClosedHandCursor)
        elif event.button == 8:  # Side mouse button
            self.undo_last_action()

    def on_mouse_release(self, event):
        if event.button == 2:  # Middle mouse button
            self.panning = False
            self.canvas.setCursor(Qt.ArrowCursor)

    def on_mouse_move(self, event):
        if self.panning and event.inaxes == self.ax1:
            self.pan(event)

    def activate_zoom_mode(self):
        self.zooming = True
        self.panning = False
        self.canvas.setCursor(Qt.CrossCursor)

    def activate_pan_mode(self):
        self.panning = True
        self.zooming = False
        self.canvas.setCursor(Qt.OpenHandCursor)

    def reset_view(self):
        self.ax1.autoscale()
        self.canvas.draw()

    def zoom(self, factor, event):
        cur_xlim = self.ax1.get_xlim()
        cur_ylim = self.ax1.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return  # Cursor is outside the axes

        new_width = (cur_xlim[1] - cur_xlim[0]) * factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * factor
        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
        
        # Convert to tuples for set_xlim/set_ylim
        self.ax1.set_xlim((xdata - new_width * (1-relx), xdata + new_width * (relx)))
        self.ax1.set_ylim((ydata - new_height * (1-rely), ydata + new_height * (rely)))
        self.canvas.draw_idle()

    def load_data(self):
        import duckdb

        if not self.name:
            raise ValueError("River name must be set before loading data")
        
        self.data_dict = self.load_data_dict('data/data_dict.json')
        # Initialize DuckDB connection and enable spatial extension
        self.conn = duckdb.connect(database=':memory:', read_only=False)
        self.conn.execute("INSTALL 'spatial';")
        self.conn.execute("LOAD 'spatial';")
        
        # Read elevation data into DuckDB as a table named elevation_data
        self.elevation_table_name = "elevation_data"
        self.conn.execute(f"CREATE TABLE {self.elevation_table_name} AS SELECT * FROM 'src/data/data/all_elevations_gdf_{self.name}.parquet'")
        
        # Check if recalculated_edited.csv exists, if so, load it, else load recalculated.csv
        import os
        recalculated_edited_path = f"src/data/data/{self.name}_recalculated_edited.csv"
        recalculated_path = f"src/data/data/{self.name}_recalculated.csv"
        
        if os.path.exists(recalculated_edited_path):
            self.river_table_name = "river_data"
            self.conn.execute(f"CREATE TABLE {self.river_table_name} AS SELECT * FROM '{recalculated_edited_path}'")
        else:
            self.river_table_name = "river_data"
            self.conn.execute(f"CREATE TABLE {self.river_table_name} AS SELECT * FROM '{recalculated_path}'")

    def plot_initial_bottom_plot(self):
        if not hasattr(self, 'river_table_name') or not self.name:
            print("Data not loaded yet.")
            return  # Exit the method if data is not ready
        print("Plotting initial bottom plot...")
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.grid(True)
        query = f"SELECT dist_out, lambda, node_id, reach_id FROM {self.river_table_name} WHERE river_name = '{self.name}'"
        print(f"Executing query: {query}")  # Debug: Print the query
        full_river_data = self.conn.execute(query).df()
        
        # Debug: Print the first few rows of the river_table
        print(f"First few rows of {self.river_table_name}:")
        print(self.conn.execute(f"SELECT * FROM {self.river_table_name} LIMIT 5").df())
        
        # Debug: Check if river_name column exists and its unique values
        print("Unique values in river_name column:")
        print(self.conn.execute(f"SELECT DISTINCT river_name FROM {self.river_table_name}").df())
        
        full_river_data.drop_duplicates(subset=['node_id', 'reach_id'], inplace=True)
        
        print(f"Data shape: {full_river_data.shape}")
        print(f"Data head:\n{full_river_data.head()}")
        
        self.scatter = ax.scatter(full_river_data['dist_out'], full_river_data['lambda'], picker=True)
        ax.axhline(2, color='red', linestyle='--', label=r'$\Lambda = 2$')
        ax.set_xlabel('Dist Out')
        ax.set_ylabel('Lambda')
        ax.set_yscale('log')
        if self.name in self.data_dict:
            riv_data = self.data_dict[self.name]
            # Plot avulsion lines
            avulsion_lines = riv_data.get("avulsion_lines", [])
            for avulsion_line in avulsion_lines:
                position = avulsion_line.get('position', None)
                if position is not None:
                    ax.axvline(x=position * 1000, color='yellow', linestyle='--', linewidth=1)  # Convert km to m
            # Plot crevasse splay lines
            crevasse_splay_lines = riv_data.get("crevasse_splay_lines", [])
            for crevasse_line in crevasse_splay_lines:
                position = crevasse_line.get('position', None)
                if position is not None:
                    ax.axvline(x=position * 1000, color='pink', linestyle=':', linewidth=1)  # Convert km to m
        
        # Add these lines to force the plot to update
        self.canvas.draw()
        self.canvas.flush_events()
        
    def enable_edit_mode(self, label):
        self.current_edit_label = label
        self.edit_mode = True
        self.select_label_for_editing(label)  # Ensure this is called when entering edit mode
        for lbl, btn in self.edit_buttons.items():
            btn.setText(f"Edit {lbl}" if lbl != label else f"Editing {label}...")
        
    def on_plot(self):
        node_id = self.node_id_entry.text()
        self.node_id = node_id
        if node_id:
            try:
                node_id_int = int(node_id)
                plot_data_query = f"SELECT * FROM {self.elevation_table_name} WHERE node_id = {node_id_int}"
                df = self.conn.execute(plot_data_query).df()  # Convert the result to a pandas DataFrame for plotting
                df.drop_duplicates(subset=['node_id', 'reach_id'], inplace=True)
                if not df.empty:
                    self.plot_cross_section(df)
                else:
                    QMessageBox.information(self, "No Data", "No data found for the given input.")
            except ValueError:
                QMessageBox.warning(self, "Input Error", "Node ID must be an integer.")
        else:
            QMessageBox.warning(self, "Input Error", "Please enter a Node ID.")
            
    def plot_cross_section(self, df):
        if not hasattr(self, 'ax1') or not hasattr(self, 'ax2'):
            self.figure.clear()
            self.ax1 = self.figure.add_subplot(211)
            self.ax2 = self.figure.add_subplot(212)
            self.figure.subplots_adjust(hspace=1.0)
        else:
            self.ax1.clear()
            self.ax2.clear()

        node_id = df['node_id'].iloc[0]
        self.node_id = node_id

        # Remove the chunking code and use the original plotting logic

    def center_on_key_points(self):
        if hasattr(self, 'ax1'):
            labels = ['channel', 'ridge1', 'floodplain1', 'ridge2', 'floodplain2']
            x_data = []
            y_data = []

            for label in labels:
                for artist in self.ax1.get_children():
                    if isinstance(artist, plt.Line2D) and artist.get_label() == label:
                        x_data.extend(np.array(artist.get_xdata()).tolist())  # Convert numpy array to list
                        y_data.extend(np.array(artist.get_ydata()).tolist())  # Convert numpy array to list

            if x_data and y_data:
                x_min, x_max = min(x_data), max(x_data)
                y_min, y_max = min(y_data), max(y_data)

                # Calculate current aspect ratio
                current_aspect = (y_max - y_min) / (x_max - x_min)

                # Add padding (adjust the factor as needed)
                padding_factor = 0.5  # 50% padding
                x_padding = (x_max - x_min) * padding_factor
                y_padding = (y_max - y_min) * padding_factor

                new_x_min, new_x_max = x_min - x_padding, x_max + x_padding
                new_y_min, new_y_max = y_min - y_padding, y_max + y_padding

                # Adjust y-limits to maintain aspect ratio
                y_center = (new_y_max + new_y_min) / 2
                y_range = (new_x_max - new_x_min) * current_aspect
                new_y_min = y_center - y_range / 2
                new_y_max = y_center + y_range / 2

                self.ax1.set_xlim(new_x_min, new_x_max)
                self.ax1.set_ylim(new_y_min, new_y_max)
            else:
                print("No ridge, floodplain, or channel points found to center on.")

    def center_to_points(self):
        self.center_on_key_points()
        self.canvas.draw()

    # Event Handlers
    def on_hover(self, event):
        visibility = self.tooltip.get_visible()
        if event.inaxes is not None:
            contains, attr = self.scatter.contains(event)
            if contains:
                index = attr['ind'][0]
                offsets = np.array(self.scatter.get_offsets())  # Convert to numpy array
                x, y = offsets[index]
                self.tooltip.set_text(f'Node ID: {index}, Dist Out: {x:.2f}')
                self.tooltip.set_position((event.xdata, event.ydata))
                self.tooltip.set_visible(True)
                self.canvas.draw_idle()
            else:
                if visibility:
                    self.tooltip.set_visible(False)
                    self.canvas.draw_idle()
                    
    def on_click(self, event):
        if self.panning:
            self.pan(event)
        elif self.zooming:
            self.zoom_to_rect(event)
        elif self.edit_mode and self.current_edit_label:
            # Existing edit mode logic
            new_x, new_y = event.xdata, event.ydata
            if new_x is not None and new_y is not None:
                print(f"Clicked at ({new_x}, {new_y})")
                print(f"Current edit label: {self.current_edit_label}")
                self.update_label_position(self.current_edit_label, new_x, new_y)
                self.edit_mode = False
                self.current_edit_label = None
                for lbl, btn in self.edit_buttons.items():
                    btn.setText(f"Edit {lbl}")
            else:
                print("Click was outside the plot area.")
                
    def on_label_selected(self, label):
        self.select_label_for_editing(label)
        # Optionally, update the UI to indicate that this label is in edit mode.
     

    def on_pick(self, event):
        if hasattr(event, 'ind') and isinstance(event.artist, type(self.scatter)):
            index = event.ind[0]  # Access the index directly
            query = f"SELECT dist_out, lambda, node_id, reach_id FROM {self.river_table_name} WHERE river_name = '{self.name}'"
            full_river_data = self.conn.execute(query).df()
            full_river_data.drop_duplicates(subset=['node_id', 'reach_id'], inplace=True)
            node_id = full_river_data.iloc[index]['node_id']
            print(f"Selected node_id: {node_id}")
            self.node_id_entry.setText(str(node_id))
            # Filter the data based on the selected node_id and plot the cross-section using DuckDB
            plot_data_query = f"SELECT * FROM {self.elevation_table_name} WHERE node_id = {node_id}"
            df = self.conn.execute(plot_data_query).df()  # Convert the result to a pandas DataFrame for plotting
            df.drop_duplicates(subset=['node_id', 'reach_id'], inplace=True)
            if not df.empty:
                self.plot_cross_section(df)
            else:
                QMessageBox.information(self, "No Data", "No data found for the selected node.")
    
    # Data Manipulation and Update
    
    def select_label_for_editing(self, label):
        try:
            # First, convert the text to a float, then to an integer
            node_id = int(float(self.node_id_entry.text()))
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Node ID must be a valid number.")
            return  # Exit the method if the conversion fails

        query = f"""
        SELECT {label}_dist_along, {label}_elevation
        FROM {self.river_table_name}
        WHERE node_id = {node_id} AND river_name = '{self.name}';
        """
        result = self.conn.execute(query).fetchone()
        if result:
            self.original_position = {'x': result[0], 'y': result[1]}
            print(f"Original position of {label}: {self.original_position}")
        else:
            self.original_position = None
            print(f"No original position found for {label}.")
            
    def update_label_position(self, label, x, y):
        print(f"Attempting to update {label} to new position: x={x}, y={y}")
        old_values_query = f"SELECT lambda FROM {self.river_table_name} WHERE node_id = {self.node_id} AND river_name = '{self.name}'"
        old_values = self.conn.execute(old_values_query).df()
        if not old_values.empty:
            self.old_lambda_value = old_values['lambda'].iloc[0]
        else:
            self.old_lambda_value = None
        if not self.original_position:
            print("No original position stored. Cannot proceed with update.")
            return
        try:
            # Store current state for undo
            current_state_query = f"SELECT * FROM {self.river_table_name} WHERE node_id = {self.node_id} AND river_name = '{self.name}'"
            current_state = self.conn.execute(current_state_query).df()
            self.actions_history.append((self.node_id, current_state))

            # Store current limits before updating
            xlim = self.ax1.get_xlim()
            ylim = self.ax1.get_ylim()

            self.update_database_with_new_position(label, x, y)
            self.recalculate_derived_values(label, x, y)
            self.canvas.draw()

            # Restore limits after refresh
            self.ax1.set_xlim(xlim)
            self.ax1.set_ylim(ylim)
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating {label}: {e}")

    def update_database_with_new_position(self, label, x, y):
        print(f"Attempting to update {label} to new position: x={x}, y={y}")
        if pd.isna(x) or pd.isna(y):
            print(f"Cannot update {label} with NaN values.")
            return

        update_query = f"""
        UPDATE {self.river_table_name}
        SET {label}_dist_along = ?, {label}_elevation = ?
        WHERE node_id = ? AND river_name = ?;
        """
        try:
            self.conn.execute(update_query, (float(x), float(y), int(self.node_id), self.name))
            print(f"Database updated for {label} to new position: x={x}, y={y}")
        except Exception as e:
            print(f"Error updating {label}: {e}")
            

    def recalculate_derived_values(self, label, x, y):
        # Offload heavy recalculation to a background thread, using a dedicated DB connection for thread safety.
        import duckdb  # ensure duckdb is available in this scope
        def heavy_recalc(label, x, y):
            print(f"Recalculating derived values for {label} in background thread")
            if not self.original_position:
                print("No original position stored. Cannot proceed with update.")
                return
            print(f"Attempting to update {label} to new position: x={x}, y={y}")
            print(f"Original position was: {self.original_position}")
            node_id = self.node_id
            # Create a new, local DuckDB connection so that we don't use self.conn from multiple threads
            local_conn = duckdb.connect(database=':memory:', read_only=False)
            try:
                local_conn.execute("INSTALL 'spatial';")
                local_conn.execute("LOAD 'spatial';")
                # If needed, re-create temporary tables or load data into this connection before proceeding.
                local_conn.begin()
                query = f"SELECT * FROM {self.river_table_name} WHERE node_id = {node_id} AND river_name = '{self.name}'"
                river_output = local_conn.execute(query).df()
                river_output.drop_duplicates(subset=['node_id', 'reach_id'], inplace=True)
                # ... perform heavy recalculation steps here ...
                local_conn.commit()
            except Exception as e:
                local_conn.rollback()
                print(f"Error in background recalculation for {label}: {str(e)}")
            finally:
                local_conn.close()
        worker = WorkerQRunnable(heavy_recalc, label, x, y)
        worker.signals.finished.connect(lambda: print(f"Finished recalculation for {label}"))
        worker.signals.error.connect(lambda err: print(f"Error in worker for {label}: {err}"))
        QThreadPool.globalInstance().start(worker)

    def undo_last_action(self):
        if self.actions_history:
            node_id, prev_state = self.actions_history.pop()
            print(f"Undoing last action for node_id {node_id}")

            # Begin a transaction to revert to the previous state
            try:
                self.conn.begin()
                # Delete current state
                self.conn.execute(f"DELETE FROM {self.river_table_name} WHERE node_id = {node_id} AND river_name = '{self.name}'")
                # Insert previous state
                for _, row in prev_state.iterrows():
                    # Prepare the values for insertion, handling NaNs and formatting
                    values = []
                    for col in prev_state.columns:
                        value = row[col]
                        if pd.isna(value):
                            values.append('NULL')
                        elif isinstance(value, str):
                            values.append(f"'{value}'")
                        else:
                            values.append(str(value))
                    insert_query = f"""
                    INSERT INTO {self.river_table_name} ({', '.join(prev_state.columns)}) VALUES ({', '.join(values)})
                    """
                    self.conn.execute(insert_query)
                self.conn.commit()
                self.canvas.draw()
            except Exception as e:
                self.conn.rollback()
                print(f"Failed to undo last action: {e}")
        else:
            print("No actions to undo")
            

    def delete_label_data(self, label, delete_entire_node=False):
        print(f"Deleting data for {label}")
        if self.node_id:
            try:
                # Save current state for undo
                current_state_query = f"SELECT * FROM {self.river_table_name} WHERE node_id = {self.node_id} AND river_name = '{self.name}'"
                current_state = self.conn.execute(current_state_query).df()
                self.actions_history.append((self.node_id, current_state))

                if delete_entire_node:
                    # Delete the entire row for the current node_id
                    delete_query = f"DELETE FROM {self.river_table_name} WHERE node_id = {self.node_id} AND river_name = '{self.name}'"
                else:
                    # Delete only the specific label data for the current node_id
                    delete_query = f"""
                    UPDATE {self.river_table_name}
                    SET {label}_dist_along = NULL, {label}_elevation = NULL
                    WHERE node_id = {self.node_id} AND river_name = '{self.name}'
                    """
                self.conn.execute(delete_query)
                self.canvas.draw()
            except Exception as e:
                print(f"Error deleting {label} data: {e}")
        
    #Utility and Cleanup
    def save_data(self, verbose=True):
        print("Saving data...")

        try:
            self.conn.commit()
            export_query = f"COPY (SELECT * FROM {self.river_table_name}) TO 'src/data/data/{self.name}_recalculated_edited.csv' (FORMAT CSV, HEADER)"
            self.conn.execute(export_query)
            if verbose:
                QMessageBox.information(self, "Data Saved", f"All changes to the {self.name} river data have been saved.")
            
            # Print final statistics
            print("\nFinal statistics:")
            for column in ['gamma_mean', 'superelevation_mean', 'lambda', 'single_side_measurement']:
                df = self.conn.execute(f"SELECT {column} FROM {self.river_table_name}").df()
                if df[column].dtype == bool:
                    print(f"{column}:")
                    print(f"  True count: {df[column].sum()}")
                    print(f"  False count: {(~df[column]).sum()}")
                else:
                    print(f"{column}:")
                    print(f"  Min: {df[column].min()}")
                    print(f"  Max: {df[column].max()}")
                    print(f"  Mean: {df[column].mean()}")
                    print(f"  Median: {df[column].median()}")
                    print(f"  NaN count: {df[column].isna().sum()}")
                print()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save changes: {e}")
            print(f"Error saving data: {e}")

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
            plt.close('all')
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def closeEvent(self, event):
        try:
            self.save_data(verbose=False)
            self.cleanup()
        finally:
            event.accept()

    def update_aspect_ratio(self):
        if hasattr(self, 'ax1'):
            ve = self.aspect_ratio_slider.value()
            self.update_y_limits(ve)
            actual_ve = self.calculate_actual_ve()
            self.ve_label.setText(f"VE: {actual_ve:.2f}")
            self.canvas.draw()

    def update_y_limits(self, ve=None):
        if ve is None:
            ve = self.aspect_ratio_slider.value()
        
        y_data = np.array(self.ax1.get_lines()[0].get_ydata())  # Convert to numpy array
        y_min, y_max = np.nanmin(y_data), np.nanmax(y_data)  # Use numpy min/max
        y_range = y_max - y_min
        
        # Calculate new y limits based on vertical exaggeration
        y_center = (y_max + y_min) / 2
        new_y_range = y_range * (ve / 500)  # Normalize VE to 500 as the baseline
        
        new_y_min = y_center - new_y_range / 2
        new_y_max = y_center + new_y_range / 2
        
        self.ax1.set_ylim(new_y_min, new_y_max)

    def calculate_actual_ve(self):
        x_min, x_max = self.ax1.get_xlim()
        y_min, y_max = self.ax1.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Calculate the actual vertical exaggeration
        actual_ve = (y_range / x_range) * (self.figure.get_figwidth() / self.figure.get_figheight())
        
        return actual_ve

    def delete_cross_section(self):
        if not self.node_id:
            QMessageBox.warning(self, "Warning", "No cross section selected.")
            return

        reply = QMessageBox.question(self, 'Delete Cross Section',
                                     f"Are you sure you want to delete the entire cross section for node {self.node_id}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                # Save current state for undo
                current_state_query = f"SELECT * FROM {self.river_table_name} WHERE node_id = {self.node_id} AND river_name = '{self.name}'"
                current_state = self.conn.execute(current_state_query).df()
                self.actions_history.append((self.node_id, current_state))

                # Delete the entire row for the current node_id
                delete_query = f"DELETE FROM {self.river_table_name} WHERE node_id = {self.node_id} AND river_name = '{self.name}'"
                self.conn.execute(delete_query)

                # Also delete from elevation data
                delete_elevation_query = f"DELETE FROM {self.elevation_table_name} WHERE node_id = {self.node_id}"
                self.conn.execute(delete_elevation_query)

                # Commit the changes to the database
                self.conn.commit()

                QMessageBox.information(self, "Success", f"Cross section for node {self.node_id} has been deleted.")
                self.node_id = None  # Reset current node_id
                self.node_id_entry.clear()  # Clear the node_id entry field
                
                # Return to the initial view
                self.plot_initial_bottom_plot()
                
                # Clear the top plot
                self.ax1.clear()
                self.ax1.set_title("No cross section selected")
                self.canvas.draw()

            except Exception as e:
                self.conn.rollback()  # Rollback changes if an error occurred
                QMessageBox.critical(self, "Error", f"Failed to delete cross section: {e}")
                print(f"Error deleting cross section: {e}")

    def pan(self, event):
        if self.pan_start is None or event.xdata is None or event.ydata is None:
            return
        dx = self.pan_start[0] - event.xdata
        dy = self.pan_start[1] - event.ydata
        xlim = self.ax1.get_xlim()
        ylim = self.ax1.get_ylim()
        self.ax1.set_xlim(xlim + dx)
        self.ax1.set_ylim(ylim + dy)
        self.canvas.draw_idle()
        self.pan_start = (event.xdata, event.ydata)

    def zoom_to_rect(self, event):
        # Implement zoom to rectangle functionality
        pass  # You'll need to implement this based on your specific requirements

    def load_next_cross_section(self):
        # Implement logic to load the next cross section
        pass  # You'll need to implement this based on your data structure


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2B2B2B; /* Darker gray background */
        }
        QPushButton {
            background-color: #ADD8E6; /* Soft light blue buttons */
            color: #2B2B2B;
            border-radius: 5px;
            padding: 10px;
        }
        QPushButton:hover {
            background-color: #B0E0E6; /* Slightly different shade of light blue for hover */
        }
        QLabel {
            font-weight: bold;
            color: #2B2B2B; /* Making label text white for better contrast against dark background */
        }
        QTextEdit {
            border: 1px solid #C0C0C0;
            color: white; /* Making text color white for readability */
            background-color: #3C3F41; /* Slightly lighter gray than main background for contrast */
        }
    """)
    mainWin = CrossSectionViewer()
    mainWin.show()
    
    # Add this debug print
    print("Main window shown. Starting event loop.")
    
    # Optionally, you can force an immediate redraw
    mainWin.canvas.draw()
    mainWin.canvas.flush_events()
    
    sys.exit(app.exec_())