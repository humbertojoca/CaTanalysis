"""
PySide6 GUI for Calcium Transient Analysis

A modern graphical user interface for analyzing calcium transients
from linescan microscopy images.

Author: hjoca
Date: 2026-01-21
"""

import os
# Suppress Java native access warnings from JPype/Bioformats
os.environ['JAVA_TOOL_OPTIONS'] = '--enable-native-access=ALL-UNNAMED'

import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('QtAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, 
    QGroupBox, QFileDialog, QTableWidget, QTableWidgetItem, 
    QTabWidget,  QProgressBar, QStatusBar, QToolBar, QMessageBox, 
    QScrollArea, QFormLayout, QTextEdit, QSlider
)
from PySide6.QtCore import Qt, Signal,  Slot, QObject
from PySide6.QtGui import QAction

from ca_analyzer import CalciumAnalyzer, AnalysisConfig
from plot_style import apply_calcium_style
from config_manager import ConfigManager


class AnalysisWorker(QObject):
    """Worker thread for running calcium analysis without blocking the UI."""
    
    progress = Signal(str, int)  # message, percentage
    finished = Signal(bool, str)  # success, message
    results_ready = Signal(object)  # analyzer object
    
    def __init__(self, filepath: str, config: AnalysisConfig,
                 cell_limits: Optional[Tuple[float, float]] = None,
                 baseline_limits: Optional[Tuple[float, float]] = None):
        super().__init__()
        self.filepath = filepath
        self.config = config
        self.cell_limits = cell_limits
        self.baseline_limits = baseline_limits
        
    def run(self):
        """Execute the analysis."""
        try:
            self.progress.emit("Loading image...", 10)
            QApplication.processEvents()
            analyzer = CalciumAnalyzer(self.config)
            
            self.progress.emit("Preprocessing image...", 30)
            QApplication.processEvents()
            analyzer.load_image(self.filepath)
            
            self.progress.emit("Selecting ROIs...", 50)
            QApplication.processEvents()
            analyzer.select_rois(self.cell_limits, self.baseline_limits)
            
            self.progress.emit("Analyzing transients...", 70)
            QApplication.processEvents()
            analyzer.find_and_analyze_transients()
            
            self.progress.emit("Exporting results...", 90)
            QApplication.processEvents()
            analyzer.export_results()
            
            self.progress.emit("Complete!", 100)
            QApplication.processEvents()
            self.results_ready.emit(analyzer)
            self.finished.emit(True, f"Analysis complete! Found {len(analyzer.transients)} transient(s)")
            QApplication.processEvents()
            
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")
        finally:
            # Close any matplotlib figures created in this thread
            plt.close('all')
            QApplication.processEvents()

class MplCanvas(FigureCanvas):
    """Matplotlib canvas widget for embedding plots in Qt."""
    
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        

class ROISelector(MplCanvas):
    """Interactive ROI selector with matplotlib and visual feedback."""
    
    roi_selected = Signal(tuple)  # (start, end)
    
    def __init__(self, parent=None, selection_type='horizontal'):
        super().__init__(parent, width=10, height=5)
        self.selection_type = selection_type  # 'horizontal' or 'vertical'
        self.selector = None
        self.data = None
        self.preview_lines = []  # Red preview lines
        self.confirmed_lines = []  # Green confirmed lines
        
    def set_data(self, data: np.ndarray, title: str = ""):
        """Display data and enable ROI selection."""
        self.data = data
        self.axes.clear()
        self.preview_lines = []
        self.confirmed_lines = []
        
        if data.ndim == 1:
            self.axes.plot(data)
        else:
            self.axes.imshow(data, cmap='inferno', aspect='auto',
                           vmin=0, vmax=np.mean(data) * 10)
        
        self.axes.set_title(title)
        apply_calcium_style()
        self.draw()
        
    def draw_preview_lines(self, start, end):
        """Draw red preview lines for ROI."""
        # Remove old preview lines
        for line in self.preview_lines:
            line.remove()
        self.preview_lines = []
        
        if self.data is None:
            return
            
        if self.selection_type == 'horizontal':
            # Vertical lines for horizontal selection
            line1 = self.axes.axvline(start, color='red', linestyle='--', linewidth=2, alpha=0.7)
            line2 = self.axes.axvline(end, color='red', linestyle='--', linewidth=2, alpha=0.7)
        else:
            # Horizontal lines for vertical selection
            line1 = self.axes.axhline(start, color='red', linestyle='--', linewidth=2, alpha=0.7)
            line2 = self.axes.axhline(end, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
        self.preview_lines = [line1, line2]
        self.draw()
        
    def confirm_selection(self, start, end):
        """Convert preview lines to green confirmed lines."""
        # Remove preview lines
        for line in self.preview_lines:
            line.remove()
        self.preview_lines = []
        
        # Remove old confirmed lines
        for line in self.confirmed_lines:
            line.remove()
        self.confirmed_lines = []
        
        if self.data is None:
            return
            
        if self.selection_type == 'horizontal':
            # Vertical lines for horizontal selection
            line1 = self.axes.axvline(start, color='green', linestyle='-', linewidth=2, alpha=0.8)
            line2 = self.axes.axvline(end, color='green', linestyle='-', linewidth=2, alpha=0.8)
        else:
            # Horizontal lines for vertical selection
            line1 = self.axes.axhline(start, color='green', linestyle='-', linewidth=2, alpha=0.8)
            line2 = self.axes.axhline(end, color='green', linestyle='-', linewidth=2, alpha=0.8)
            
        self.confirmed_lines = [line1, line2]
        self.draw()
        
    def clear_lines(self):
        """Clear all ROI lines."""
        for line in self.preview_lines + self.confirmed_lines:
            line.remove()
        self.preview_lines = []
        self.confirmed_lines = []
        self.draw()
        
    def enable_selection(self):
        """Enable interactive ROI selection."""
        from matplotlib.widgets import SpanSelector
        
        if self.selection_type == 'horizontal':
            direction = 'horizontal'
        else:
            direction = 'vertical'
            
        self.selector = SpanSelector(
            self.axes,
            self.on_select,
            direction,
            useblit=True,
            props=dict(alpha=0.3, facecolor='red'),
            interactive=True,
            drag_from_anywhere=True
        )
        
    def on_select(self, xmin, xmax):
        """Handle ROI selection."""
        self.roi_selected.emit((xmin, xmax))


class ConfigPanel(QWidget):
    """Configuration panel for analysis parameters."""
    
    config_changed = Signal(object)  # AnalysisConfig
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        
        # Image Parameters
        img_group = QGroupBox("Image Parameters")
        img_layout = QFormLayout()
        
        self.fluo_index = QSpinBox()
        self.fluo_index.setRange(0, 10)
        self.fluo_index.setValue(0)
        img_layout.addRow("Fluorescence Channel:", self.fluo_index)
        
        self.num_index = QSpinBox()
        self.num_index.setRange(0, 10)
        self.num_index.setValue(0)
        img_layout.addRow("Numerator Channel:", self.num_index)
        
        self.den_index = QSpinBox()
        self.den_index.setRange(0, 10)
        self.den_index.setValue(1)
        img_layout.addRow("Denominator Channel:", self.den_index)
        
        self.linescan_speed = QDoubleSpinBox()
        self.linescan_speed.setRange(0.1, 100.0)
        self.linescan_speed.setValue(1.87)
        self.linescan_speed.setDecimals(2)
        self.linescan_speed.setSuffix(" ms/line")
        img_layout.addRow("Linescan Speed:", self.linescan_speed)
        
        self.filter_kernel = QSpinBox()
        self.filter_kernel.setRange(3, 15)
        self.filter_kernel.setValue(5)
        self.filter_kernel.setSingleStep(2)
        img_layout.addRow("Filter Kernel Size:", self.filter_kernel)
        
        img_group.setLayout(img_layout)
        layout.addWidget(img_group)
        
        # Analysis Parameters
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QFormLayout()
        
        self.mode = QComboBox()
        self.mode.addItems(["single", "ratio"])
        analysis_layout.addRow("Mode:", self.mode)
        
        self.analyze_synchrony = QCheckBox()
        self.analyze_synchrony.setChecked(True)
        analysis_layout.addRow("Analyze Synchrony:", self.analyze_synchrony)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Peak Detection
        peak_group = QGroupBox("Peak Detection")
        peak_layout = QFormLayout()
        
        self.peak_prominence = QDoubleSpinBox()
        self.peak_prominence.setRange(0.1, 1.0)
        self.peak_prominence.setValue(0.45)
        self.peak_prominence.setDecimals(2)
        self.peak_prominence.setSingleStep(0.05)
        peak_layout.addRow("Prominence Ratio:", self.peak_prominence)
        
        self.min_peak_distance = QSpinBox()
        self.min_peak_distance.setRange(10, 1000)
        self.min_peak_distance.setValue(200)
        peak_layout.addRow("Min Peak Distance:", self.min_peak_distance)
        
        peak_group.setLayout(peak_layout)
        layout.addWidget(peak_group)
        
        # Visualization
        viz_group = QGroupBox("Visualization")
        viz_layout = QFormLayout()
        
        self.show_images = QCheckBox()
        self.show_images.setChecked(True)
        viz_layout.addRow("Show Images:", self.show_images)
        
        self.max_ff0 = QDoubleSpinBox()
        self.max_ff0.setRange(1.0, 20.0)
        self.max_ff0.setValue(8.0)
        self.max_ff0.setDecimals(1)
        viz_layout.addRow("Max F/F0:", self.max_ff0)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Export
        export_group = QGroupBox("Export")
        export_layout = QFormLayout()
        
        self.export_csv = QCheckBox()
        self.export_csv.setChecked(True)
        export_layout.addRow("Export CSV:", self.export_csv)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Apply button
        apply_btn = QPushButton("Apply Configuration")
        apply_btn.clicked.connect(self.emit_config)
        layout.addWidget(apply_btn)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Connect signals for auto-update
        self.fluo_index.valueChanged.connect(self.emit_config)
        self.linescan_speed.valueChanged.connect(self.emit_config)
        self.filter_kernel.valueChanged.connect(self.emit_config)
        self.num_index.valueChanged.connect(self.emit_config)
        self.den_index.valueChanged.connect(self.emit_config)
        self.mode.currentTextChanged.connect(self.emit_config)
        self.analyze_synchrony.stateChanged.connect(self.emit_config)
        self.peak_prominence.valueChanged.connect(self.emit_config)
        self.min_peak_distance.valueChanged.connect(self.emit_config)
        self.max_ff0.valueChanged.connect(self.emit_config)
        
    def emit_config(self):
        """Emit the current configuration."""
        config = AnalysisConfig(
            fluo_index=self.fluo_index.value(),
            numerator_index=self.num_index.value(),
            denominator_index=self.den_index.value(),
            linescan_speed=self.linescan_speed.value(),
            filter_kernelsize=self.filter_kernel.value(),
            mode=self.mode.currentText(),
            analyze_synchrony=self.analyze_synchrony.isChecked(),
            show_images=self.show_images.isChecked(),
            max_ff0=self.max_ff0.value(),
            export_csv=self.export_csv.isChecked(),
            peak_prominence_ratio=self.peak_prominence.value(),
            min_peak_distance=self.min_peak_distance.value()
        )
        self.config_changed.emit(config)
        
    def get_config(self) -> AnalysisConfig:
        """Get the current configuration."""
        return AnalysisConfig(
            fluo_index=self.fluo_index.value(),
            numerator_index=self.num_index.value(),
            denominator_index=self.den_index.value(),
            linescan_speed=self.linescan_speed.value(),
            filter_kernelsize=self.filter_kernel.value(),
            mode=self.mode.currentText(),
            analyze_synchrony=self.analyze_synchrony.isChecked(),
            show_images=self.show_images.isChecked(),
            max_ff0=self.max_ff0.value(),
            export_csv=self.export_csv.isChecked(),
            peak_prominence_ratio=self.peak_prominence.value(),
            min_peak_distance=self.min_peak_distance.value()
        )
    
    def set_config(self, config: AnalysisConfig):
        """Set the configuration values in the UI widgets."""
        # Temporarily block signals to avoid triggering config_changed
        self.blockSignals(True)
        
        self.fluo_index.setValue(config.fluo_index)
        self.num_index.setValue(config.numerator_index)
        self.den_index.setValue(config.denominator_index)
        self.linescan_speed.setValue(config.linescan_speed)
        self.filter_kernel.setValue(config.filter_kernelsize)
        self.mode.setCurrentText(config.mode)
        self.analyze_synchrony.setChecked(config.analyze_synchrony)
        self.show_images.setChecked(config.show_images)
        self.max_ff0.setValue(config.max_ff0)
        self.export_csv.setChecked(config.export_csv)
        self.peak_prominence.setValue(config.peak_prominence_ratio)
        self.min_peak_distance.setValue(config.min_peak_distance)
        
        self.blockSignals(False)
        
        # Emit the configuration once after all values are set
        self.emit_config()


class ResultsPanel(QWidget):
    """Panel for displaying analysis results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.analyzer = None
        self.linescan_colorbar = None  # Track colorbar to prevent accumulation
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        
        # Create tab widget for different views
        self.tabs = QTabWidget()
        
        # Signal plot tab
        self.signal_canvas = MplCanvas(self, width=10, height=5)
        signal_widget = QWidget()
        signal_layout = QVBoxLayout()
        signal_layout.addWidget(NavigationToolbar(self.signal_canvas, self))
        signal_layout.addWidget(self.signal_canvas)
        signal_widget.setLayout(signal_layout)
        self.tabs.addTab(signal_widget, "Signal Plot")
        
        # Linescan image tab
        self.linescan_canvas = MplCanvas(self, width=10, height=5)
        linescan_widget = QWidget()
        linescan_layout = QVBoxLayout()
        linescan_layout.addWidget(NavigationToolbar(self.linescan_canvas, self))
        linescan_layout.addWidget(self.linescan_canvas)
        linescan_widget.setLayout(linescan_layout)
        self.tabs.addTab(linescan_widget, "Linescan Image")
        
        # Synchrony analysis tab
        self.synchrony_canvas = MplCanvas(self, width=10, height=5)
        synchrony_widget = QWidget()
        synchrony_layout = QVBoxLayout()
        synchrony_layout.addWidget(NavigationToolbar(self.synchrony_canvas, self))
        synchrony_layout.addWidget(self.synchrony_canvas)
        synchrony_widget.setLayout(synchrony_layout)
        self.tabs.addTab(synchrony_widget, "Synchrony Analysis")
        
        # Statistics table tab
        self.stats_table = QTableWidget()
        self.tabs.addTab(self.stats_table, "Statistics Table")
        
        # Summary tab
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.tabs.addTab(self.summary_text, "Summary")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def cleanup(self):
        """Clean up all matplotlib canvases."""
        try:
            self.signal_canvas.cleanup()
            self.linescan_canvas.cleanup()
            self.synchrony_canvas.cleanup()
        except Exception:
            pass
        
    def display_results(self, analyzer: CalciumAnalyzer):
        """Display analysis results."""
        self.analyzer = analyzer
        
        if analyzer.image is None or analyzer.results is None:
            return
            
        # Plot signal
        self.plot_signal()
        
        # Plot linescan
        self.plot_linescan()
        
        # Plot synchrony if enabled
        if analyzer.config.analyze_synchrony:
            self.plot_synchrony()
            self.tabs.setTabVisible(2, True)
        else:
            self.tabs.setTabVisible(2, False)
        
        # Update statistics table
        self.update_stats_table()
        
        # Update summary
        self.update_summary()
        
    def plot_signal(self):
        """Plot the F/F0 signal with detected transients."""
        if self.analyzer is None:
            return
            
        ax = self.signal_canvas.axes
        ax.clear()
        
        time_axis = self.analyzer.image.get_time_axis()
        signal = self.analyzer.image.signal_1d
        
        ax.plot(time_axis, signal, 'b-', linewidth=1.5)
        
        # Mark transient boundaries
        valid_rows = ~np.isnan(self.analyzer.results[:, 0])
        if np.any(valid_rows):
            begin_indices = self.analyzer.results[valid_rows, 0].astype(int)
            end_indices = self.analyzer.results[valid_rows, 1].astype(int)
            
            ax.plot(time_axis[begin_indices], signal[begin_indices],
                   'ro', label='Begin', markersize=8, fillstyle='none')
            ax.plot(time_axis[end_indices], signal[end_indices],
                   'go', label='End', markersize=8, fillstyle='none')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Ca Signal (F/F0)')
        ax.set_title('Calcium Transient Signal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        apply_calcium_style()
        self.signal_canvas.draw()
        
    def plot_linescan(self):
        """Plot the normalized linescan image."""
        if self.analyzer is None:
            return
            
        # Clear the entire figure to remove old colorbar
        self.linescan_canvas.fig.clear()
        ax = self.linescan_canvas.fig.add_subplot(111)
        self.linescan_canvas.axes = ax
        
        time_axis = self.analyzer.image.get_time_axis()
        normalized_data = self.analyzer.image.normalized_data
        
        extent = [0, self.analyzer.config.linescan_speed * len(self.analyzer.image.signal_1d),
                 0, self.analyzer.image.voxel_size * normalized_data.shape[0]]
        
        im = ax.imshow(normalized_data, cmap='inferno',
                      vmin=0, vmax=self.analyzer.config.max_ff0,
                      aspect='auto', extent=extent)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Distance (μm)')
        ax.set_title('Normalized Linescan Image')
        
        # Create colorbar
        self.linescan_colorbar = self.linescan_canvas.fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.6)
        self.linescan_colorbar.set_label('F/F0')
        apply_calcium_style()
        self.linescan_canvas.draw()
        
    def plot_synchrony(self):
        """Plot synchrony analysis results - delay times for each transient."""
        if self.analyzer is None or not self.analyzer.config.analyze_synchrony:
            return
            
        ax = self.synchrony_canvas.axes
        ax.clear()
        
        # Check if we have transients with delay_times
        if not self.analyzer.transients:
            ax.text(0.5, 0.5, 'No synchrony data available', 
                   ha='center', va='center', transform=ax.transAxes)
            self.synchrony_canvas.draw()
            return
        
        # Get space axis for y-axis
        space_axis = self.analyzer.image.get_space_axis()
        
        # Plot delay times for each transient
        for i, transient in enumerate(self.analyzer.transients):
            if transient.delay_times is not None:
                # Filter out NaN values for plotting
                valid_mask = ~np.isnan(transient.delay_times)
                if np.any(valid_mask):
                    ax.plot(transient.delay_times[valid_mask], 
                           space_axis[valid_mask],
                           marker='o', linestyle='-', alpha=0.7,
                           label=f'Transient #{i+1}')
        
        ax.set_xlabel('Delay Time (ms)')
        ax.set_ylabel('Distance (μm)')
        ax.set_title('Spatial Synchrony Analysis - Activation Delays')
        ax.legend()
        ax.grid(True, alpha=0.3)
        apply_calcium_style()
        self.synchrony_canvas.draw()
        
    def update_stats_table(self):
        """Update the statistics table."""
        if self.analyzer is None or self.analyzer.results is None:
            return
            
        from ca_analyzer import TransientMetrics
        
        headers = TransientMetrics.get_header(self.analyzer.config.analyze_synchrony)
        results = self.analyzer.results
        
        # Filter valid results
        valid_rows = ~np.isnan(results[:, 0])
        valid_results = results[valid_rows]
        
        self.stats_table.setRowCount(len(valid_results))
        self.stats_table.setColumnCount(len(headers))
        self.stats_table.setHorizontalHeaderLabels(headers)
        
        for i, row in enumerate(valid_results):
            for j, value in enumerate(row):
                item = QTableWidgetItem(f"{value:.3f}")
                self.stats_table.setItem(i, j, item)
        
        self.stats_table.resizeColumnsToContents()
        
    def update_summary(self):
        """Update the summary text."""
        if self.analyzer is None or self.analyzer.results is None:
            return
            
        valid_results = self.analyzer.results[~np.isnan(self.analyzer.results[:, 0])]
        n_transients = len(valid_results)
        
        summary = f"<h2>Analysis Summary</h2>\n"
        summary += f"<p><b>File:</b> {self.analyzer.image.filepath.name}</p>\n"
        summary += f"<p><b>Number of transients:</b> {n_transients}</p>\n"
        
        if n_transients > 0:
            summary += "<h3>Statistics</h3>\n"
            summary += f"<p><b>Mean amplitude:</b> {np.nanmean(valid_results[:, 5]):.3f}</p>\n"
            summary += f"<p><b>Mean rise time (10-90%):</b> {np.nanmean(valid_results[:, 6]):.2f} ms</p>\n"
            summary += f"<p><b>Mean decay time (50%):</b> {np.nanmean(valid_results[:, 7]):.2f} ms</p>\n"
            summary += f"<p><b>Mean decay time (90%):</b> {np.nanmean(valid_results[:, 8]):.2f} ms</p>\n"
            
            if self.analyzer.config.analyze_synchrony and valid_results.shape[1] > 9:
                summary += "<h3>Synchrony Metrics</h3>\n"
                summary += f"<p><b>Mean delay:</b> {np.nanmean(valid_results[:, 9]):.2f} ms</p>\n"
                summary += f"<p><b>Delay SD:</b> {np.nanmean(valid_results[:, 10]):.2f} ms</p>\n"
                summary += f"<p><b>Synchrony Index:</b> {np.nanmean(valid_results[:, 11]):.3f}</p>\n"
        
        self.summary_text.setHtml(summary)


class CalciumAnalysisApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.current_file = None
        
        # Initialize configuration manager and load config
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        
        self.cell_limits = None
        self.baseline_limits = None
        self.worker = None
        self.current_image_data = None  # Store current image for ROI visualization
        self.cell_roi_lines = []  # Track cell ROI lines
        self.baseline_roi_lines = []  # Track baseline ROI lines
        
        self.init_ui()
        
        # Load configuration into UI after UI is initialized
        self.config_panel.set_config(self.config)
        
        # Update status bar with config file path
        self.status_bar.showMessage(f"Configuration loaded from: {self.config_manager.get_config_path()}")
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Calcium Transient Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left panel: Configuration
        self.config_panel = ConfigPanel()
        self.config_panel.config_changed.connect(self.on_config_changed)
        config_scroll = QScrollArea()
        config_scroll.setWidget(self.config_panel)
        config_scroll.setWidgetResizable(True)
        config_scroll.setMaximumWidth(350)
        
        # Center/Right: Tabs for different views
        self.tabs = QTabWidget()
        
        # Image viewer tab
        image_tab = QWidget()
        image_layout = QVBoxLayout()
        
        # ROI selection controls
        roi_controls = QWidget()
        roi_layout = QVBoxLayout()
        
        # Cell ROI Group
        cell_roi_group = QGroupBox("Cell ROI (Y-axis / Vertical)")
        cell_roi_layout = QFormLayout()
        
        # Y Start controls
        y_start_layout = QHBoxLayout()
        self.y_start_slider = QSlider(Qt.Horizontal)
        self.y_start_slider.setMinimum(0)
        self.y_start_slider.setMaximum(1000)
        self.y_start_slider.setValue(0)
        self.y_start_slider.valueChanged.connect(self.on_cell_roi_preview)
        
        self.y_start_spinbox = QSpinBox()
        self.y_start_spinbox.setMinimum(0)
        self.y_start_spinbox.setMaximum(10000)
        self.y_start_spinbox.setValue(0)
        self.y_start_spinbox.valueChanged.connect(self.y_start_slider.setValue)
        self.y_start_slider.valueChanged.connect(self.y_start_spinbox.setValue)
        
        y_start_layout.addWidget(self.y_start_slider, stretch=3)
        y_start_layout.addWidget(self.y_start_spinbox, stretch=1)
        cell_roi_layout.addRow("Y Start:", y_start_layout)
        
        # Y End controls
        y_end_layout = QHBoxLayout()
        self.y_end_slider = QSlider(Qt.Horizontal)
        self.y_end_slider.setMinimum(0)
        self.y_end_slider.setMaximum(1000)
        self.y_end_slider.setValue(100)
        self.y_end_slider.valueChanged.connect(self.on_cell_roi_preview)
        
        self.y_end_spinbox = QSpinBox()
        self.y_end_spinbox.setMinimum(0)
        self.y_end_spinbox.setMaximum(10000)
        self.y_end_spinbox.setValue(100)
        self.y_end_spinbox.valueChanged.connect(self.y_end_slider.setValue)
        self.y_end_slider.valueChanged.connect(self.y_end_spinbox.setValue)
        
        y_end_layout.addWidget(self.y_end_slider, stretch=3)
        y_end_layout.addWidget(self.y_end_spinbox, stretch=1)
        cell_roi_layout.addRow("Y End:", y_end_layout)
        
        # Apply button for cell ROI
        self.apply_cell_roi_btn = QPushButton("Apply Cell ROI")
        self.apply_cell_roi_btn.clicked.connect(self.apply_cell_roi)
        self.apply_cell_roi_btn.setEnabled(False)
        self.apply_cell_roi_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        cell_roi_layout.addRow(self.apply_cell_roi_btn)
        
        cell_roi_group.setLayout(cell_roi_layout)
        roi_layout.addWidget(cell_roi_group)
        
        # Baseline ROI Group
        baseline_roi_group = QGroupBox("Baseline ROI (X-axis / Horizontal)")
        baseline_roi_layout = QFormLayout()
        
        # X Start controls
        x_start_layout = QHBoxLayout()
        self.x_start_slider = QSlider(Qt.Horizontal)
        self.x_start_slider.setMinimum(0)
        self.x_start_slider.setMaximum(5000)
        self.x_start_slider.setValue(0)
        self.x_start_slider.valueChanged.connect(self.on_baseline_roi_preview)
        
        self.x_start_spinbox = QSpinBox()
        self.x_start_spinbox.setMinimum(0)
        self.x_start_spinbox.setMaximum(50000)
        self.x_start_spinbox.setValue(0)
        self.x_start_spinbox.valueChanged.connect(self.x_start_slider.setValue)
        self.x_start_slider.valueChanged.connect(self.x_start_spinbox.setValue)
        
        x_start_layout.addWidget(self.x_start_slider, stretch=3)
        x_start_layout.addWidget(self.x_start_spinbox, stretch=1)
        baseline_roi_layout.addRow("X Start:", x_start_layout)
        
        # X End controls
        x_end_layout = QHBoxLayout()
        self.x_end_slider = QSlider(Qt.Horizontal)
        self.x_end_slider.setMinimum(0)
        self.x_end_slider.setMaximum(5000)
        self.x_end_slider.setValue(100)
        self.x_end_slider.valueChanged.connect(self.on_baseline_roi_preview)
        
        self.x_end_spinbox = QSpinBox()
        self.x_end_spinbox.setMinimum(0)
        self.x_end_spinbox.setMaximum(50000)
        self.x_end_spinbox.setValue(100)
        self.x_end_spinbox.valueChanged.connect(self.x_end_slider.setValue)
        self.x_end_slider.valueChanged.connect(self.x_end_spinbox.setValue)
        
        x_end_layout.addWidget(self.x_end_slider, stretch=3)
        x_end_layout.addWidget(self.x_end_spinbox, stretch=1)
        baseline_roi_layout.addRow("X End:", x_end_layout)
        
        # Apply button for baseline ROI
        self.apply_baseline_roi_btn = QPushButton("Apply Baseline ROI")
        self.apply_baseline_roi_btn.clicked.connect(self.apply_baseline_roi)
        self.apply_baseline_roi_btn.setEnabled(False)
        self.apply_baseline_roi_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        baseline_roi_layout.addRow(self.apply_baseline_roi_btn)
        
        baseline_roi_group.setLayout(baseline_roi_layout)
        roi_layout.addWidget(baseline_roi_group)
        
        roi_controls.setLayout(roi_layout)
        image_layout.addWidget(roi_controls)
        
        # Image canvas
        self.image_canvas = MplCanvas(self, width=10, height=6)
        image_layout.addWidget(NavigationToolbar(self.image_canvas, self))
        image_layout.addWidget(self.image_canvas)
        
        image_tab.setLayout(image_layout)
        self.tabs.addTab(image_tab, "Image Viewer")
        
        # Results tab
        self.results_panel = ResultsPanel()
        self.tabs.addTab(self.results_panel, "Results")
        
        # Add to main layout
        main_layout.addWidget(config_scroll)
        main_layout.addWidget(self.tabs, stretch=1)
        
        central_widget.setLayout(main_layout)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        self.status_bar.showMessage("Ready")
        
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("&Export Results...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Configuration menu
        config_menu = menubar.addMenu("&Configuration")
        
        save_config_action = QAction("&Save Configuration", self)
        save_config_action.setShortcut("Ctrl+S")
        save_config_action.triggered.connect(self.save_configuration)
        config_menu.addAction(save_config_action)
        
        load_config_action = QAction("&Load Configuration", self)
        load_config_action.setShortcut("Ctrl+L")
        load_config_action.triggered.connect(self.load_configuration)
        config_menu.addAction(load_config_action)
        
        config_menu.addSeparator()
        
        reset_config_action = QAction("&Reset to Defaults", self)
        reset_config_action.triggered.connect(self.reset_configuration)
        config_menu.addAction(reset_config_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_toolbar(self):
        """Create the toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        open_btn = QPushButton("Open File")
        open_btn.clicked.connect(self.open_file)
        toolbar.addWidget(open_btn)
        
        toolbar.addSeparator()
        
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)
        toolbar.addWidget(self.run_btn)
        
    def on_config_changed(self, config: AnalysisConfig):
        """Handle configuration changes."""
        self.config = config
        
    def open_file(self):
        """Open a Image file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "TIFF Images (*.tif *.tiff);;Nikon Images (*.nd2);;Zeiss Images (*.czi *.lsm);;All Files (*.*)"
        )
        
        if filename:
            self.current_file = filename
            self.load_image()
            
    def load_image(self):
        """Load and display the current image."""
        if not self.current_file:
            return
            
        try:
            self.status_bar.showMessage(f"Loading {Path(self.current_file).name}...")
            
            # Load image using analyzer
            from ca_analyzer import CalciumImage
            img = CalciumImage(self.current_file, self.config)
            img.apply_median_filter()
            
            # Store image data for ROI visualization
            self.current_image_data = img.filtered_data
            
            # Update slider ranges based on image dimensions
            height, width = img.filtered_data.shape
            
            # Update Y sliders (cell ROI)
            self.y_start_slider.setMaximum(height - 1)
            self.y_start_spinbox.setMaximum(height - 1)
            self.y_end_slider.setMaximum(height - 1)
            self.y_end_spinbox.setMaximum(height - 1)
            self.y_end_slider.setValue(min(100, height - 1))
            self.y_end_spinbox.setValue(min(100, height - 1))
            
            # Update X sliders (baseline ROI)
            self.x_start_slider.setMaximum(width - 1)
            self.x_start_spinbox.setMaximum(width - 1)
            self.x_end_slider.setMaximum(width - 1)
            self.x_end_spinbox.setMaximum(width - 1)
            self.x_end_slider.setValue(min(100, width - 1))
            self.x_end_spinbox.setValue(min(100, width - 1))
            
            # Display filtered image
            ax = self.image_canvas.axes
            ax.clear()
            self.cell_roi_lines = []
            self.baseline_roi_lines = []
            
            ax.imshow(img.filtered_data, cmap='inferno', aspect='auto',
                     vmin=0, vmax=np.mean(img.filtered_data) * 10)
            ax.set_title(f"Filtered Image: {Path(self.current_file).name}")
            ax.set_xlabel("Time (frames)")
            ax.set_ylabel("Space (pixels)")
            apply_calcium_style()
            self.image_canvas.draw()
            
            # Enable ROI selection controls
            self.apply_cell_roi_btn.setEnabled(True)
            self.apply_baseline_roi_btn.setEnabled(True)
            self.status_bar.showMessage(f"Loaded {Path(self.current_file).name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            self.status_bar.showMessage("Error loading image")
            
    def on_cell_roi_preview(self):
        """Preview cell ROI with red lines."""
        if self.current_image_data is None:
            return
            
        # Remove old preview lines
        for line in self.cell_roi_lines:
            if line in self.image_canvas.axes.lines:
                line.remove()
        self.cell_roi_lines = []
        
        # Draw red preview lines
        y_start = self.y_start_spinbox.value()
        y_end = self.y_end_spinbox.value()
        
        line1 = self.image_canvas.axes.axhline(y_start, color='red', linestyle='--', linewidth=2, alpha=0.7)
        line2 = self.image_canvas.axes.axhline(y_end, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        self.cell_roi_lines = [line1, line2]
        self.image_canvas.draw()
        
    def apply_cell_roi(self):
        """Apply cell ROI and turn lines green."""
        if self.current_image_data is None:
            return
            
        y_start = self.y_start_spinbox.value()
        y_end = self.y_end_spinbox.value()
        
        if y_start >= y_end:
            QMessageBox.warning(self, "Invalid ROI", "Y Start must be less than Y End")
            return
            
        # Remove old lines
        for line in self.cell_roi_lines:
            if line in self.image_canvas.axes.lines:
                line.remove()
        self.cell_roi_lines = []
        
        # Draw green confirmed lines
        line1 = self.image_canvas.axes.axhline(y_start, color='green', linestyle='-', linewidth=2, alpha=0.8)
        line2 = self.image_canvas.axes.axhline(y_end, color='green', linestyle='-', linewidth=2, alpha=0.8)
        
        self.cell_roi_lines = [line1, line2]
        self.image_canvas.draw()
        
        # Store the limits
        self.cell_limits = (y_start, y_end)
        self.status_bar.showMessage(f"Cell ROI applied: Y {y_start} - {y_end}")
        
        # Enable run button if both ROIs are set
        if self.baseline_limits is not None:
            self.run_btn.setEnabled(True)
            
    def on_baseline_roi_preview(self):
        """Preview baseline ROI with red lines."""
        if self.current_image_data is None:
            return
            
        # Remove old preview lines
        for line in self.baseline_roi_lines:
            if line in self.image_canvas.axes.lines:
                line.remove()
        self.baseline_roi_lines = []
        
        # Draw red preview lines
        x_start = self.x_start_spinbox.value()
        x_end = self.x_end_spinbox.value()
        
        line1 = self.image_canvas.axes.axvline(x_start, color='red', linestyle='--', linewidth=2, alpha=0.7)
        line2 = self.image_canvas.axes.axvline(x_end, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        self.baseline_roi_lines = [line1, line2]
        self.image_canvas.draw()
        
    def apply_baseline_roi(self):
        """Apply baseline ROI and turn lines green."""
        if self.current_image_data is None:
            return
            
        x_start = self.x_start_spinbox.value()
        x_end = self.x_end_spinbox.value()
        
        if x_start >= x_end:
            QMessageBox.warning(self, "Invalid ROI", "X Start must be less than X End")
            return
            
        # Remove old lines
        for line in self.baseline_roi_lines:
            if line in self.image_canvas.axes.lines:
                line.remove()
        self.baseline_roi_lines = []
        
        # Draw green confirmed lines
        line1 = self.image_canvas.axes.axvline(x_start, color='green', linestyle='-', linewidth=2, alpha=0.8)
        line2 = self.image_canvas.axes.axvline(x_end, color='green', linestyle='-', linewidth=2, alpha=0.8)
        
        self.baseline_roi_lines = [line1, line2]
        self.image_canvas.draw()
        
        # Store the limits
        self.baseline_limits = (x_start, x_end)
        self.status_bar.showMessage(f"Baseline ROI applied: X {x_start} - {x_end}")
        
        # Enable run button if both ROIs are set
        if self.cell_limits is not None:
            self.run_btn.setEnabled(True)
        
    def run_analysis(self):
        """Run the calcium transient analysis."""
        if not self.current_file:
            QMessageBox.warning(self, "Warning", "Please open a file first")
            return
            
        if self.cell_limits is None or self.baseline_limits is None:
            QMessageBox.warning(self, "Warning", "Please select ROIs first")
            return
            
        # Disable controls
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Get config and force show_images=False to prevent matplotlib plots in worker thread
        config = self.config_panel.get_config()
        # Create a new config with show_images disabled for thread safety
        worker_config = AnalysisConfig(
            fluo_index=config.fluo_index,
            linescan_speed=config.linescan_speed,
            filter_kernelsize=config.filter_kernelsize,
            mode=config.mode,
            analyze_synchrony=config.analyze_synchrony,
            show_images=False,  # Force False to prevent Qt timer issues in worker thread
            max_ff0=config.max_ff0,
            export_csv=config.export_csv,
            peak_prominence_ratio=config.peak_prominence_ratio,
            min_peak_distance=config.min_peak_distance
        )
        
        # Create and start worker thread
        self.worker = AnalysisWorker(
            self.current_file,
            worker_config,
            self.cell_limits,
            self.baseline_limits
        )
        
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.results_ready.connect(self.on_results_ready)
        
        # Run synchronously (UI still updates due to processEvents in worker)
        self.worker.run()
        
    @Slot(str, int)
    def on_progress(self, message: str, percentage: int):
        """Update progress bar."""
        self.status_bar.showMessage(message)
        self.progress_bar.setValue(percentage)
        
    @Slot(bool, str)
    def on_analysis_finished(self, success: bool, message: str):
        """Handle analysis completion."""
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        
        if success:
            self.status_bar.showMessage(message)
            self.tabs.setCurrentIndex(1)  # Switch to results tab
        else:
            QMessageBox.critical(self, "Analysis Error", message)
            self.status_bar.showMessage("Analysis failed")
            
    @Slot(object)
    def on_results_ready(self, analyzer: CalciumAnalyzer):
        """Display analysis results."""
        self.results_panel.display_results(analyzer)
        
    def export_results(self):
        """Export analysis results."""
        QMessageBox.information(
            self,
            "Export",
            "Results are automatically exported to CSV when analysis completes.\n"
            "Check the same directory as your input file."
        )
    
    def save_configuration(self):
        """Save current configuration to JSON file."""
        config = self.config_panel.get_config()
        success = self.config_manager.save_config(config)
        
        if success:
            QMessageBox.information(
                self,
                "Configuration Saved",
                f"Configuration saved successfully to:\n{self.config_manager.get_config_path()}"
            )
            self.status_bar.showMessage(f"Configuration saved to: {self.config_manager.get_config_path()}", 5000)
        else:
            QMessageBox.critical(
                self,
                "Save Error",
                "Failed to save configuration. Check console for details."
            )
    
    def load_configuration(self):
        """Load configuration from JSON file."""
        config = self.config_manager.load_config()
        self.config = config
        self.config_panel.set_config(config)
        
        QMessageBox.information(
            self,
            "Configuration Loaded",
            f"Configuration loaded from:\n{self.config_manager.get_config_path()}"
        )
        self.status_bar.showMessage(f"Configuration loaded from: {self.config_manager.get_config_path()}", 5000)
    
    def reset_configuration(self):
        """Reset configuration to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Configuration",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            config = self.config_manager.reset_to_defaults()
            self.config = config
            self.config_panel.set_config(config)
            
            QMessageBox.information(
                self,
                "Configuration Reset",
                "Configuration has been reset to defaults."
            )
            self.status_bar.showMessage("Configuration reset to defaults", 5000)
        
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Calcium Transient Analysis",
            "<h2>Calcium Transient Analysis</h2>"
            "<p>Version 2.0</p>"
            "<p>A tool for analyzing calcium transients from linescan microscopy images.</p>"
            "<p>Author: hjoca</p>"
            "<p>Date: 2026-02-11</p>"
        )
    
    def closeEvent(self, event):
        """Handle window close event - clean up matplotlib resources."""
        # Close all remaining matplotlib figures
        plt.close('all')
        
        # Process any pending events to ensure cleanup completes
        QApplication.processEvents()
        
        # Accept the close event
        event.accept()

def main():
    """Main entry point."""
    # Create application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Ensure app quits when last window is closed
    app.setQuitOnLastWindowClosed(True)
    
    window = CalciumAnalysisApp()
    window.show()
    
    # Run event loop and exit cleanly
    exit_code = app.exec()
    
    # Close all matplotlib figures before exit
    plt.close('all')
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
