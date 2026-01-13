"""
Calcium Transient Analysis Module

This module provides the framework for analyzing calcium transients
from linescan microscopy images.

Author: hjoca
Refactored: 2026-01-13
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import skimage.filters as filters
import skimage.morphology as morphology
import pims

from plot_style import apply_calcium_style


@dataclass
class AnalysisConfig:
    """Configuration parameters for calcium transient analysis."""
    
    # Image parameters
    fluo_index: int = 0
    linescan_speed: float = 1.87  # ms per line
    filter_kernelsize: int = 5
    
    # Analysis parameters
    mode: str = 'single'  # 'single' or 'ratio'
    analyze_synchrony: bool = False
    
    # Peak detection thresholds
    peak_prominence_ratio: float = 0.45  # For find_peaks
    peak_prominence_single: float = 0.8  # For single transient
    peak_prominence_sync: float = 0.9    # For synchrony analysis
    min_peak_distance: int = 200
    
    # Visualization
    show_images: bool = True
    max_ff0: float = 8.0
    
    # Export
    export_csv: bool = False
    export_npz: bool = False
    
    @property
    def sampling_rate(self) -> float:
        """Calculate sampling rate from linescan speed."""
        return 1 / self.linescan_speed * 1000


class CalciumImage:
    """Handles loading and preprocessing of calcium linescan images."""
    
    def __init__(self, filepath: str, config: AnalysisConfig):
        """
        Initialize calcium image from file.
        
        Args:
            filepath: Path to TIFF image file
            config: Analysis configuration
        """
        self.filepath = Path(filepath)
        self.config = config
        
        # Load image
        self._image = pims.Bioformats(str(filepath))
        self.metadata = self._image.metadata
        
        # Extract calcium fluorescence channel
        self.raw_data = np.transpose(self._image[config.fluo_index])
        self.voxel_size = self.metadata.PixelsPhysicalSizeX(0)  # μm
        
        # Processed data (will be set during preprocessing)
        self.filtered_data: Optional[np.ndarray] = None
        self.cropped_data: Optional[np.ndarray] = None
        self.normalized_data: Optional[np.ndarray] = None
        self.signal_1d: Optional[np.ndarray] = None
        
        # ROI information
        self.cell_limits: Optional[Tuple[int, int]] = None
        self.baseline_limits: Optional[Tuple[int, int]] = None
        self.baseline_value: Optional[float] = None
        
    def apply_median_filter(self) -> 'CalciumImage':
        """Apply median filter to reduce noise."""
        kernel_size = self.config.filter_kernelsize
        footprint = morphology.footprint_rectangle((kernel_size, kernel_size))
        self.filtered_data = filters.median(self.raw_data, footprint)
        return self
    
    def select_cell_roi(self, ylimits: Optional[Tuple[float, float]] = None) -> 'CalciumImage':
        """
        Select cell region of interest.
        
        Args:
            ylimits: Optional tuple of (y_start, y_end). If None, prompts user.
        """
        if self.filtered_data is None:
            raise ValueError("Must apply median filter before selecting ROI")
        
        if ylimits is None:
            # Interactive selection
            fig = plt.figure(figsize=(16, 5), constrained_layout=True)
            plt.imshow(self.filtered_data, cmap='inferno', vmin=0,
                      vmax=np.mean(self.filtered_data) * 10, aspect='auto')
            plt.xlim(0, 2000)
            plt.title("Select cell limits (2 clicks)")
            clicks = plt.ginput(n=2, timeout=60)
            apply_calcium_style()
            plt.close(fig)
            ylimits = (clicks[0][1], clicks[1][1])
        
        self.cell_limits = (int(ylimits[0]), int(ylimits[1]))
        self.cropped_data = self.filtered_data[self.cell_limits[0]:self.cell_limits[1]]
        return self
    
    def select_baseline_roi(self, xlimits: Optional[Tuple[float, float]] = None) -> 'CalciumImage':
        """
        Select baseline region for F/F0 normalization.
        
        Args:
            xlimits: Optional tuple of (x_start, x_end). If None, prompts user.
        """
        if self.cropped_data is None:
            raise ValueError("Must select cell ROI before baseline")
        
        # Create 1D signal by averaging along spatial axis
        ca_flat = np.mean(self.cropped_data, axis=0)
        
        if xlimits is None:
            # Interactive selection
            fig = plt.figure(figsize=(16, 5), constrained_layout=True)
            plt.plot(ca_flat)
            plt.xlim(0, 2000)
            plt.title("Select baseline region (2 clicks)")
            clicks = plt.ginput(n=2, timeout=60)
            apply_calcium_style()
            plt.close(fig)
            xlimits = (clicks[0][0], clicks[1][0])
        
        self.baseline_limits = (int(xlimits[0]), int(xlimits[1]))
        self.baseline_value = np.mean(ca_flat[self.baseline_limits[0]:self.baseline_limits[1]])
        return self
    
    def normalize(self) -> 'CalciumImage':
        """Normalize to F/F0 and smooth the signal."""
        if self.baseline_value is None:
            raise ValueError("Must select baseline before normalization")
        
        # Create 1D signal
        ca_flat = np.mean(self.cropped_data, axis=0)
        self.signal_1d = ca_flat / self.baseline_value
        
        # Smooth with Savitzky-Golay filter
        self.signal_1d = signal.savgol_filter(self.signal_1d, 21, 3)
        
        # Normalize 2D image
        self.normalized_data = self.cropped_data / self.baseline_value
        
        return self
    
    def get_time_axis(self) -> np.ndarray:
        """Get time axis in milliseconds."""
        if self.signal_1d is None:
            raise ValueError("Must normalize before getting time axis")
        return np.linspace(0, self.config.linescan_speed * len(self.signal_1d), len(self.signal_1d))
    
    def get_space_axis(self) -> np.ndarray:
        """Get spatial axis in micrometers."""
        if self.normalized_data is None:
            raise ValueError("Must normalize before getting space axis")
        return np.linspace(0, self.voxel_size * self.normalized_data.shape[0], 
                          self.normalized_data.shape[0])


@dataclass
class TransientMetrics:
    """Container for calcium transient analysis metrics."""
    
    begin_idx: int
    end_idx: int
    frequency: float
    baseline: float
    peak: float
    amplitude: float
    rise_time_10_90: float  # ms
    decay_time_50: float    # ms
    decay_time_90: float    # ms
    
    # Synchrony metrics (optional)
    delay_mean: Optional[float] = None  # ms
    delay_std: Optional[float] = None   # ms
    synchrony_index: Optional[float] = None
    
    def to_array(self, include_synchrony: bool = False) -> np.ndarray:
        """Convert metrics to numpy array."""
        base = np.array([
            self.begin_idx, self.end_idx, self.frequency,
            self.baseline, self.peak, self.amplitude,
            self.rise_time_10_90, self.decay_time_50, self.decay_time_90
        ])
        
        if include_synchrony and self.delay_mean is not None:
            return np.concatenate([base, [self.delay_mean, self.delay_std, self.synchrony_index]])
        return base
    
    @staticmethod
    def get_header(include_synchrony: bool = False) -> List[str]:
        """Get CSV header for metrics."""
        base = ['Begin', 'End', 'Freq (Hz)', 'Ca Baseline', 'Ca Peak',
                'Ca Amplitude', 'Ca Rise Time 10-90%(ms)', 
                'Ca Decay time 50%(ms)', 'Ca Decay time 90%(ms)']
        
        if include_synchrony:
            base.extend(['Delay(ms)', 'SD(ms)', 'SI'])
        return base


class CalciumTransient:
    """Represents a single calcium transient with analysis methods."""
    
    def __init__(self, signal: np.ndarray, sampling_rate: float, 
                 transient_id: int = 0, mode: str = 'single'):
        """
        Initialize calcium transient.
        
        Args:
            signal: 1D calcium signal (F/F0)
            sampling_rate: Sampling rate in Hz
            transient_id: Identifier for this transient
            mode: 'single' or 'ratio' wavelength mode
        """
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.transient_id = transient_id
        self.mode = mode
        
        self.metrics: Optional[TransientMetrics] = None
        
    def analyze(self, config: AnalysisConfig, linescan: Optional[np.ndarray] = None) -> Optional[TransientMetrics]:
        """
        Analyze the transient to extract metrics.
        
        Args:
            config: Analysis configuration
            linescan: Optional 2D linescan for synchrony analysis
            
        Returns:
            TransientMetrics object or None if analysis fails
        """
        # Determine threshold based on mode
        if self.mode == 'ratio':
            prominence = 0.9
        else:
            prominence = 0.8
        
        # Calculate baseline
        baseline = np.mean(self.signal[:10])
        
        # Detect peaks
        peaks, _ = signal.find_peaks(self.signal, 
                                    prominence=prominence * np.ptp(self.signal),
                                    distance=100)
        
        # Validate single peak
        if len(peaks) != 1:
            print(f'Invalid Transient #{self.transient_id}: {len(peaks)} peaks found')
            return None
        
        peak_idx = int(peaks[0])
        peak_value = self.signal[peak_idx]
        amplitude = peak_value - baseline
        
        # Calculate rise time (10-90%)
        rise_time = self._calculate_rise_time(baseline, amplitude)
        
        # Calculate decay times
        decay_50 = self._calculate_decay_time(peak_idx, peak_value, amplitude, 0.5)
        decay_90 = self._calculate_decay_time(peak_idx, peak_value, amplitude, 0.9)
        
        # Create metrics
        self.metrics = TransientMetrics(
            begin_idx=0,
            end_idx=len(self.signal),
            frequency=1.0,  # Will be updated by analyzer
            baseline=baseline,
            peak=peak_value,
            amplitude=amplitude,
            rise_time_10_90=rise_time,
            decay_time_50=decay_50,
            decay_time_90=decay_90
        )
        
        # Analyze synchrony if linescan provided
        if linescan is not None and config.analyze_synchrony:
            self._analyze_synchrony(linescan, baseline, peak_idx, amplitude)
        
        return self.metrics
    
    def _calculate_rise_time(self, baseline: float, amplitude: float) -> float:
        """Calculate 10-90% rise time in milliseconds."""
        r10_idx = np.where(self.signal >= (baseline + amplitude * 0.1))[0]
        r90_idx = np.where(self.signal >= (baseline + amplitude * 0.9))[0]
        
        if r10_idx.size and r90_idx.size:
            r10 = int(r10_idx[0])
            r90 = int(r90_idx[0])
            rise_time = ((r90 - r10) / self.sampling_rate) * 1000
            return rise_time if rise_time >= 10 else np.nan
        return np.nan
    
    def _calculate_decay_time(self, peak_idx: int, peak_value: float, 
                             amplitude: float, fraction: float) -> float:
        """Calculate decay time to specified fraction in milliseconds."""
        threshold = peak_value - amplitude * fraction
        decay_idx = np.where(self.signal[peak_idx:] <= threshold)[0]
        
        if decay_idx.size:
            idx = int(decay_idx[0])
            decay_time = (idx / self.sampling_rate) * 1000
            min_time = 10 if fraction == 0.5 else 50
            return decay_time if decay_time >= min_time else np.nan
        return np.nan
    
    def _analyze_synchrony(self, linescan: np.ndarray, baseline: float,
                          peak_idx: int, amplitude: float, ampd: float = 50):
        """Analyze spatial synchrony from linescan data."""
        # Calculate per-row metrics
        sy_max = np.max(linescan, axis=1)
        sy_bl = np.mean(linescan[:, :10], axis=1)
        sy_amp = sy_max - sy_bl
        sy_thres = sy_max - sy_amp * ((100 - ampd) / 100)
        
        delay_times = np.zeros(linescan.shape[0])
        
        # Calculate delay for each row
        for r in range(linescan.shape[0]):
            delay_idx = np.where(linescan[r] >= sy_thres[r])[0]
            if delay_idx.size:
                delay = int(delay_idx[0])
                delay_time = (delay / self.sampling_rate) * 1000
                delay_times[r] = delay_time if delay_time >= 10 else np.nan
            else:
                delay_times[r] = np.nan
        
        # Calculate synchrony index
        delay_mean = np.nanmean(delay_times)
        delay_std = np.nanstd(delay_times)
        si = delay_std / delay_mean if delay_mean > 0 else np.nan
        
        # Update metrics
        if self.metrics:
            self.metrics.delay_mean = delay_mean
            self.metrics.delay_std = delay_std
            self.metrics.synchrony_index = si


class CalciumAnalyzer:
    """Main analyzer class that orchestrates the calcium transient analysis workflow."""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize analyzer.
        
        Args:
            config: Analysis configuration. If None, uses defaults.
        """
        self.config = config or AnalysisConfig()
        self.image: Optional[CalciumImage] = None
        self.transients: List[CalciumTransient] = []
        self.results: Optional[np.ndarray] = None
        
    def load_image(self, filepath: str) -> 'CalciumAnalyzer':
        """Load and preprocess calcium image."""
        self.image = CalciumImage(filepath, self.config)
        self.image.apply_median_filter()
        return self
    
    def select_rois(self, cell_limits: Optional[Tuple[float, float]] = None,
                   baseline_limits: Optional[Tuple[float, float]] = None) -> 'CalciumAnalyzer':
        """Select ROIs for analysis."""
        if self.image is None:
            raise ValueError("Must load image first")
        
        self.image.select_cell_roi(cell_limits)
        self.image.select_baseline_roi(baseline_limits)
        self.image.normalize()
        return self
    
    def find_and_analyze_transients(self) -> 'CalciumAnalyzer':
        """Find and analyze all calcium transients in the signal."""
        if self.image is None or self.image.signal_1d is None:
            raise ValueError("Must load and process image first")
        
        # Determine threshold based on mode
        if self.config.mode == 'ratio':
            prominence = 0.6
        else:
            prominence = self.config.peak_prominence_ratio
        
        # Find peaks
        signal_1d = self.image.signal_1d
        peaks, _ = signal.find_peaks(signal_1d,
                                    prominence=prominence * np.ptp(signal_1d),
                                    distance=self.config.min_peak_distance)
        
        n_peaks = len(peaks)
        if n_peaks < 1:
            print('No transients found')
            return self
        
        # Determine if we should skip last transient
        skip_last = n_peaks > 1
        n_analyze = n_peaks - 1 if skip_last else n_peaks
        
        # Prepare results array
        n_cols = 12 if self.config.analyze_synchrony else 9
        self.results = np.zeros((n_analyze, n_cols))
        
        # Analyze each transient
        for i in range(n_analyze):
            peak_idx = peaks[i]
            
            # Skip if peak too early
            if peak_idx < 50:
                self.results[i, :] = np.nan
                continue
            
            # Calculate frequency
            if i < n_analyze - 1 and n_peaks > 1:
                freq = round(self.config.sampling_rate / (peaks[i + 1] - peaks[i]))
                freq = max(freq, 1)
            else:
                freq = 1
            
            # Extract transient window
            duration = int(((1000 / freq) * 0.8) / (1 / self.config.sampling_rate * 1000))
            start_idx = peak_idx - 50
            end_idx = peak_idx + duration
            
            transient_signal = signal_1d[start_idx:end_idx]
            
            # Create and analyze transient
            transient = CalciumTransient(transient_signal, self.config.sampling_rate, 
                                        i, self.config.mode)
            
            # Get linescan crop if needed
            linescan_crop = None
            if self.config.analyze_synchrony and self.image.normalized_data is not None:
                linescan_crop = self.image.normalized_data[:, start_idx:end_idx]
            
            metrics = transient.analyze(self.config, linescan_crop)
            
            if metrics:
                # Update indices and frequency
                metrics.begin_idx = start_idx
                metrics.end_idx = end_idx
                metrics.frequency = freq
                
                # Store results
                self.results[i, :] = metrics.to_array(self.config.analyze_synchrony)
                self.transients.append(transient)
            else:
                self.results[i, :] = np.nan
        
        return self
    
    def visualize(self) -> 'CalciumAnalyzer':
        """Visualize the analysis results."""
        if not self.config.show_images or self.image is None:
            return self
        
        if self.image.signal_1d is None or self.results is None:
            raise ValueError("Must analyze transients before visualization")
        
        time_axis = self.image.get_time_axis()
        space_axis = self.image.get_space_axis()
        
        # Plot 1: Signal with detected transients
        plt.figure(figsize=(16, 5), constrained_layout=True)
        plt.plot(time_axis, self.image.signal_1d)
        
        # Mark begin/end points
        valid_rows = ~np.isnan(self.results[:, 0])
        if np.any(valid_rows):
            begin_indices = self.results[valid_rows, 0].astype(int)
            end_indices = self.results[valid_rows, 1].astype(int)
            
            plt.plot(time_axis[begin_indices],
                    self.image.signal_1d[begin_indices],
                    label='Begin', marker='o', color='r',
                    fillstyle='none', linestyle='none')
            plt.plot(time_axis[end_indices],
                    self.image.signal_1d[end_indices],
                    label='End', marker='o', color='g',
                    fillstyle='none', linestyle='none')
        
        plt.xlim(0, 3000)
        plt.xlabel('Time (ms)')
        plt.ylabel('Ca Signal (F/F0)')
        plt.legend()
        apply_calcium_style()
        
        # Plot 2: Normalized linescan image
        plt.figure(figsize=(16, 5), constrained_layout=True)
        extent = [0, self.config.linescan_speed * len(self.image.signal_1d),
                 0, self.image.voxel_size * self.image.normalized_data.shape[0]]
        
        plt.imshow(self.image.normalized_data, cmap='inferno',
                  vmin=0, vmax=self.config.max_ff0,
                  aspect='auto', extent=extent)
        plt.xlabel('Time (ms)')
        plt.ylabel('Distance (μm)')
        plt.xlim(0, 3000)
        cbar = plt.colorbar(orientation='horizontal', shrink=0.6)
        cbar.set_label('F/F0')
        apply_calcium_style()
        
        plt.show()
        return self
    
    def export_results(self, output_path: Optional[str] = None) -> 'CalciumAnalyzer':
        """
        Export analysis results to CSV and NPZ files.
        
        Args:
            output_path: Optional output path. If None, uses input image path.
        """
        if not (self.config.export_csv or self.config.export_npz):
            return self
        
        if self.results is None or self.image is None:
            raise ValueError("Must analyze transients before export")
        
        # Determine output path
        if output_path is None:
            output_path = self.image.filepath.parent / self.image.filepath.stem
        else:
            output_path = Path(output_path)
        
        # Prepare header
        header = TransientMetrics.get_header(self.config.analyze_synchrony)
        
        # Export CSV
        if self.config.export_csv:
            csv_path = output_path.with_suffix('.csv')
            np.savetxt(csv_path, self.results, delimiter=',',
                      header=','.join(header), fmt='%1.3f', comments='')
            print(f"Results exported to {csv_path}")
        
        # Export NPZ
        if self.config.export_npz:
            npz_path = output_path.with_suffix('.npz')
            np.savez_compressed(npz_path,
                              signal=self.image.signal_1d,
                              data=self.results,
                              sampling=self.config.sampling_rate,
                              header=header)
            print(f"Data exported to {npz_path}")
        
        return self
    
    def run_full_analysis(self, filepath: str,
                         cell_limits: Optional[Tuple[float, float]] = None,
                         baseline_limits: Optional[Tuple[float, float]] = None) -> 'CalciumAnalyzer':
        """
        Run complete analysis pipeline.
        
        Args:
            filepath: Path to TIFF image
            cell_limits: Optional cell ROI limits
            baseline_limits: Optional baseline ROI limits
        """
        return (self.load_image(filepath)
                   .select_rois(cell_limits, baseline_limits)
                   .find_and_analyze_transients()
                   .visualize()
                   .export_results())
