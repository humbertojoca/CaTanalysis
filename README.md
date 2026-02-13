# Calcium Transient Analysis Framework v3.01 (Feb/2026)

A Python framework for analyzing calcium transients from linescan microscopy images.

## 🎯 Features
- **Modern GUI**: Built with PySide6 for interactive and user-friendly analysis.
- **Interactive ROI Selection**: Precise control over cell boundaries (Y-axis) and baseline regions (X-axis) with sliders and spinboxes.
- **Ratio Mode**: Support for multi-channel ratio analysis (e.g., Fura-2 or Ratiometric indicators).
- **Multi-format Support**: Loads any image supported by Bio-formats, including **ND2 (Nikon)**, **CZI/LSM (Zeiss)**, and standard **TIFF**.
- **Peak Detection**: Uses scipy's `find_peaks` with prominence-based detection.
- **Synchrony Analysis**: Spatial synchrony analysis with activation delay maps (single-wavelength mode).
- **Export Options**: Automatic CSV export of all kinetic parameters.

## 📦 Installation

### Requirements

```bash
pip install numpy scipy matplotlib scikit-image pims jpype1 PySide6 seaborn
```
The code uses `pims.Bioformats` which requires Java. Make sure you have Java installed.

## 🚀 Usage

### 1. Graphical User Interface (Recommended)
Launch the modern GUI for an interactive workflow:
```bash
python main_gui.py
```
- **Configuration**: Adjust all analysis parameters (sampling, filters, thresholds) in the left panel.
- **ROI Selection**: Use the sliders in the "Image Viewer" tab to select cell limits and baseline regions.
- **Visualization**: View signal plots, normalized linescan images, and synchrony maps in dedicated tabs.

### 2. Command Line Interface
Run batch analysis on multiple files:
```bash
python main.py
```
This will open a file dialog to select images and process them using the settings defined in `ca_analysis_config.json`.

---

## 🔧 Configuration Options

The framework uses `ca_analysis_config.json` to persist settings.

### Key Parameters

- **Image settings**: `fluo_index` (for single mode), `numerator_index`, `denominator_index` (for ratio mode), `linescan_speed` (ms/line).
- **Analysis mode**: `'single'` or `'ratio'`.
- **Synchrony**: `analyze_synchrony` (True/False).
- **Peak detection**: `peak_prominence_ratio`, `min_peak_distance`.
- **Visualization**: `max_ff0` (clipping for colorbar).

---

## 📂 Core Components

### 1. `CalciumAnalyzer`
Main class that manages the complete analysis workflow, from loading to export.

### 2. `CalciumImage`
Handles Bio-formats loading, median filtering, and ROI-based normalization (F/F0).

### 3. `CalciumTransient`
Analyzes individual transients to extract:
- **Rise Time**: 10-90% interval.
- **Decay Times**: 50% and 90% (tau/kinetics).
- **Synchrony Index**: Spatial variation in activation delays.

## 📊 Output Results

| Column | Metric | Unit |
|--------|--------|------|
| 0 | Begin index | - |
| 1 | End index | - |
| 2 | Frequency | Hz |
| 3 | Baseline | F/F0 |
| 4 | Peak | F/F0 |
| 5 | Amplitude | F/F0 |
| 6 | Rise time (10-90%) | ms |
| 7 | Decay time (50%) | ms |
| 8 | Decay time (90%) | ms |
| 9* | Delay mean | ms |
| 10* | Delay SD | ms |
| 11* | Synchrony index | - |

\* Synchrony metrics available in `'single'` mode.

## ⚠️ Java Warnings
If you see Java native access warnings:
```
WARNING: A restricted method in java.lang.System has been called
```
The code automatically suppresses these by setting `JAVA_TOOL_OPTIONS`.