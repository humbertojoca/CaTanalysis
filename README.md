# Calcium Transient Analysis Framework v3.06 (Nov/2026)

A Python framework for analyzing calcium transients from linescan microscopy images.

## 🎯 Features
- **Modern GUI**: Built with PySide6 for interactive and user-friendly analysis.
- **Interactive ROI Selection**: Precise control over cell boundaries (Y-axis) and baseline regions (X-axis) with sliders and spinboxes.
- **Ratio Mode**: Support for multi-channel ratio analysis (e.g., Fura-2 or Ratiometric indicators).
- **Multi-format Support**: Loads images through [bioio](https://github.com/bioio-devs/bioio) (no Java required): **ND2 (Nikon)**, **CZI (Zeiss)**, **LIF (Leica)**, and standard/OME **TIFF**.
- **Nikon ND2 linescans**: ND2 files store the linescan split into frames; they are automatically concatenated along time and rotated 90° to the left before analysis.
- **Fast & Responsive**: OpenCV median filter (~25x faster), downsampled image display, and blitted ROI lines (moving ROI lines takes ~6 ms instead of ~400 ms).
- **Peak Detection**: Uses scipy's `find_peaks` with prominence-based detection.
- **Synchrony Analysis**: Spatial synchrony analysis with activation delay maps (single-wavelength mode).
- **Export Options**: Automatic CSV export of all kinetic parameters.

## 📦 Installation

### Requirements

```bash
mamba install -n basic numpy scipy matplotlib scikit-image opencv bioio bioio-nd2 bioio-tifffile bioio-ome-tiff bioio-czi bioio-lif pyside6 seaborn
```
Run the software in your environment, e.g. `mamba run -n basic python main_gui.py`.

`opencv` is optional: without it the median filter falls back to scikit-image (much slower). Image loading uses `bioio` plugins, so install the plugin for each format you need (Java is **not** required).

**Pixel size:** it is read from the file metadata (`physical_pixel_sizes.X`, in µm). If a file has none (e.g. some exported TIFFs), 1.0 is used and a warning is printed, so distance axes are then in pixels.

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
Handles bioio loading (including ND2 frame concatenation/rotation), median filtering, and ROI-based normalization (F/F0).

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

## ⚡ Performance Notes
Typical 1024 x 13312 ND2 linescan, load + filter: ~18 s (Bio-Formats + scikit-image) -> ~1 s (bioio + OpenCV), with bit-identical pixel data.
- The displayed image is block-averaged along time (~2000 columns) for speed; analysis always uses the full-resolution data.
- Zoom/pan still re-renders the image; ROI line changes do not.
