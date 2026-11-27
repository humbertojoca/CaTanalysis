# Calcium Transient Analysis Framework v3.06 (Nov/2026)

A Python framework for analyzing calcium transients from linescan microscopy images.

## 🎯 Features
- **Modern GUI**: Built with PySide6 for interactive and user-friendly analysis.
- **Interactive ROI Selection**: Precise control over cell boundaries (Y-axis) and baseline regions (X-axis) with sliders and spinboxes.
- **Ratio Mode**: Support for multi-channel ratio analysis (e.g., Fura-2 or Ratiometric indicators).
- **Multi-format Support**: Loads images through [bioio](https://github.com/bioio-devs/bioio) (no Java required): **ND2 (Nikon)**, **CZI (Zeiss)**, **LIF (Leica)**, and standard/OME **TIFF**.
- **Nikon ND2 linescans**: ND2 files store the linescan split into frames; they are automatically concatenated along time and rotated 90° to the left before analysis. The linescan time (ms/line) is read automatically from the frame timestamps in the file (see *Linescan speed from file* below).
- **File Explorer**: Docked folder browser (double-click/Enter to open, Prev/Next with Alt+Left/Alt+Right). Images that already have a results CSV are marked with a green background and a check mark.
- **Batch-friendly ROIs**: ROIs are reset on every load by default; tick *Keep ROI positions* (Visualization box, config key `keep_roi`) to re-apply the previous ROIs to the next file.
- **Fast & Responsive**: OpenCV median filter (~25x faster), downsampled image display, and blitted ROI lines (moving ROI lines takes ~6 ms instead of ~400 ms).
- **Peak Detection**: Uses scipy's `find_peaks` with prominence-based detection.
- **Synchrony Analysis**: Spatial synchrony analysis with activation delay maps (single-wavelength mode).
- **Export Options**: Automatic CSV export of all kinetic parameters, plus optional PNG export of the signal profile (`<name>_profile.png`) and the normalized linescan (`<name>_linescan.png`) from the GUI.

## 📦 Installation

### Requirements

```bash
mamba create -n catanalysis python=3.12 numpy scipy matplotlib scikit-image opencv bioio bioio-nd2 bioio-tifffile bioio-ome-tiff bioio-lif pyside6 seaborn
mamba activate catanalysis
pip install bioio-czi    # only needed for Zeiss CZI files (not available on conda-forge)
```
(Use `mamba install -n <env> ...` instead of `create` to add the packages to an existing environment.)

Run the software from that environment, e.g. `mamba run -n catanalysis python main_gui.py`.

`opencv` is optional: it speeds up the median filter for kernel sizes 3 and 5 (the default is 5). Without it, or for other kernel sizes, the median filter falls back to scikit-image (much slower). Image loading uses `bioio` plugins, so install the plugin for each format you need. Java is **no longer** required.

**Pixel size:** it is read from the file metadata (`physical_pixel_sizes.X`, in µm). If a file has none (e.g. some exported TIFFs), 1.0 is used and a warning is printed, so distance axes are then in pixels.

**Linescan speed from file (ND2):** the line time is calculated as the frame-to-frame interval divided by the number of lines per frame, and replaces the *Linescan Speed* value for that file (status bar: "line time ... ms (from file)"). It is only used when frames are acquired back to back and evenly spaced. Files without frame timestamps (e.g. `*_Concatenated.nd2`, TIFF) use the value typed in the config panel. Untick *Linescan speed from file* (`use_file_linescan_speed`) to always use the manual value. *Save Configuration* stores your manual value, not the one read from a file.

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

The framework uses `ca_analysis_config.json` to persist settings. Results are exported next to each image as `<image name without extension>.csv` (e.g. `cell1.qc3005.nd2` -> `cell1.qc3005.csv`).

### Key Parameters

- **Image settings**: `fluo_index` (for single mode), `numerator_index`, `denominator_index` (for ratio mode), `linescan_speed` (ms/line), `use_file_linescan_speed` (ND2: read line time from file), `filter_kernelsize` (median filter size).
- **Analysis mode**: `'single'` or `'ratio'`.
- **Synchrony**: `analyze_synchrony` (True/False).
- **Peak detection**: `peak_prominence_ratio`, `peak_prominence_single`, `peak_prominence_sync`, `min_peak_distance`.
- **Visualization**: `show_images`, `max_ff0` (clipping for colorbar), `keep_roi` (keep ROI positions between files).
- **Export**: `export_csv`, `export_signal_png`, `export_linescan_png`.

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
| 3 | Ca Baseline | F/F0 |
| 4 | Ca Peak | F/F0 |
| 5 | Ca Amplitude | F/F0 |
| 6 | Rise time (10-90%) | ms |
| 7 | Decay time (50%) | ms |
| 8 | Decay time (90%) | ms |
| 9 | Ca Peak Idx (index of the peak) | - |
| 10* | Delay mean | ms |
| 11* | Delay SD | ms |
| 12* | Synchrony index (SI) | - |

\* Synchrony metrics available in `'single'` mode.

## ⚡ Performance Notes
In version 3.06 the image reader and the image display were changed for better performance. Typical 1024 x 13312 ND2 linescan, load + filter: ~18 s (previous solution: Bio-Formats + scikit-image) -> ~1 s (bioio + OpenCV), with bit-identical pixel data.

- The displayed image is block-averaged along time (~2000 columns) for speed; analysis always uses the full-resolution data.
- Zoom/pan still re-renders the image; ROI line changes do not.
