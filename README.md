# Calcium Transient Analysis Framework v2 (Jan/2026)

A Python framework for analyzing calcium transients from linescan microscopy images.

## 🎯 Features
- **Peak Detection**: Uses scipy's `find_peaks` with prominence-based detection
- **Synchrony Analysis**: Optional spatial synchrony analysis from linescan data
- **ROI Selection**: User-friendly region selection with matplotlib
- **Export Options**: CSV and NPZ export formats

## 📦 Installation

### Requirements

```bash
pip install numpy scipy matplotlib scikit-image pims jpype1 seaborn
```
The code uses `pims.Bioformats` which requires Java. Make sure you have Java installed.

### Core Classes

#### 1. `AnalysisConfig`
Configuration dataclass containing all analysis parameters:
- Image parameters (channel index, linescan speed, filter size)
- Peak detection thresholds
- Visualization settings
- Export options

#### 2. `CalciumImage`
Handles image loading and preprocessing:
- Loads images via pims/Bioformats
- Applies median filtering
- Interactive ROI selection (cell boundaries and baseline)
- F/F0 normalization
- Signal smoothing with Savitzky-Golay filter

#### 3. `CalciumTransient`
Represents a single calcium transient:
- Peak detection
- Rise time calculation (10-90%)
- Decay time calculation (50%, 90%)
- Optional synchrony analysis

#### 4. `TransientMetrics`
Dataclass for storing analysis results:
- Temporal indices (begin, end)
- Frequency
- Baseline, peak, amplitude
- Kinetic parameters (rise/decay times)
- Optional synchrony metrics (delay, SD, synchrony index)

#### 5. `CalciumAnalyzer`
Main orchestrator class:
- Manages the complete analysis workflow
- Finds and analyzes multiple transients
- Visualization
- Results export

## 📊 Output

### Results Array Structure

Each row contains metrics for one transient:

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

\* Only when `analyze_synchrony=True`

### Export Formats

**CSV**: Human-readable format with headers
```csv
Begin,End,Freq (Hz),Ca Baseline,Ca Peak,Ca Amplitude,...
50,450,1.0,1.02,3.45,2.43,45.2,120.5,350.2
```

**NPZ**: Compressed numpy format containing:
- `signal`: 1D normalized calcium signal
- `data`: Results array
- `sampling`: Sampling rate
- `header`: Column names

## 🔧 Configuration Options

### Key Parameters

```python
config = AnalysisConfig(
    # Image settings
    fluo_index=0,              # Fluorescence channel (0-indexed)
    linescan_speed=1.87,       # ms per scan line
    filter_kernelsize=5,       # Median filter kernel size
    
    # Analysis mode
    mode='single',             # 'single' or 'ratio' wavelength
    analyze_synchrony=False,   # Enable spatial synchrony analysis
    
    # Peak detection
    peak_prominence_ratio=0.45,  # Prominence for finding transients
    min_peak_distance=200,       # Minimum distance between peaks
    
    # Visualization
    show_images=True,          # Display plots
    max_ff0=8.0,              # Max F/F0 for colormap
    
    # Export
    export_csv=True,          # Export to CSV
    export_npz=True           # Export to NPZ
)
```

### Java Warnings
If you see Java native access warnings:
```
WARNING: A restricted method in java.lang.System has been called
```

The code automatically sets the environment variable to suppress these:
```python
os.environ['JAVA_TOOL_OPTIONS'] = '--enable-native-access=ALL-UNNAMED'
```