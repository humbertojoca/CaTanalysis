"""
Main script for calcium transient analysis.

This script provides a simple interface to analyze calcium transients
from linescan microscopy images.

Author: hjoca
Date: 2026-01-13
"""

import os
# Suppress Java native access warnings from JPype/Bioformats
os.environ['JAVA_TOOL_OPTIONS'] = '--enable-native-access=ALL-UNNAMED'

import sys
import tkinter as tk
from tkinter import filedialog

from ca_analyzer import CalciumAnalyzer, AnalysisConfig


def select_files() -> list:
    """Open file dialog to select TIFF images."""
    root = tk.Tk()
    root.withdraw()
    
    filenames = filedialog.askopenfilenames(
        parent=root,
        filetypes=[('TIFF Image', '*.tif;*.tiff'), ('All files', '*.*')],
        title='Choose TIFF Image(s)'
    )
    
    if not filenames:
        sys.exit('No file selected!')
    
    return list(filenames)


def main():
    """Main analysis workflow."""
    
    # Configure analysis parameters
    config = AnalysisConfig(
        fluo_index=0,
        linescan_speed=1.87,  # ms per line
        filter_kernelsize=5,
        mode='single',  # 'single' or 'ratio'
        analyze_synchrony=False,
        show_images=True,
        max_ff0=8.0,
        export_csv=False,
        export_npz=False,
        peak_prominence_ratio=0.45,
        min_peak_distance=200
    )
    
    # Select files
    print("Select TIFF image(s) for analysis...")
    files = select_files()
    print(f"Selected {len(files)} file(s)")
    
    # Analyze each file
    for i, filepath in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"Analyzing file {i}/{len(files)}: {filepath}")
        print(f"{'='*60}")
        
        try:
            # Create analyzer and run full analysis
            analyzer = CalciumAnalyzer(config)
            analyzer.run_full_analysis(filepath)
            
            # Print summary
            if analyzer.results is not None:
                n_transients = len(analyzer.transients)
                print(f"\n✓ Analysis complete!")
                print(f"  Found {n_transients} transient(s)")
                
                if n_transients > 0:
                    # Print summary statistics
                    valid_results = analyzer.results[~np.isnan(analyzer.results[:, 0])]
                    if len(valid_results) > 0:
                        print(f"\n  Summary Statistics:")
                        print(f"  - Mean amplitude: {np.nanmean(valid_results[:, 5]):.3f}")
                        print(f"  - Mean rise time: {np.nanmean(valid_results[:, 6]):.2f} ms")
                        print(f"  - Mean decay 50%: {np.nanmean(valid_results[:, 7]):.2f} ms")
                        print(f"  - Mean decay 90%: {np.nanmean(valid_results[:, 8]):.2f} ms")
            
        except Exception as e:
            print(f"\n✗ Error analyzing file: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("All files processed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    import numpy as np  # Import here for summary statistics
    main()
