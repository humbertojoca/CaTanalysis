"""
Configuration Manager for Calcium Transient Analysis

Handles loading and saving analysis configuration to/from JSON files.

Author: hjoca
Date: 2026-02-09
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict

from ca_analyzer import AnalysisConfig


class ConfigManager:
    """Manages loading and saving of analysis configuration."""
    
    DEFAULT_CONFIG_FILE = "ca_analysis_config.json"
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default.
        """
        if config_path is None:
            # Use the script directory for the config file
            self.config_path = Path(__file__).parent / self.DEFAULT_CONFIG_FILE
        else:
            self.config_path = Path(config_path)
    
    def load_config(self) -> AnalysisConfig:
        """
        Load configuration from JSON file.
        
        Returns:
            AnalysisConfig object with loaded parameters.
            If file doesn't exist, returns default configuration.
        """
        if not self.config_path.exists():
            print(f"Configuration file not found: {self.config_path}")
            print("Using default configuration.")
            config = AnalysisConfig()
            # Save default configuration for future use
            self.save_config(config)
            return config
        
        try:
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Validate and create AnalysisConfig
            config = self._dict_to_config(config_dict)
            print(f"Configuration loaded from: {self.config_path}")
            return config
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration.")
            return AnalysisConfig()
    
    def save_config(self, config: AnalysisConfig) -> bool:
        """
        Save configuration to JSON file.
        
        Args:
            config: AnalysisConfig object to save.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            config_dict = self._config_to_dict(config)
            
            # Create parent directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
            
            print(f"Configuration saved to: {self.config_path}")
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def _config_to_dict(self, config: AnalysisConfig) -> Dict[str, Any]:
        """
        Convert AnalysisConfig to dictionary.
        
        Args:
            config: AnalysisConfig object.
            
        Returns:
            Dictionary representation of the configuration.
        """
        # Use dataclass asdict, but exclude the computed property
        config_dict = asdict(config)
        return config_dict
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AnalysisConfig:
        """
        Convert dictionary to AnalysisConfig.
        
        Args:
            config_dict: Dictionary with configuration parameters.
            
        Returns:
            AnalysisConfig object.
        """
        # Filter out any keys that aren't valid AnalysisConfig fields
        valid_keys = {
            'fluo_index', 'numerator_index', 'denominator_index', 'linescan_speed', 'filter_kernelsize',
            'mode', 'analyze_synchrony', 'peak_prominence_ratio',
            'peak_prominence_single', 'peak_prominence_sync',
            'min_peak_distance', 'show_images', 'max_ff0',
            'export_csv'
        }
        
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return AnalysisConfig(**filtered_dict)
    
    def reset_to_defaults(self) -> AnalysisConfig:
        """
        Reset configuration to defaults and save.
        
        Returns:
            Default AnalysisConfig object.
        """
        config = AnalysisConfig()
        self.save_config(config)
        print("Configuration reset to defaults.")
        return config
    
    def get_config_path(self) -> Path:
        """Get the current configuration file path."""
        return self.config_path


def get_default_config_manager() -> ConfigManager:
    """
    Get a ConfigManager instance with the default configuration file.
    
    Returns:
        ConfigManager instance.
    """
    return ConfigManager()
