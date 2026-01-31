"""
Utility functions for configuration management
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Manages application configuration from YAML file"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation)
        
        Args:
            key: Configuration key (e.g., 'model.name' or 'detection.vehicle_classes')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration"""
        return self.config.get('detection', {})
    
    def get_tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration"""
        return self.config.get('tracking', {})
    
    def get_checkpoint_config(self) -> Dict[str, Any]:
        """Get checkpoint configuration"""
        return self.config.get('checkpoint', {})
    
    def get_video_config(self) -> Dict[str, Any]:
        """Get video processing configuration"""
        return self.config.get('video', {})
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        paths = self.config.get('paths', {})
        for path in paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        if self.config.get('logging', {}).get('save_logs'):
            log_file = self.config.get('logging', {}).get('log_file', 'logs/app.log')
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)


# Global config instance
_config = None

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = ConfigManager()
    return _config
