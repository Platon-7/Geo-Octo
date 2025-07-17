# In scripts/configs/preprocess_libero_config.py

from ml_collections import ConfigDict

def get_config(config_string=None):
    """
    Simplified config for the VGGT preprocessing script.
    It only contains the parameters needed by this script.
    """
    return ConfigDict({
        "resize_size": (224, 224),  # The size for the primary camera view
        "window_size": 2,           # The number of frames per window
    })