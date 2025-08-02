"""Configuration constants for TeachMe application."""

from pathlib import Path


class RenderConfig:
    """Configuration for animation rendering."""
    
    # Retry settings
    MAX_RETRY_ATTEMPTS = 5
    INITIAL_RETRY_DELAY = 1.0  # seconds
    MAX_RETRY_DELAY = 30.0     # seconds
    BACKOFF_MULTIPLIER = 2.0
    
    # Timeout settings  
    RENDER_TIMEOUT = 300       # seconds (5 minutes)
    LLM_TIMEOUT = 120         # seconds (2 minutes)
    SUBJECT_MATTER_TIMEOUT = 90  # seconds
    
    # Quality settings
    DEFAULT_QUALITY = "low"
    QUALITY_FLAGS = {
        "low": ["-ql"],
        "medium": ["-qm"], 
        "high": ["-qh"]
    }


class LLMConfig:
    """Configuration for LLM client."""
    
    # Token limits
    MAX_COMPLETION_TOKENS = 20000
    DEFAULT_MAX_TOKENS = 4000
    
    # Temperature settings
    DEFAULT_TEMPERATURE = 0.7
    GENERATION_TEMPERATURE = 0.7
    REVIEW_TEMPERATURE = 0.2
    ERROR_CORRECTION_TEMPERATURE = 0.3
    CONTENT_ANALYSIS_TEMPERATURE = 0.3
    VISUAL_PLANNING_TEMPERATURE = 0.4
    
    # Reasoning effort for o3 models
    DEFAULT_REASONING_EFFORT = "medium"
    HIGH_REASONING_EFFORT = "high"
    
    # Model settings
    DEFAULT_MODEL = "o3"
    FALLBACK_MODEL = "gpt-4o"


class AnimationConfig:
    """Configuration for animation generation."""
    
    # Duration settings
    MIN_DURATION = 15.0        # seconds
    MAX_DURATION = 30.0        # seconds
    DEFAULT_DURATION = 20.0    # seconds
    
    # Style settings
    DEFAULT_STYLE = "light"
    SUPPORTED_STYLES = ["light", "dark"]
    
    # File naming
    MAX_FILENAME_LENGTH = 50
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


class PathConfig:
    """Configuration for file paths."""
    
    # Default directories
    DEFAULT_OUTPUT_DIR = Path("./outputs")
    ANIMATIONS_SUBDIR = "animations"
    SCRIPTS_SUBDIR = "scripts"
    
    # File extensions
    SCRIPT_EXTENSION = ".py"
    VIDEO_EXTENSION = ".mp4"
    
    # Temporary file settings
    TEMP_SCRIPT_NAME = "scene.py"


class ValidationConfig:
    """Configuration for code validation."""
    
    # Dangerous operations to check for
    DANGEROUS_IMPORTS = ["os", "subprocess", "sys", "shutil"]
    DANGEROUS_FUNCTIONS = ["open", "exec", "eval", "__import__"]
    
    # Valid scene base classes
    VALID_SCENE_CLASSES = ["Scene", "MovingCameraScene", "ThreeDScene"]
    
    # Content limits
    MAX_PREVIEW_LENGTH = 500   # characters
    MAX_CONTEXT_LENGTH = 200   # characters


class LoggingConfig:
    """Configuration for logging and output."""
    
    # Console output settings
    MAX_CONTENT_PREVIEW = 300  # characters
    MAX_ERROR_CONTEXT = 500    # characters
    
    # Progress indicators
    PROGRESS_SYMBOLS = {
        "success": "✅",
        "error": "❌", 
        "warning": "⚠️",
        "info": "💡",
        "working": "🔧",
        "save": "💾",
        "review": "🔍"
    }