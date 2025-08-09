"""Base agent class with utilities."""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from ..config import PathConfig


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize the agent with an output directory."""
        self.output_dir = output_dir or PathConfig.DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content based on input data."""
        pass

