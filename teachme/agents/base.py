"""Base agent class with utilities."""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize the agent with an output directory."""
        self.output_dir = output_dir or Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content based on input data."""
        pass
    
    def create_output_path(self, filename: str, subdir: str = "") -> Path:
        """Create a path for output files."""
        if subdir:
            output_path = self.output_dir / subdir
            output_path.mkdir(parents=True, exist_ok=True)
            return output_path / filename
        return self.output_dir / filename