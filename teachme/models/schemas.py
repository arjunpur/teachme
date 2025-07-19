"""Pydantic models and schemas for teachme."""

from typing import Optional
from pydantic import BaseModel


class AnimationInput(BaseModel):
    """Input schema for AnimationGenerator."""
    asset_prompt: str
    style: str = "light"
    quality: str = "low"


class AnimationOutput(BaseModel):
    """Output schema for AnimationGenerator."""
    video_path: str
    alt_text: str
    scene_name: str
    duration: float


class ManimScriptResponse(BaseModel):
    """LLM response schema for Manim script generation."""
    filename: str = "scene.py"
    scene_name: str
    description: str
    code: str
    estimated_duration: float
    fix_description: Optional[str] = None  # Optional field for error correction