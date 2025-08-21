"""Pydantic models and schemas for teachme."""

from typing import Optional
from pydantic import BaseModel, model_validator


class AnimationOutput(BaseModel):
    """Output schema for ManimCodeGenerator."""
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
    review_notes: Optional[str] = None  # Optional field for code review
    confidence_score: Optional[float] = None  # Optional field for code review confidence


# SubjectMatterAgent schemas

class SubjectMatterInput(BaseModel):
    """Input schema for SubjectMatterAgent."""
    user_prompt: str


# Simplified unified animation input
class AnimationRequest(BaseModel):
    """Unified input for animation generation with optional subject-matter step."""
    user_prompt: str
    use_subject_matter: bool = True  # Whether to leverage SubjectMatterAgent
    style: str = "light"