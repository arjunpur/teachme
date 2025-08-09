"""Pydantic models and schemas for teachme."""

from typing import Optional, List
from pydantic import BaseModel, model_validator


class AnimationInput(BaseModel):
    """Input schema for ManimCodeGenerator."""
    asset_prompt: str
    style: str = "light"


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


class AnimationStep(BaseModel):
    """Schema for individual animation steps."""
    step_number: int
    visual_description: str
    explanation: str
    key_insight: str


class TextOverlay(BaseModel):
    """Schema for text overlays in animations."""
    text: str
    timing_description: str
    purpose: str


class ExpandedPrompt(BaseModel):
    """Comprehensive animation specification from SubjectMatterAgent."""
    learning_objective: str
    key_concepts: List[str]
    visual_strategy: str
    animation_sequence: List[AnimationStep]
    explanatory_text: List[TextOverlay]
    quality_checklist: List[str]


# Simplified unified animation input
class AnimationRequest(BaseModel):
    """Unified input schema for animation generation with optional enhancement."""
    user_prompt: str
    enhance: bool = True  # Whether to use subject matter enhancement
    style: str = "light"
    
    def should_enhance(self) -> bool:
        """Check if the request should use subject matter enhancement."""
        return self.enhance