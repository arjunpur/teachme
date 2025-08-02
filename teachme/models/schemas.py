"""Pydantic models and schemas for teachme."""

from typing import Optional, List
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


# Enhanced AnimationInput to support both direct prompts and ExpandedPrompt
class EnhancedAnimationInput(BaseModel):
    """Enhanced input schema that supports both direct prompts and expanded prompts."""
    # Either asset_prompt OR expanded_prompt should be provided
    asset_prompt: Optional[str] = None
    expanded_prompt: Optional[ExpandedPrompt] = None
    style: str = "light"
    quality: str = "low"
    
    def model_post_init(self, __context):
        """Validate that exactly one of asset_prompt or expanded_prompt is provided."""
        if not self.asset_prompt and not self.expanded_prompt:
            raise ValueError("Either asset_prompt or expanded_prompt must be provided")
        if self.asset_prompt and self.expanded_prompt:
            raise ValueError("Only one of asset_prompt or expanded_prompt should be provided")