"""SubjectMatterAgent implementation."""

import asyncio
from typing import Dict, Any
from pathlib import Path

from rich.console import Console
from ..agents.base import BaseAgent
from ..models.schemas import (
    SubjectMatterInput, 
    ExpandedPrompt,
    AnimationStep,
    TextOverlay
)
from ..utils.llm_client import LLMClient
from ..prompts.subject_matter import (
    CONTENT_ANALYSIS_SYSTEM_PROMPT,
    VISUAL_PLANNING_SYSTEM_PROMPT, 
    SEQUENCE_GENERATION_SYSTEM_PROMPT,
    create_content_analysis_prompt,
    create_visual_planning_prompt,
    create_sequence_generation_prompt
)

console = Console()

class SubjectMatterAgent(BaseAgent):
    """Agent for transforming user prompts into detailed animation instructions."""
    
    def __init__(self, output_dir: Path = None, llm_client: LLMClient = None):
        """Initialize the SubjectMatterAgent."""
        super().__init__(output_dir)
        self.llm_client = llm_client or LLMClient()
    
    def _is_verbose(self) -> bool:
        """Check if verbose logging is enabled."""
        return hasattr(self.llm_client, 'verbose') and self.llm_client.verbose
    
    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an expanded prompt from user input through 3-stage LLM processing."""
        # Validate input
        subject_matter_input = SubjectMatterInput(**input_data)
        
        try:
            # Stage 1: Content Analysis
            if self._is_verbose():
                console.print("[blue]Stage 1: Analyzing content and learning objectives...[/blue]")
            
            content_analysis = await self._analyze_content(subject_matter_input.user_prompt)
            
            # Stage 2: Visual Planning
            if self._is_verbose():
                console.print("[blue]Stage 2: Designing visualization strategy...[/blue]")
            
            visual_planning = await self._plan_visuals(content_analysis)
            
            # Stage 3: Sequence Generation
            if self._is_verbose():
                console.print("[blue]Stage 3: Creating step-by-step animation sequence...[/blue]")
            
            sequence_data = await self._generate_sequence(content_analysis, visual_planning)
            
            # Construct ExpandedPrompt
            expanded_prompt = ExpandedPrompt(
                learning_objective=content_analysis["learning_objective"],
                key_concepts=content_analysis["key_concepts"],
                visual_strategy=visual_planning["visual_strategy"],
                animation_sequence=[
                    AnimationStep(**step) for step in sequence_data["animation_sequence"]
                ],
                explanatory_text=[
                    TextOverlay(**text) for text in sequence_data["explanatory_text"]
                ],
                quality_checklist=sequence_data["quality_checklist"]
            )
            
            if self._is_verbose():
                console.print(f"[green]✓ Generated expanded prompt with {len(expanded_prompt.animation_sequence)} steps[/green]")
                console.print(f"[green]✓ Learning objective: {expanded_prompt.learning_objective[:100]}...[/green]")
            
            return expanded_prompt.model_dump()
            
        except Exception as e:
            raise RuntimeError(f"Subject matter analysis failed: {e}")
    
    async def _analyze_content(self, user_prompt: str) -> Dict[str, Any]:
        """Stage 1: Analyze content to identify core concepts and learning objectives."""
        try:
            # Custom response model for content analysis
            from pydantic import BaseModel
            from typing import List
            
            class ContentAnalysisResponse(BaseModel):
                learning_objective: str
                key_concepts: List[str]
                prerequisite_knowledge: List[str]
                common_misconceptions: List[str]
                difficulty_level: str
            
            response = await self.llm_client.generate_json_response(
                CONTENT_ANALYSIS_SYSTEM_PROMPT,
                create_content_analysis_prompt(user_prompt),
                ContentAnalysisResponse,
                temperature=0.3,
                reasoning_effort="medium",
                max_completion_tokens=2000
            )
            
            return response.model_dump()
            
        except Exception as e:
            raise RuntimeError(f"Content analysis failed: {e}")
    
    async def _plan_visuals(self, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Design visual metaphors and visualization strategy."""
        try:
            from pydantic import BaseModel
            from typing import List, Dict
            
            class VisualPlanningResponse(BaseModel):
                visual_strategy: str
                visual_metaphors: List[str]
                color_scheme: Dict[str, str]
                progression_strategy: str
                misconception_corrections: List[str]
                key_visual_techniques: List[str]
            
            response = await self.llm_client.generate_json_response(
                VISUAL_PLANNING_SYSTEM_PROMPT,
                create_visual_planning_prompt(content_analysis),
                VisualPlanningResponse,
                temperature=0.4,
                reasoning_effort="medium",
                max_completion_tokens=2000
            )
            
            return response.model_dump()
            
        except Exception as e:
            raise RuntimeError(f"Visual planning failed: {e}")
    
    async def _generate_sequence(self, content_analysis: Dict[str, Any], visual_planning: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Create step-by-step animation breakdown with quality requirements."""
        try:
            from pydantic import BaseModel
            from typing import List, Dict
            
            class AnimationStepResponse(BaseModel):
                step_number: int
                visual_description: str
                explanation: str
                key_insight: str
            
            class TextOverlayResponse(BaseModel):
                text: str
                timing_description: str
                purpose: str
            
            class SequenceGenerationResponse(BaseModel):
                animation_sequence: List[AnimationStepResponse]
                explanatory_text: List[TextOverlayResponse]
                quality_checklist: List[str]
                total_estimated_duration: float
                pacing_notes: str
            
            response = await self.llm_client.generate_json_response(
                SEQUENCE_GENERATION_SYSTEM_PROMPT,
                create_sequence_generation_prompt(content_analysis, visual_planning),
                SequenceGenerationResponse,
                temperature=0.3,
                reasoning_effort="high",
                max_completion_tokens=4000
            )
            
            # Convert to dict format expected by ExpandedPrompt
            sequence_dict = response.model_dump()
            
            # Convert nested models to dicts
            sequence_dict["animation_sequence"] = [
                step.model_dump() for step in response.animation_sequence
            ]
            sequence_dict["explanatory_text"] = [
                text.model_dump() for text in response.explanatory_text
            ]
            
            return sequence_dict
            
        except Exception as e:
            raise RuntimeError(f"Sequence generation failed: {e}")
    
    async def process_with_timeout(self, user_prompt: str, timeout_seconds: int = 90) -> ExpandedPrompt:
        """Process user prompt with timeout, returning ExpandedPrompt object."""
        try:
            # Use asyncio.wait_for to implement timeout
            result = await asyncio.wait_for(
                self.generate({"user_prompt": user_prompt}),
                timeout=timeout_seconds
            )
            
            return ExpandedPrompt(**result)
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Subject matter processing timed out after {timeout_seconds} seconds")
        except Exception as e:
            raise RuntimeError(f"Subject matter processing failed: {e}")