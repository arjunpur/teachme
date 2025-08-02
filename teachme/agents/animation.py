"""AnimationGenerator agent implementation."""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from rich.console import Console
from ..agents.base import BaseAgent
from ..models.schemas import AnimationInput, AnimationOutput, ManimScriptResponse, EnhancedAnimationInput, ExpandedPrompt
from ..utils.llm_client import LLMClient
from ..utils.manim_runner import ManimRunner
from ..prompts.animation import ANIMATION_SYSTEM_PROMPT, create_animation_user_prompt, ERROR_CORRECTION_SYSTEM_PROMPT, create_error_correction_prompt

console = Console()

class AnimationGenerator(BaseAgent):
    """Agent for generating Manim animations from natural language prompts."""
    
    def __init__(self, output_dir: Path = None, llm_client: LLMClient = None):
        """Initialize the AnimationGenerator."""
        super().__init__(output_dir)
        self.llm_client = llm_client or LLMClient()
        self.manim_runner = ManimRunner()
        
        # Create subdirectories
        self.animations_dir = self.output_dir / "animations"
        self.animations_dir.mkdir(parents=True, exist_ok=True)
        
        # Create scripts subdirectory for saving successful scripts
        self.scripts_dir = self.output_dir / "scripts"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Track the last saved script path
        self.last_saved_script_path = None
    
    def _is_verbose(self) -> bool:
        """Check if verbose logging is enabled."""
        return hasattr(self.llm_client, 'verbose') and self.llm_client.verbose
    
    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an animation from the input data."""
        # Validate input - try EnhancedAnimationInput first, fallback to legacy AnimationInput
        try:
            enhanced_input = EnhancedAnimationInput(**input_data)
            animation_input = enhanced_input
        except:
            # Fallback to legacy AnimationInput for backward compatibility
            animation_input = AnimationInput(**input_data)
        
        try:
            # Check Manim installation
            is_installed, version_info = self.manim_runner.check_manim_installation()
            if not is_installed:
                raise RuntimeError(f"Manim installation check failed: {version_info}")
            
            # Generate and render animation with retry logic
            if hasattr(animation_input, 'expanded_prompt') and animation_input.expanded_prompt:
                # Use ExpandedPrompt flow
                script_response, video_path = await self._generate_and_render_with_expanded_prompt(
                    animation_input.expanded_prompt,
                    animation_input.style,
                    animation_input.quality
                )
            else:
                # Use direct prompt flow (legacy)
                prompt = animation_input.asset_prompt
                script_response, video_path = await self._generate_and_render_with_retry(
                    prompt,
                    animation_input.style,
                    animation_input.quality
                )
            
            # Create output
            output = AnimationOutput(
                video_path=str(video_path),
                alt_text=script_response.description,
                scene_name=script_response.scene_name,
                duration=script_response.estimated_duration
            )
            
            result = output.model_dump()
            
            # Add script path to result if available
            if self.last_saved_script_path:
                result["script_path"] = str(self.last_saved_script_path)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Animation generation failed: {e}")
    
    async def _generate_and_render_with_retry(self, prompt: str, style: str, quality: str) -> tuple[ManimScriptResponse, Path]:
        """Generate Manim script and render with retry logic for error correction."""
        max_attempts = 3
        script_response = await self._generate_manim_script(prompt, style)
        
        for attempt in range(1, max_attempts + 1):
            success, video_path, error_msg = await self.manim_runner.render_animation(
                script_response.code, script_response.scene_name, quality, self.animations_dir
            )
            
            if success:
                self.last_saved_script_path = await self._save_successful_script(script_response, prompt, attempt)
                return script_response, video_path
            
            # If this was the last attempt, give up
            if attempt == max_attempts:
                raise RuntimeError(f"Animation rendering failed after {max_attempts} attempts. Final error: {error_msg}")
            
            # Try to fix the script using LLM
            if self._is_verbose():
                console.print(f"[yellow]🔧 Attempt {attempt} failed. Trying to fix error with LLM...[/yellow]")
                console.print(f"[red]Error:[/red] {error_msg}")
            
            try:
                script_response = await self._fix_manim_script(script_response.code, error_msg, attempt + 1)
            except Exception as fix_error:
                raise RuntimeError(f"Failed to fix script on attempt {attempt}: {fix_error}")
        
        # Should never reach here due to the logic above
        raise RuntimeError("Unexpected error in retry loop")
    
    async def _generate_and_render_with_expanded_prompt(self, expanded_prompt: ExpandedPrompt, style: str, quality: str) -> tuple[ManimScriptResponse, Path]:
        """Generate Manim script from ExpandedPrompt and render with retry logic."""
        max_attempts = 3
        script_response = await self._generate_manim_script_from_expanded_prompt(expanded_prompt, style)
        
        for attempt in range(1, max_attempts + 1):
            success, video_path, error_msg = await self.manim_runner.render_animation(
                script_response.code, script_response.scene_name, quality, self.animations_dir
            )
            
            if success:
                self.last_saved_script_path = await self._save_successful_script(script_response, expanded_prompt.learning_objective, attempt)
                return script_response, video_path
            
            # If this was the last attempt, give up
            if attempt == max_attempts:
                raise RuntimeError(f"Animation rendering failed after {max_attempts} attempts. Final error: {error_msg}")
            
            # Try to fix the script using LLM
            if self._is_verbose():
                console.print(f"[yellow]🔧 Attempt {attempt} failed. Trying to fix error with LLM...[/yellow]")
                console.print(f"[red]Error:[/red] {error_msg}")
            
            try:
                script_response = await self._fix_manim_script(script_response.code, error_msg, attempt + 1)
            except Exception as fix_error:
                raise RuntimeError(f"Failed to fix script on attempt {attempt}: {fix_error}")
        
        # Should never reach here due to the logic above
        raise RuntimeError("Unexpected error in retry loop")
    
    async def _generate_manim_script(self, prompt: str, style: str) -> ManimScriptResponse:
        """Generate a Manim script using the LLM."""
        return await self._call_llm_for_script(
            ANIMATION_SYSTEM_PROMPT,
            create_animation_user_prompt(prompt, style),
            temperature=0.7,
            max_completion_tokens=20000,
            error_context="generate Manim script"
        )
    
    async def _generate_manim_script_from_expanded_prompt(self, expanded_prompt: ExpandedPrompt, style: str) -> ManimScriptResponse:
        """Generate a Manim script from an ExpandedPrompt using enhanced prompts."""
        from ..prompts.animation import create_enhanced_animation_user_prompt
        
        return await self._call_llm_for_script(
            ANIMATION_SYSTEM_PROMPT,
            create_enhanced_animation_user_prompt(expanded_prompt, style),
            temperature=0.7,
            max_completion_tokens=20000,
            error_context="generate Manim script from expanded prompt"
        )
    
    async def _fix_manim_script(self, original_code: str, error_message: str, attempt_number: int) -> ManimScriptResponse:
        """Fix a Manim script using the LLM based on an error message."""
        return await self._call_llm_for_script(
            ERROR_CORRECTION_SYSTEM_PROMPT,
            create_error_correction_prompt(original_code, error_message, attempt_number),
            temperature=0.3,
            max_completion_tokens=20000,
            error_context="fix Manim script"
        )
    
    async def _call_llm_for_script(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float, 
        max_completion_tokens: int,
        error_context: str
    ) -> ManimScriptResponse:
        """Call the LLM to generate or fix a Manim script."""
        try:
            response = await self.llm_client.generate_json_response(
                system_prompt, user_prompt, ManimScriptResponse,
                temperature=temperature, reasoning_effort="high",
                max_completion_tokens=max_completion_tokens
            )
            
            # Validate and fix scene name if needed
            extracted_scene = self.manim_runner.extract_scene_name(response.code)
            if not extracted_scene:
                raise ValueError("Generated code does not contain a valid Scene class")
            
            if extracted_scene != response.scene_name:
                response.scene_name = extracted_scene
            
            return response
            
        except Exception as e:
            raise RuntimeError(f"Failed to {error_context}: {e}")
    
    def _generate_script_filename(self, prompt: str, scene_name: str, attempt: int) -> str:
        """Generate a clear filename for the saved script."""
        # Clean the prompt to create a readable filename
        # Remove special characters and limit length
        clean_prompt = re.sub(r'[^\w\s-]', '', prompt.lower())
        clean_prompt = re.sub(r'\s+', '_', clean_prompt.strip())
        
        # Limit length to avoid filesystem issues
        if len(clean_prompt) > 50:
            clean_prompt = clean_prompt[:50].rstrip('_')
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include attempt number if it's not the first attempt
        attempt_suffix = f"_attempt{attempt}" if attempt > 1 else ""
        
        # Combine all parts
        filename = f"{timestamp}_{clean_prompt}_{scene_name}{attempt_suffix}.py"
        
        return filename
    
    async def _save_successful_script(self, script_response: ManimScriptResponse, prompt: str, attempt: int) -> Optional[Path]:
        """Save a successful Manim script to the scripts directory."""
        try:
            filename = self._generate_script_filename(prompt, script_response.scene_name, attempt)
            script_path = self.scripts_dir / filename
            
            # Create a header comment with metadata
            header = f'''"""
Manim Script: {script_response.scene_name}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Prompt: {prompt}
Scene: {script_response.scene_name}
Description: {script_response.description}
Duration: {script_response.estimated_duration}s
Attempt: {attempt}/3
"""

'''
            
            # Write the script with header
            full_content = header + script_response.code
            
            script_path.write_text(full_content, encoding='utf-8')
            
            if self._is_verbose():
                console.print(f"[green]💾 Script saved:[/green] {script_path}")
            
            return script_path
            
        except Exception as e:
            # Don't fail the entire process if script saving fails
            console.print(f"[yellow]⚠️  Failed to save script: {e}[/yellow]")
            return None