"""ManimCodeGenerator agent implementation."""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from rich.console import Console
from ..agents.base import BaseAgent
from ..models.schemas import AnimationOutput, ManimScriptResponse, AnimationRequest, ExpandedPrompt
from ..utils.llm_client import LLMClient
from ..utils.manim_runner import ManimRunner
from ..config import RenderConfig, LLMConfig, AnimationConfig
from ..exceptions import ManimInstallationError, AnimationRenderError
from ..prompts.animation import ANIMATION_SYSTEM_PROMPT, create_animation_user_prompt, ERROR_CORRECTION_SYSTEM_PROMPT, create_error_correction_prompt, CODE_REVIEW_SYSTEM_PROMPT, create_code_review_prompt

console = Console()

class ManimCodeGenerator(BaseAgent):
    """Agent for generating Manim animations from natural language prompts."""
    
    def __init__(self, output_dir: Path = None, llm_client: LLMClient = None, verbose: bool = False):
        """Initialize the ManimCodeGenerator."""
        super().__init__(output_dir)
        self.llm_client = llm_client or LLMClient()
        self.manim_runner = ManimRunner()
        self.verbose = verbose
        
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
        return self.verbose
    
    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an animation from the input data."""
        # Validate input using simplified AnimationRequest
        request = AnimationRequest(**input_data)
        
        try:
            # Check Manim installation
            is_installed, version_info = self.manim_runner.check_manim_installation()
            if not is_installed:
                raise ManimInstallationError(
                    "Manim installation check failed",
                    version_info=version_info
                )
            
            # Generate and render animation with simplified retry logic
            quality = "low"  # Fixed quality for now
            script_response, video_path = await self._generate_and_render_with_retry(
                request, quality
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
            
        except (ManimInstallationError, AnimationRenderError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            raise AnimationRenderError(
                f"Animation generation failed: {e}"
            ) from e
    
    async def _generate_and_render_with_retry(
        self, 
        request: AnimationRequest, 
        quality: str
    ) -> tuple[ManimScriptResponse, Path]:
        """Generate and render animation with simplified retry logic."""
        
        # Create the appropriate prompt based on enhance flag
        prompt_for_generation = await self._create_prompt(request)
        
        # Generate initial script
        script_response = await self._generate_manim_script(prompt_for_generation, request.style)
        
        # Apply code review to improve the initial script
        script_response = await self._review_manim_script(script_response)
        
        # Try rendering with retry logic
        max_attempts = RenderConfig.MAX_RETRY_ATTEMPTS
        for attempt in range(1, max_attempts + 1):
            success, video_path, error_msg = await self.manim_runner.render_animation(
                script_response.code, script_response.scene_name, quality, self.animations_dir
            )
            
            if success:
                # Save successful script and return
                self.last_saved_script_path = await self._save_successful_script(
                    script_response, request.user_prompt, attempt
                )
                return script_response, video_path
            
            # If this was the last attempt, give up
            if attempt == max_attempts:
                raise AnimationRenderError(
                    f"Animation rendering failed after {max_attempts} attempts",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    scene_name=script_response.scene_name,
                    error_output=error_msg
                )
            
            # Try to fix the script using LLM for next attempt
            if self._is_verbose():
                console.print(f"[yellow]🔧 Attempt {attempt} failed. Trying to fix error with LLM...[/yellow]")
                console.print(f"[red]Error:[/red] {error_msg}")
            
            try:
                script_response = await self._fix_manim_script(
                    script_response.code, error_msg, attempt + 1
                )
            except Exception as fix_error:
                from ..exceptions import LLMGenerationError
                raise LLMGenerationError(
                    f"Failed to fix script on attempt {attempt}: {fix_error}",
                    model=self.llm_client.model,
                    prompt_type="error_correction"
                ) from fix_error
        
        # Should never reach here
        raise AnimationRenderError("Unexpected error in retry loop")
    
    async def _create_prompt(self, request: AnimationRequest) -> str:
        """Create the appropriate prompt for script generation."""
        if not request.should_enhance():
            # Direct prompt - format with basic animation prompt
            return create_animation_user_prompt(request.user_prompt, request.style)
        
        # Enhanced prompt - use subject matter agent to expand
        if self._is_verbose():
            console.print("[blue]🧠 Enhancing prompt with subject matter analysis...[/blue]")
        
        # Import here to avoid circular imports
        from ..agents.subject_matter import SubjectMatterAgent
        
        subject_matter_agent = SubjectMatterAgent(
            output_dir=self.output_dir,
            llm_client=self.llm_client,
            verbose=self.verbose
        )
        
        # Generate expanded prompt
        expanded_result = await subject_matter_agent.generate({
            "user_prompt": request.user_prompt
        })
        
        expanded_prompt = ExpandedPrompt(**expanded_result)
        
        # Convert expanded prompt to enhanced animation prompt
        from ..prompts.animation import create_enhanced_animation_user_prompt
        return create_enhanced_animation_user_prompt(expanded_prompt, request.style)
    
    async def _generate_manim_script(self, prompt: str, style: str) -> ManimScriptResponse:
        """Generate a Manim script using the LLM."""
        return await self._call_llm_for_script(
            ANIMATION_SYSTEM_PROMPT,
            prompt,
            temperature=LLMConfig.GENERATION_TEMPERATURE,
            max_completion_tokens=LLMConfig.MAX_COMPLETION_TOKENS,
            error_context="generate Manim script"
        )
    
    async def _review_manim_script(self, script_response: ManimScriptResponse) -> ManimScriptResponse:
        """Review and improve a Manim script using the code review agent."""
        try:
            reviewed_response = await self._call_llm_for_script(
                CODE_REVIEW_SYSTEM_PROMPT,
                create_code_review_prompt(script_response.code, script_response.scene_name, script_response.description),
                temperature=LLMConfig.REVIEW_TEMPERATURE,
                max_completion_tokens=LLMConfig.MAX_COMPLETION_TOKENS,
                error_context="review Manim script"
            )
            
            if self._is_verbose():
                console.print(f"[blue]🔍 Code review completed with confidence: {getattr(reviewed_response, 'confidence_score', 'N/A')}[/blue]")
                review_notes = getattr(reviewed_response, 'review_notes', 'No review notes available')
                console.print(f"[blue]📝 Review notes: {review_notes}[/blue]")
            
            return reviewed_response
            
        except Exception as e:
            if self._is_verbose():
                console.print(f"[yellow]⚠️  Code review failed, using original script: {e}[/yellow]")
            return script_response

    async def _fix_manim_script(self, original_code: str, error_message: str, attempt_number: int) -> ManimScriptResponse:
        """Fix a Manim script using the LLM based on an error message."""
        return await self._call_llm_for_script(
            ERROR_CORRECTION_SYSTEM_PROMPT,
            create_error_correction_prompt(original_code, error_message, attempt_number),
            temperature=LLMConfig.ERROR_CORRECTION_TEMPERATURE,
            max_completion_tokens=LLMConfig.MAX_COMPLETION_TOKENS,
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
                temperature=temperature, reasoning_effort=LLMConfig.HIGH_REASONING_EFFORT,
                max_completion_tokens=max_completion_tokens
            )
            
            # Validate and fix scene name if needed
            extracted_scene = self.manim_runner.extract_scene_name(response.code)
            if not extracted_scene:
                from ..exceptions import ScriptValidationError
                raise ScriptValidationError(
                    "Generated code does not contain a valid Scene class",
                    validation_type="scene_class",
                    code_snippet=response.code
                )
            
            if extracted_scene != response.scene_name:
                response.scene_name = extracted_scene
            
            return response
            
        except Exception as e:
            from ..exceptions import LLMGenerationError
            raise LLMGenerationError(
                f"Failed to {error_context}: {e}",
                model=self.llm_client.model,
                prompt_type=error_context
            ) from e
    
    def _generate_script_filename(self, prompt: str, scene_name: str, attempt: int) -> str:
        """Generate a clear filename for the saved script."""
        # Clean the prompt to create a readable filename
        # Remove special characters and limit length
        clean_prompt = re.sub(r'[^\w\s-]', '', prompt.lower())
        clean_prompt = re.sub(r'\s+', '_', clean_prompt.strip())
        
        # Limit length to avoid filesystem issues
        if len(clean_prompt) > AnimationConfig.MAX_FILENAME_LENGTH:
            clean_prompt = clean_prompt[:AnimationConfig.MAX_FILENAME_LENGTH].rstrip('_')
        
        # Generate timestamp
        timestamp = datetime.now().strftime(AnimationConfig.TIMESTAMP_FORMAT)
        
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