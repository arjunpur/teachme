"""ManimCodeGenerator agent implementation."""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from rich.console import Console
from ..agents.base import BaseAgent
from ..models.schemas import AnimationOutput, ManimScriptResponse, AnimationRequest, ExpandedPrompt
from ..utils.responses_llm_client import ResponsesLLMClient
from ..utils.manim_runner import ManimRunner
from ..config import RenderConfig, LLMConfig, AnimationConfig
from ..exceptions import ManimInstallationError, AnimationRenderError
from ..prompts.animation import ANIMATION_SYSTEM_PROMPT, create_animation_user_prompt, ERROR_CORRECTION_SYSTEM_PROMPT, create_error_correction_prompt, CODE_REVIEW_SYSTEM_PROMPT, create_code_review_prompt

console = Console()

class ManimCodeGenerator(BaseAgent):
    """Agent for generating Manim animations from natural language prompts."""
    
    def __init__(self, output_dir: Path = None, llm_client: ResponsesLLMClient = None, verbose: bool = False):
        """Initialize the ManimCodeGenerator."""
        super().__init__(output_dir)
        self.llm_client = llm_client or ResponsesLLMClient()
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
        prompt_for_generation, subject_matter_response_id = await self._create_prompt(request)
        
        # Generate initial script (start of conversation chain from subject matter agent)
        script_result = await self._generate_manim_script(prompt_for_generation, request.style, subject_matter_response_id)
        script_response = script_result.content
        current_response_id = script_result.response_id
        
        # Apply code review to improve the initial script (chain from previous response)
        review_result = await self._review_manim_script(script_response, current_response_id)
        script_response = review_result.content
        # Preserve chaining if review step returns an empty response_id (fallback path)
        current_response_id = review_result.response_id or current_response_id
        
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
                # Log the exact error from Manim execution in red for clarity
                console.print(f"[red]Manim execution error (attempt {attempt}/{max_attempts}):[/red]")
                console.print(error_msg, style="red")
                console.print(f"[yellow]🔧 Attempt {attempt} failed. Trying to fix error with LLM...[/yellow]")
            
            try:
                fix_result = await self._fix_manim_script(
                    script_response.code, error_msg, attempt + 1, current_response_id
                )
                script_response = fix_result.content
                # Ensure we keep a valid chain by not overwriting with empty IDs
                current_response_id = fix_result.response_id or current_response_id
            except Exception as fix_error:
                from ..exceptions import LLMGenerationError
                raise LLMGenerationError(
                    f"Failed to fix script on attempt {attempt}: {fix_error}",
                    model=self.llm_client.model,
                    prompt_type="error_correction"
                ) from fix_error
        
        # Should never reach here
        raise AnimationRenderError("Unexpected error in retry loop")
    
    async def _create_prompt(self, request: AnimationRequest) -> tuple[str, Optional[str]]:
        """Create the appropriate prompt for script generation.
        
        Returns:
            tuple: (prompt_text, response_id_for_chaining)
        """
        if not request.should_enhance():
            # Direct prompt - format with basic animation prompt
            return create_animation_user_prompt(request.user_prompt, request.style), None
        
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
        
        # Generate expanded brief text (single-step)
        expansion_output = await subject_matter_agent.generate({
            "user_prompt": request.user_prompt
        })

        response_id = expansion_output.get("_response_id")
        brief_text = expansion_output["expanded_prompt_text"]

        # Wrap brief for the code generator
        from ..prompts.animation import create_animation_prompt_from_brief
        return create_animation_prompt_from_brief(brief_text, request.style), response_id
    
    async def _generate_manim_script(self, prompt: str, style: str, previous_response_id: Optional[str] = None):
        """Generate a Manim script using the LLM."""
        return await self._call_llm_for_script(
            ANIMATION_SYSTEM_PROMPT,
            prompt,
            temperature=LLMConfig.GENERATION_TEMPERATURE,
            max_completion_tokens=LLMConfig.MAX_COMPLETION_TOKENS,
            error_context="generate Manim script",
            previous_response_id=previous_response_id
        )
    
    async def _review_manim_script(self, script_response: ManimScriptResponse, previous_response_id: Optional[str] = None):
        """Review and improve a Manim script using the code review agent."""
        try:
            reviewed_result = await self._call_llm_for_script(
                CODE_REVIEW_SYSTEM_PROMPT,
                create_code_review_prompt(script_response.code, script_response.scene_name, script_response.description),
                temperature=LLMConfig.REVIEW_TEMPERATURE,
                max_completion_tokens=LLMConfig.MAX_COMPLETION_TOKENS,
                error_context="review Manim script",
                previous_response_id=previous_response_id
            )
            reviewed_response = reviewed_result.content
            
            if self._is_verbose():
                console.print(f"[blue]🔍 Code review completed with confidence: {getattr(reviewed_response, 'confidence_score', 'N/A')}[/blue]")
                review_notes = getattr(reviewed_response, 'review_notes', 'No review notes available')
                console.print(f"[blue]📝 Review notes: {review_notes}[/blue]")
            
            return reviewed_result
            
        except Exception as e:
            if self._is_verbose():
                console.print(f"[yellow]⚠️  Code review failed, using original script: {e}[/yellow]")
            # Return original script with a dummy response ID
            from ..utils.responses_llm_client import ResponseResult
            # Preserve previous_response_id if provided so chaining continues
            return ResponseResult(content=script_response, response_id=(previous_response_id or ""), usage=None)

    async def _fix_manim_script(self, original_code: str, error_message: str, attempt_number: int, previous_response_id: Optional[str] = None):
        """Fix a Manim script using the LLM based on an error message."""
        return await self._call_llm_for_script(
            ERROR_CORRECTION_SYSTEM_PROMPT,
            create_error_correction_prompt(original_code, error_message, attempt_number),
            temperature=LLMConfig.ERROR_CORRECTION_TEMPERATURE,
            max_completion_tokens=LLMConfig.MAX_COMPLETION_TOKENS,
            error_context="fix Manim script",
            previous_response_id=previous_response_id
        )
    
    async def _call_llm_for_script(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float, 
        max_completion_tokens: int,
        error_context: str,
        previous_response_id: Optional[str] = None
    ):
        """Call the LLM to generate or fix a Manim script."""
        try:
            # Use the new generate method to get ResponseResult with response ID
            result = await self.llm_client.generate(
                input=user_prompt,
                instructions=system_prompt,
                response_format=ManimScriptResponse,
                previous_response_id=previous_response_id,
                return_response_id=True,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens
            )
            
            response = result.content
            
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
            
            # Update the result with the potentially modified response
            result.content = response
            return result
            
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