"""Custom exception hierarchy for TeachMe application."""

from typing import Optional


class TeachMeError(Exception):
    """Base exception for all TeachMe errors."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None, context: Optional[dict] = None):
        """Initialize with message, optional suggestion, and context."""
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.context = context or {}
    
    def __str__(self) -> str:
        """Return formatted error message with suggestion if available."""
        error_msg = self.message
        if self.suggestion:
            error_msg += f"\n💡 Suggestion: {self.suggestion}"
        return error_msg


class ManimInstallationError(TeachMeError):
    """Raised when Manim installation issues are detected."""
    
    def __init__(self, message: str, version_info: Optional[str] = None):
        suggestion = "Please ensure Manim Community Edition is installed: pip install manim"
        context = {"version_info": version_info} if version_info else {}
        super().__init__(message, suggestion, context)


class AnimationRenderError(TeachMeError):
    """Raised when animation rendering fails."""
    
    def __init__(self, message: str, attempt: int = 1, max_attempts: int = 5, 
                 scene_name: Optional[str] = None, error_output: Optional[str] = None):
        suggestion = None
        if attempt < max_attempts:
            suggestion = f"Retrying with error correction (attempt {attempt + 1}/{max_attempts})"
        elif "timeout" in message.lower():
            suggestion = "Try simplifying the animation or increasing timeout"
        elif "syntax" in message.lower() or "name" in message.lower():
            suggestion = "Check for syntax errors or undefined variables in the generated code"
        
        context = {
            "attempt": attempt,
            "max_attempts": max_attempts,
            "scene_name": scene_name,
            "error_output": error_output
        }
        super().__init__(message, suggestion, context)


class LLMGenerationError(TeachMeError):
    """Raised when LLM fails to generate valid responses."""
    
    def __init__(self, message: str, model: Optional[str] = None, 
                 prompt_type: Optional[str] = None, response_content: Optional[str] = None):
        suggestion = None
        if "json" in message.lower():
            suggestion = "The model returned invalid JSON. This may be a temporary issue - try again"
        elif "token" in message.lower():
            suggestion = "Try reducing the complexity of your request or breaking it into smaller parts"
        
        context = {
            "model": model,
            "prompt_type": prompt_type,
            "response_content": response_content[:500] if response_content else None
        }
        super().__init__(message, suggestion, context)


class ScriptValidationError(TeachMeError):
    """Raised when generated scripts fail validation."""
    
    def __init__(self, message: str, validation_type: Optional[str] = None, 
                 code_snippet: Optional[str] = None):
        suggestion = None
        if "scene" in message.lower():
            suggestion = "Ensure the code contains exactly one Scene class that inherits from manim.Scene"
        elif "import" in message.lower():
            suggestion = "Check that all required imports are present and correct"
        
        context = {
            "validation_type": validation_type,
            "code_snippet": code_snippet[:200] if code_snippet else None
        }
        super().__init__(message, suggestion, context)


class SubjectMatterAnalysisError(TeachMeError):
    """Raised when subject matter analysis fails."""
    
    def __init__(self, message: str, stage: Optional[str] = None, user_prompt: Optional[str] = None):
        suggestion = None
        if "timeout" in message.lower():
            suggestion = "Try simplifying your prompt or breaking it into smaller concepts"
        elif stage:
            suggestion = f"The error occurred during {stage}. Try rephrasing your request"
        
        context = {
            "stage": stage,
            "user_prompt": user_prompt[:100] if user_prompt else None
        }
        super().__init__(message, suggestion, context)


class ConfigurationError(TeachMeError):
    """Raised when configuration issues are detected."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, expected_type: Optional[str] = None):
        suggestion = "Check your environment variables and configuration settings"
        if config_key:
            suggestion = f"Ensure {config_key} is properly set"
            if expected_type:
                suggestion += f" and is of type {expected_type}"
        
        context = {
            "config_key": config_key,
            "expected_type": expected_type
        }
        super().__init__(message, suggestion, context)