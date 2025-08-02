"""OpenAI client wrapper for LLM interactions."""

import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from rich.console import Console

# Load environment variables from .env file
load_dotenv()

console = Console()


class LLMClient:
    """Wrapper for OpenAI API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = None, verbose: bool = False):
        """Initialize the LLM client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.model = model or self._determine_best_model()
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.verbose = verbose
        
        # Model-specific configuration
        self.is_o3_model = self.model.startswith("o3")
        self.is_o1_model = self.model.startswith("o1")
    
    def _determine_best_model(self) -> str:
        """Determine the best available model from environment or fallback to gpt-4o."""
        # Check environment variable for preferred model
        preferred_model = os.getenv("TEACHME_MODEL")
        if preferred_model:
            return preferred_model
        
        # Default to gpt-4o (most reliable for JSON responses)
        return "gpt-4o"
    
    def _build_request_params(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        max_completion_tokens: int,
        temperature: float = 0.7,
        reasoning_effort: str = "medium",
        response_format: dict = None
    ) -> dict:
        """Build request parameters for the API call."""
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_completion_tokens": max_completion_tokens
        }
        
        # Add temperature only for non-o3/o1 models
        if not (self.is_o3_model or self.is_o1_model):
            params["temperature"] = temperature
        
        # Add reasoning_effort for o3 models
        if self.is_o3_model:
            params["reasoning_effort"] = reasoning_effort
            
        # Add response format if specified
        if response_format:
            params["response_format"] = response_format
            
        return params
    
    def _log_request_info(self, max_completion_tokens: int, reasoning_effort: str = None) -> None:
        """Log request information if verbose mode is enabled."""
        console.print(f"[dim]🤖 Using model:[/dim] [bold blue]{self.model}[/bold blue]")
        console.print(f"[dim]📝 Max tokens:[/dim] {max_completion_tokens}")
        if self.is_o3_model and reasoning_effort:
            console.print(f"[dim]🧠 Reasoning effort:[/dim] {reasoning_effort}")
    
    def _log_response_info(self, response, content: str) -> None:
        """Log response information if verbose mode is enabled."""
        usage = response.usage
        console.print(f"[dim]📊 Token usage:[/dim] {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total")
        if hasattr(usage.completion_tokens_details, 'reasoning_tokens') and usage.completion_tokens_details.reasoning_tokens:
            console.print(f"[dim]🧠 Reasoning tokens:[/dim] {usage.completion_tokens_details.reasoning_tokens}")
        
        console.print(f"[dim]✅ Response length:[/dim] {len(content)} characters")
        preview = content[:300].replace('\n', '\\n')
        if len(content) > 300:
            preview += "..."
        console.print(f"[dim]📄 Preview:[/dim] {preview}")
    
    def _handle_empty_response(self, choice, max_completion_tokens: int) -> None:
        """Handle empty response from the LLM."""
        finish_reason = choice.finish_reason
        if self.verbose:
            console.print(f"[red]❌ Empty response! Finish reason:[/red] {finish_reason}")
        
        if finish_reason == 'length':
            raise ValueError(f"Response was truncated due to token limit ({max_completion_tokens} tokens). Try increasing max_completion_tokens or simplifying the prompt.")
        else:
            raise ValueError(f"Empty response from LLM. Finish reason: {finish_reason}")
    
    async def generate_json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: BaseModel,
        temperature: float = 0.7,
        reasoning_effort: str = "medium",
        max_completion_tokens: int = 4000
    ) -> BaseModel:
        """Generate a structured JSON response from the LLM."""
        try:
            params = self._build_request_params(
                system_prompt, user_prompt, max_completion_tokens,
                temperature, reasoning_effort, {"type": "json_object"}
            )
            
            if self.verbose:
                self._log_request_info(max_completion_tokens, reasoning_effort)
                
            response = await self.client.chat.completions.create(**params)
            
            choice = response.choices[0]
            content = choice.message.content
            
            if not content:
                self._handle_empty_response(choice, max_completion_tokens)
            
            if self.verbose:
                self._log_response_info(response, content)
            
            # Parse JSON and validate with Pydantic model
            json_data = json.loads(content)
            return response_model(**json_data)
            
        except json.JSONDecodeError as e:
            if self.verbose:
                # Show the problematic content with proper formatting
                content_preview = content[:500] if content else "None"
                console.print(f"[red]❌ JSON Parse Error:[/red] {e}")
                console.print(f"[dim]📄 Content that failed to parse:[/dim]")
                console.print(f"[dim]{content_preview}[/dim]")
            raise ValueError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            if self.verbose:
                console.print(f"[red]❌ API Error:[/red] {type(e).__name__}: {e}")
            raise RuntimeError(f"LLM API error: {e}")
    
    async def generate_text_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        reasoning_effort: str = "medium",
        max_completion_tokens: int = 4000
    ) -> str:
        """Generate a plain text response from the LLM."""
        try:
            params = self._build_request_params(
                system_prompt, user_prompt, max_completion_tokens,
                temperature, reasoning_effort
            )
            
            if self.verbose:
                self._log_request_info(max_completion_tokens, reasoning_effort)
                
            response = await self.client.chat.completions.create(**params)
            
            choice = response.choices[0]
            content = choice.message.content
            
            if not content:
                self._handle_empty_response(choice, max_completion_tokens)
            
            if self.verbose:
                self._log_response_info(response, content)
            
            return content.strip()
            
        except Exception as e:
            if self.verbose:
                console.print(f"[red]❌ API Error:[/red] {type(e).__name__}: {e}")
            raise RuntimeError(f"LLM API error: {e}")