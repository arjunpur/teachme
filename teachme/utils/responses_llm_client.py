"""OpenAI Responses API client wrapper for LLM interactions."""

import os
from typing import Optional, Union, List, Dict, Any, Type, Callable
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from rich.console import Console
from ..exceptions import LLMGenerationError, ConfigurationError

# Load environment variables from .env file
load_dotenv()

console = Console()

# Supported parameters for both create() and parse() methods
SUPPORTED_PARAMS = {
    "background", "include", "max_output_tokens", "max_tool_calls", 
    "metadata", "parallel_tool_calls", "prompt", "reasoning", 
    "service_tier", "temperature", "text", "tool_choice", 
    "tools", "top_logprobs", "top_p", "truncation", "user"
}


@dataclass
class ResponseResult:
    """Result from LLM generation with response ID for chaining."""
    content: Union[str, BaseModel]
    response_id: str
    usage: Optional[Dict[str, Any]] = None


class ResponsesLLMClient:
    """Single entrypoint LLM client using OpenAI Responses API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = None, verbose: bool = False, reasoning_effort: Optional[str] = None):
        """Initialize the Responses LLM client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "OpenAI API key is required",
                config_key="OPENAI_API_KEY",
                expected_type="string"
            )
        
        # Respect explicit arg, then env, then config default
        try:
            from ..config import LLMConfig
            config_default_model = getattr(LLMConfig, "DEFAULT_MODEL", None)
        except Exception:
            config_default_model = None
        self.model = model or os.getenv("TEACHME_MODEL", config_default_model or "gpt-4o")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.verbose = verbose
        # Default reasoning effort; caller can override per-call via kwargs['reasoning']
        env_effort = os.getenv("TEACHME_REASONING_EFFORT")
        configured_effort = reasoning_effort or env_effort or "medium"
        self.default_reasoning = self._normalize_reasoning_effort(configured_effort)
        # No streaming state; responses are retrieved after completion

    def _normalize_reasoning_effort(self, effort_value: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return a valid reasoning dict for the API or None if disabled."""
        if not effort_value:
            return None
        effort_norm = str(effort_value).strip().lower()
        if effort_norm not in {"low", "medium", "high"}:
            effort_norm = "medium"
        return {"effort": effort_norm}
    
    def _build_messages(self, input: Union[str, List[Dict[str, Any]]], instructions: Optional[str]) -> List[Dict[str, Any]]:
        """Convert input and instructions to messages format."""
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        
        if isinstance(input, str):
            messages.append({"role": "user", "content": input})
        elif isinstance(input, list):
            messages.extend(input)
        
        return messages
    
    def _build_params(self, input: Union[str, List[Dict[str, Any]]], instructions: Optional[str], 
                     previous_response_id: Optional[str], **kwargs) -> Dict[str, Any]:
        """Build common parameters for API calls."""
        params = {"model": self.model}
        
        # Handle chaining
        if previous_response_id:
            params["previous_response_id"] = previous_response_id
            params["store"] = True
        
        # Add supported parameters and convert legacy ones
        for key, value in kwargs.items():
            if key in SUPPORTED_PARAMS:
                params[key] = value
            elif key == "max_completion_tokens":
                params["max_output_tokens"] = value

        # Inject default reasoning if not provided by caller
        if "reasoning" not in params and self.default_reasoning is not None:
            params["reasoning"] = self.default_reasoning
        
        # Normalize/strip params that are unsupported for specific models
        return self._normalize_model_params(params)

    def _normalize_model_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or transform parameters not supported by the selected model.

        Currently, GPT-5 models do not support sampling parameters like 'temperature'.
        """
        model_name = str(params.get("model", self.model) or "")
        if model_name.startswith("gpt-5"):
            # Remove unsupported sampling parameters
            params.pop("temperature", None)
            # Some GPT-5 variants may also not support top_p/top_logprobs; remove defensively
            params.pop("top_p", None)
            params.pop("top_logprobs", None)
        return params
    
    def _log_request(self, input_type: str, instructions: Optional[str], 
                    response_format: Optional[Type[BaseModel]], previous_response_id: Optional[str]) -> None:
        """Log request information if verbose mode is enabled."""
        if not self.verbose:
            return
            
        console.print(f"[dim]🤖 Using model:[/dim] [bold blue]{self.model}[/bold blue]")
        console.print(f"[dim]📝 Input type:[/dim] {input_type}")
        if instructions:
            console.print(f"[dim]📋 Instructions:[/dim] {instructions[:100]}...")
        if previous_response_id:
            console.print(f"[dim]🔗 Chaining from:[/dim] {previous_response_id[:8]}...")
        if response_format:
            console.print(f"[dim]🏗️ Structured output:[/dim] {response_format.__name__}")
    
    def _log_response(self, response, content: Union[str, BaseModel]) -> None:
        """Log response information if verbose mode is enabled."""
        if not self.verbose:
            return
        
        if response is not None:
            console.print(f"[dim]🆔 Response ID:[/dim] {response.id}")
        if response is not None and hasattr(response, 'usage') and response.usage:
            usage = response.usage
            console.print(f"[dim]📊 Token usage:[/dim] {usage.input_tokens} input + {usage.output_tokens} output")
        
        if isinstance(content, str):
            console.print(f"[dim]✅ Response length:[/dim] {len(content)} characters")
        else:
            console.print(f"[dim]✅ Structured response:[/dim] {type(content).__name__}")
    
    def _create_usage_dict(self, response) -> Optional[Dict[str, Any]]:
        """Extract usage information from response."""
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            return {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": getattr(usage, 'total_tokens', usage.input_tokens + usage.output_tokens)
            }
        return None
    
    async def generate(
        self,
        input: Union[str, List[Dict[str, Any]]],
        instructions: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
        previous_response_id: Optional[str] = None,
        return_response_id: bool = False,
        stream_reasoning: bool = False,
        on_reasoning_token: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> Union[str, BaseModel, ResponseResult]:
        """
        Single entrypoint for all LLM interactions using Responses API.
        
        Args:
            input: String prompt, messages array, or multimodal content
            instructions: System-level instructions (simpler than system messages)
            response_format: Concrete Pydantic model class for structured output
            previous_response_id: ID from previous response for conversation chaining
            return_response_id: If True, return ResponseResult with response ID for chaining
            **kwargs: Additional parameters (temperature, max_output_tokens, etc.)
        
        Returns:
            String for text responses, Pydantic model instance for structured responses,
            or ResponseResult object with response data and ID for chaining
        """
        try:
            input_type = "messages" if response_format else type(input).__name__
            self._log_request(input_type, instructions, response_format, previous_response_id)
            
            if response_format:
                # Structured output using responses.parse(); optionally stream reasoning tokens
                messages = self._build_messages(input, instructions)
                params = self._build_params(messages, instructions, previous_response_id, **kwargs)
                params.update({
                    "input": messages,
                    "text_format": response_format
                })

                # Attempt streaming to surface reasoning deltas; fallback to non-streaming
                if stream_reasoning and on_reasoning_token is not None:
                    try:
                        async with self.client.responses.stream(**params) as stream:
                            async for event in stream:
                                try:
                                    event_type = getattr(event, "type", "") or ""
                                    # New SDK surfaces reasoning events via specific types
                                    if event_type in (
                                        "response.reasoning.delta",
                                        "response.reasoning.summary.delta",
                                    ):
                                        delta = getattr(event, "delta", None)
                                        if isinstance(delta, str) and delta:
                                            on_reasoning_token(delta)
                                    elif event_type == "response.output_text.delta":
                                        # As a fallback, also surface normal text tokens if desired
                                        delta = getattr(event, "delta", None)
                                        if isinstance(delta, str) and delta:
                                            on_reasoning_token(delta)
                                except Exception:
                                    # Do not break stream on callback/inspection errors
                                    pass
                            response = await stream.get_final_response()
                            content = response.output_parsed
                    except Exception:
                        response = await self.client.responses.parse(**params)
                        content = response.output_parsed
                else:
                    response = await self.client.responses.parse(**params)
                    content = response.output_parsed
            else:
                # Text output using Responses API create(); extract via output_text
                params = self._build_params(input, instructions, previous_response_id, **kwargs)
                params["input"] = input
                if instructions:
                    params["instructions"] = instructions
                response = await self.client.responses.create(**params)

                # Responses API returns a Response object with `output` items and convenience `output_text`
                content = getattr(response, "output_text", "")
                if not content:
                    # Fallback: concatenate any message/output_text items if present
                    try:
                        content = "".join(
                            [
                                block.text
                                for item in getattr(response, "output", [])
                                if getattr(item, "type", "") == "message"
                                for block in getattr(item, "content", [])
                                if getattr(block, "type", "") == "output_text"
                            ]
                        )
                    except Exception:
                        content = ""

                if not content:
                    raise LLMGenerationError("Empty response from LLM", model=self.model)
                content = content.strip()
            
            self._log_response(response, content)
            
            # Return with response ID if requested or chaining
            if return_response_id or previous_response_id:
                return ResponseResult(
                    content=content,
                    response_id=getattr(response, "id", ""),
                    usage=self._create_usage_dict(response)
                )
            else:
                return content
                
        except Exception as e:
            if self.verbose:
                console.print(f"[red]❌ API Error:[/red] {type(e).__name__}: {e}")
            raise LLMGenerationError(f"Responses API error: {e}", model=self.model) from e

    # Public convenience wrappers
    async def generate_structured(
        self,
        input: Union[str, List[Dict[str, Any]]],
        instructions: Optional[str],
        response_format: Type[BaseModel],
        previous_response_id: Optional[str] = None,
        return_response_id: bool = False,
        **kwargs,
    ) -> Union[BaseModel, ResponseResult]:
        """Generate a structured response parsed into the provided Pydantic model.

        This is a clearer entrypoint alias for generate(..., response_format=...).
        """
        return await self.generate(
            input=input,
            instructions=instructions,
            response_format=response_format,
            previous_response_id=previous_response_id,
            return_response_id=return_response_id,
            **kwargs,
        )

    async def generate_text(
        self,
        input: Union[str, List[Dict[str, Any]]],
        instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate plain text. Wrapper for generate() without response_format."""
        result = await self.generate(
            input=input,
            instructions=instructions,
            previous_response_id=previous_response_id,
            **kwargs,
        )
        # Base generate returns str when no response_format
        return result  # type: ignore[return-value]
    
    # Backward compatibility methods
    # (Removed legacy generate_text_response/generate_json_response)