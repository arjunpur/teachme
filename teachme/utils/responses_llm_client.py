"""OpenAI Responses API client wrapper for LLM interactions."""

import os
from typing import Optional, Union, List, Dict, Any, Type, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
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
    
    def __init__(self, api_key: Optional[str] = None, model: str = None, verbose: bool = False, reasoning_effort: Optional[str] = None, stream_reasoning: Optional[bool] = None, reasoning_buffer_lines: Optional[int] = None):
        """Initialize the Responses LLM client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "OpenAI API key is required",
                config_key="OPENAI_API_KEY",
                expected_type="string"
            )
        
        self.model = model or os.getenv("TEACHME_MODEL", "gpt-4o")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.verbose = verbose
        # Default reasoning effort; caller can override per-call via kwargs['reasoning']
        env_effort = os.getenv("TEACHME_REASONING_EFFORT")
        configured_effort = reasoning_effort or env_effort or "medium"
        self.default_reasoning = self._normalize_reasoning_effort(configured_effort)
        # Streaming of model reasoning deltas (default ON). Controlled only via constructor.
        self.stream_reasoning = True if stream_reasoning is None else bool(stream_reasoning)
        # Buffer lines for reasoning window (default 15). Controlled only via constructor.
        self.reasoning_buffer_lines = 15 if reasoning_buffer_lines is None else int(reasoning_buffer_lines)

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
    
    # Note: environment variables no longer control streaming settings; use constructor args instead.
    
    async def _stream_text_response(
        self,
        params: Dict[str, Any],
        show_reasoning: bool,
        buffer_lines: int,
    ) -> Tuple[str, Any]:
        """Stream a text response while rendering reasoning in a bounded live panel.

        Returns:
            tuple(content_text, final_response)
        """
        # Prepare parameters (no 'stream' kw for stream() API)
        params = dict(params)
        params.pop("stream", None)

        collected_text_parts: List[str] = []
        reasoning_text: List[str] = []

        # Helper to compute the sliding window view
        def _sliding_text() -> str:
            full = "".join(reasoning_text)
            lines = full.splitlines() or [full]
            return "\n".join(lines[-buffer_lines:])

        final_response = None

        try:
            # Async streaming context; API surface may evolve, so be defensive
            stream_ctx = await self.client.responses.stream(**params)
            async with stream_ctx as stream:
                live: Optional[Live] = None
                if show_reasoning:
                    live = Live(
                        Panel("", title="Model reasoning", border_style="cyan"),
                        refresh_per_second=12,
                        console=console,
                        transient=True,
                    )
                    live.start()

                try:
                    async for event in stream:
                        # Event typing can vary; normalize access
                        event_type = getattr(event, "type", None) or getattr(event, "event", None) or ""
                        # Output text delta (actual answer)
                        if str(event_type).endswith("output_text.delta"):
                            delta = getattr(event, "delta", None) or getattr(event, "data", {}).get("delta", "")
                            if delta:
                                collected_text_parts.append(delta)
                        # Reasoning delta (we render this)
                        elif str(event_type).endswith("reasoning.delta"):
                            delta = getattr(event, "delta", None) or getattr(event, "data", {}).get("delta", "")
                            if delta:
                                reasoning_text.append(delta)
                                if live is not None:
                                    live.update(Panel(_sliding_text(), title="Model reasoning", border_style="cyan"))
                        # Capture completion and/or final response
                        elif str(event_type).endswith("completed"):
                            pass
                        elif str(event_type).endswith("error"):
                            # Let the outer handler raise
                            raise RuntimeError(getattr(event, "error", "Stream error"))

                finally:
                    if live is not None:
                        live.stop()

                # Try to fetch the final response object from the stream (SDK helper)
                try:
                    final_response = await stream.get_final_response()
                except Exception:
                    final_response = None

            content_text = ("".join(collected_text_parts)).strip()
            return content_text, final_response

        except Exception as stream_error:
            # Fallback to non-streaming call to avoid breaking functionality
            if self.verbose:
                console.print(f"[yellow]⚠️  Streaming failed, falling back: {stream_error}[/yellow]")
            params.pop("stream", None)
            response = await self.client.responses.create(**params)
            content_text = getattr(response, "output_text", "").strip()
            return content_text, response
    
    async def generate(
        self,
        input: Union[str, List[Dict[str, Any]]],
        instructions: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
        previous_response_id: Optional[str] = None,
        return_response_id: bool = False,
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
                # Structured output using responses.parse()
                messages = self._build_messages(input, instructions)
                params = self._build_params(messages, instructions, previous_response_id, **kwargs)
                params.update({
                    "input": messages,
                    "text_format": response_format
                })
                
                response = await self.client.responses.parse(**params)
                content = response.output_parsed
            else:
                # Text output using Responses API create(); extract via output_text
                params = self._build_params(input, instructions, previous_response_id, **kwargs)
                params["input"] = input
                if instructions:
                    params["instructions"] = instructions

                # If enabled, stream the response and render reasoning in a sliding window
                if self.stream_reasoning or bool(kwargs.get("stream", False)):
                    content_text, final_response = await self._stream_text_response(
                        params,
                        show_reasoning=True,
                        buffer_lines=self.reasoning_buffer_lines,
                    )
                    response = final_response
                    content = content_text
                else:
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
    
    # Backward compatibility methods
    async def generate_text_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        reasoning_effort: str = "medium",
        max_completion_tokens: int = 4000,
        previous_response_id: Optional[str] = None
    ) -> str:
        """Generate a plain text response (backward compatibility)."""
        return await self.generate(
            input=user_prompt,
            instructions=system_prompt,
            previous_response_id=previous_response_id,
            temperature=temperature,
            reasoning=self._normalize_reasoning_effort(reasoning_effort),
            max_completion_tokens=max_completion_tokens
        )
    
    async def generate_json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        temperature: float = 0.7,
        reasoning_effort: str = "medium",
        max_completion_tokens: int = 4000,
        previous_response_id: Optional[str] = None
    ) -> BaseModel:
        """Generate a structured JSON response (backward compatibility)."""
        return await self.generate(
            input=user_prompt,
            instructions=system_prompt,
            response_format=response_model,
            previous_response_id=previous_response_id,
            temperature=temperature,
            reasoning=self._normalize_reasoning_effort(reasoning_effort),
            max_completion_tokens=max_completion_tokens
        )