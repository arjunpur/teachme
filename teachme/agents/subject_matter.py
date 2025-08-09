"""SubjectMatterAgent implementation (single-step brief generator)."""

import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from rich.console import Console
from ..agents.base import BaseAgent
from ..models.schemas import SubjectMatterInput
from ..utils.responses_llm_client import ResponsesLLMClient
from ..config import LLMConfig
from ..exceptions import SubjectMatterAnalysisError
from ..prompts.subject_matter import (
    SINGLE_EXPANSION_SYSTEM_PROMPT,
    create_single_expansion_prompt,
)

console = Console()

class SubjectMatterAgent(BaseAgent):
    """Agent that expands a user's idea into a structured written brief for Manim."""

    def __init__(self, output_dir: Path = None, llm_client: ResponsesLLMClient = None, verbose: bool = False):
        super().__init__(output_dir)
        self.llm_client = llm_client or ResponsesLLMClient()
        self.verbose = verbose

    def _is_verbose(self) -> bool:
        return self.verbose

    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single expanded brief (plain text) for the Manim animator."""
        subject_matter_input = SubjectMatterInput(**input_data)

        try:
            if self._is_verbose():
                console.print("[blue]🧠 Expanding subject matter into a structured brief...[/blue]")

            result = await self.llm_client.generate(
                input=create_single_expansion_prompt(subject_matter_input.user_prompt),
                instructions=SINGLE_EXPANSION_SYSTEM_PROMPT,
                previous_response_id=None,
                return_response_id=True,
                temperature=LLMConfig.CONTENT_ANALYSIS_TEMPERATURE,
                max_completion_tokens=LLMConfig.DEFAULT_MAX_TOKENS,
            )

            brief_text = result.content

            if not isinstance(brief_text, str) or len(brief_text.strip()) == 0:
                raise ValueError("Empty brief returned from LLM")

            if self._is_verbose():
                console.print("[green]✓ Subject matter brief generated. Preview:[/green]")
                # Print a bounded preview to keep logs readable
                preview = brief_text if len(brief_text) <= 2000 else brief_text[:2000] + "\n..."
                console.print(preview)

            return {"expanded_prompt_text": brief_text, "_response_id": result.response_id}

        except Exception as e:
            raise SubjectMatterAnalysisError(
                f"Subject matter analysis failed: {e}",
                user_prompt=subject_matter_input.user_prompt,
            ) from e

    async def process_with_timeout(self, user_prompt: str, timeout_seconds: int = 90) -> str:
        """Process user prompt with timeout, returning the expanded brief text."""
        try:
            output = await asyncio.wait_for(
                self.generate({"user_prompt": user_prompt}),
                timeout=timeout_seconds,
            )
            return output["expanded_prompt_text"]
        except asyncio.TimeoutError:
            raise SubjectMatterAnalysisError(
                f"Subject matter processing timed out after {timeout_seconds} seconds",
                user_prompt=user_prompt,
            )
        except Exception as e:
            raise SubjectMatterAnalysisError(
                f"Subject matter processing failed: {e}",
                user_prompt=user_prompt,
            ) from e