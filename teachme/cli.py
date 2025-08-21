"""Typer CLI commands for teachme."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables from .env file
load_dotenv()

from .agents.animation import ManimCodeGenerator
from .utils.responses_llm_client import ResponsesLLMClient

app = typer.Typer(help="TeachMe - Convert natural language prompts into educational content with animations")
console = Console()


@app.command()
def animate(
    prompt: str = typer.Argument(..., help="Natural language description of what to animate"),
    style: str = typer.Option("light", help="Visual style (light/dark)"),
    output_dir: Path = typer.Option("./outputs", help="Output directory for video"),
    quality: str = typer.Option("low", help="Video quality (low/medium/high)"),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed progress"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenAI API key (overrides environment variable)"),
    skip_subject_matter: bool = typer.Option(False, "--skip-subject-matter", help="Skip SubjectMatterAgent and use direct prompt (legacy mode)")
) -> None:
    """Generate a Manim animation from a natural language prompt."""
    
    async def _animate():
        try:
            # Initialize components
            if verbose:
                console.print("[blue]Initializing animation system...[/blue]")
            
            llm_client = ResponsesLLMClient(api_key=api_key, verbose=verbose)
            if verbose:
                console.print("[dim]Streaming model reasoning...[/dim]")
            animation_generator = ManimCodeGenerator(output_dir=output_dir, llm_client=llm_client, verbose=verbose)
            
            # Generate animation with progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=False
            ) as progress:
                
                if verbose:
                    console.print(f"[blue]Prompt:[/blue] {prompt}")
                    console.print(f"[blue]Style:[/blue] {style}, [blue]Quality:[/blue] {quality}")
                    console.print("[blue]Mode:[/blue] Enhanced" if not skip_subject_matter else "[blue]Mode:[/blue] Direct")
                
                try:
                    # Single unified path: delegate subject matter handling to animation agent
                    task = progress.add_task(
                        "Generating enhanced Manim script..." if not skip_subject_matter else "Generating Manim script...",
                        total=None
                    )
                    input_data = {"user_prompt": prompt, "style": style, "use_subject_matter": (not skip_subject_matter)}
                    result = await animation_generator.generate_animation(input_data)
                    progress.update(task, description="✓ Script generated and rendered")
                    
                except Exception as e:
                    progress.update(progress.tasks[0].id if progress.tasks else 0, description=f"✗ Failed: {str(e)}")
                    raise
            
            # Report success
            video_path = result["video_path"]
            duration = result["duration"]
            alt_text = result["alt_text"]
            
            console.print(f"\n[green]✓ Animation generated successfully![/green]")
            console.print(f"[green]Video path:[/green] {video_path}")
            console.print(f"[green]Duration:[/green] {duration:.1f} seconds")
            console.print(f"[green]Description:[/green] {alt_text}")
            
            # Show script path if available
            if "script_path" in result:
                console.print(f"[green]Script saved:[/green] {result['script_path']}")
            
            if verbose:
                console.print(f"[blue]Scene name:[/blue] {result['scene_name']}")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Animation generation cancelled by user.[/yellow]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"\n[red]✗ Failed to generate animation: {str(e)}[/red]")
            if "OpenAI API key" in str(e):
                console.print("[yellow]  Try: Set OPENAI_API_KEY environment variable or use --api-key[/yellow]")
            elif "Manim" in str(e):
                console.print("[yellow]  Check: Is Manim installed? Run 'manim --version'[/yellow]")
            else:
                console.print("[yellow]  Try: teachme animate --verbose \"simpler prompt\"[/yellow]")
            raise typer.Exit(1)
    
    # Run the async function
    asyncio.run(_animate())


@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"teachme version {__version__}")


if __name__ == "__main__":
    app()