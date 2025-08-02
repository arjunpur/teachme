"""Integration test for the animate command."""

import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from teachme.agents.animation import ManimCodeGenerator
from teachme.models.schemas import ManimScriptResponse


@pytest.mark.asyncio
async def test_animation_generation_mock_llm():
    """Test animation generation with mocked LLM response."""
    
    # Mock LLM response
    mock_response = ManimScriptResponse(
        filename="scene.py",
        scene_name="TestCircle",
        description="A simple animated circle",
        code='''
from manim import *

class TestCircle(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        self.wait(1)
''',
        estimated_duration=2.0
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Create the animation generator
        with patch('teachme.agents.animation.LLMClient') as mock_llm_class:
            # Setup mock
            mock_llm_instance = AsyncMock()
            mock_llm_instance.generate_json_response.return_value = mock_response
            mock_llm_class.return_value = mock_llm_instance
            
            generator = ManimCodeGenerator(output_dir=output_dir)
            
            # Test input
            input_data = {
                "asset_prompt": "show a simple circle",
                "style": "light", 
                "quality": "low"
            }
            
            # Generate animation
            result = await generator.generate(input_data)
            
            # Verify result structure
            assert "video_path" in result
            assert "alt_text" in result
            assert "scene_name" in result
            assert "duration" in result
            
            # Verify values
            assert result["alt_text"] == "A simple animated circle"
            assert result["scene_name"] == "TestCircle"
            assert result["duration"] == 2.0
            
            # Verify video file exists
            video_path = Path(result["video_path"])
            assert video_path.exists()
            assert video_path.suffix == ".mp4"


def test_schemas():
    """Test Pydantic schema validation."""
    from teachme.models.schemas import AnimationInput, AnimationOutput, ManimScriptResponse
    
    # Test AnimationInput
    input_data = AnimationInput(
        asset_prompt="test prompt",
        style="dark",
        quality="high"
    )
    assert input_data.asset_prompt == "test prompt"
    assert input_data.style == "dark"
    assert input_data.quality == "high"
    
    # Test defaults
    input_data_defaults = AnimationInput(asset_prompt="test")
    assert input_data_defaults.style == "light"
    assert input_data_defaults.quality == "low"
    
    # Test AnimationOutput
    output_data = AnimationOutput(
        video_path="/path/to/video.mp4",
        alt_text="Description",
        scene_name="TestScene",
        duration=15.5
    )
    assert output_data.video_path == "/path/to/video.mp4"
    assert output_data.duration == 15.5
    
    # Test ManimScriptResponse
    script_response = ManimScriptResponse(
        scene_name="MyScene",
        description="Test description",
        code="from manim import *",
        estimated_duration=20.0
    )
    assert script_response.filename == "scene.py"  # Default value
    assert script_response.scene_name == "MyScene"