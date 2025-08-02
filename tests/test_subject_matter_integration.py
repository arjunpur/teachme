"""Integration tests for SubjectMatterAgent with ManimCodeGenerator."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from teachme.agents.subject_matter import SubjectMatterAgent
from teachme.agents.animation import ManimCodeGenerator
from teachme.models.schemas import ExpandedPrompt, AnimationStep, TextOverlay
from teachme.utils.llm_client import LLMClient


class TestSubjectMatterIntegration:
    """Integration tests for SubjectMatterAgent with ManimCodeGenerator."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock(spec=LLMClient)
        client.verbose = False
        return client
    
    @pytest.fixture
    def sample_expanded_prompt(self):
        """Create a sample ExpandedPrompt for testing."""
        return ExpandedPrompt(
            learning_objective="Understand how derivatives represent rates of change",
            key_concepts=["rate of change", "slope", "tangent line"],
            visual_strategy="Show a car moving along a curved path to illustrate changing speed",
            animation_sequence=[
                AnimationStep(
                    step_number=1,
                    visual_description="Show a car moving along a curved road",
                    explanation="Establish concrete example of changing motion",
                    key_insight="Speed varies along curved paths"
                ),
                AnimationStep(
                    step_number=2,
                    visual_description="Zoom in to show instantaneous speed at one point",
                    explanation="Focus on single moment in time",
                    key_insight="We can measure instantaneous rate of change"
                )
            ],
            explanatory_text=[
                TextOverlay(
                    text="The derivative measures instantaneous rate of change",
                    timing_description="after step 2",
                    purpose="Connect visual to mathematical concept"
                )
            ],
            quality_checklist=[
                "Ensure smooth car animation without visual artifacts",
                "Position text clearly without overlapping the road visualization",
                "Use color highlighting to emphasize the instantaneous moment"
            ]
        )
    
    @pytest.mark.asyncio
    async def test_subject_matter_to_animation_pipeline(self, tmp_path, mock_llm_client, sample_expanded_prompt):
        """Test the complete pipeline from SubjectMatterAgent to ManimCodeGenerator."""
        
        # Mock SubjectMatterAgent
        subject_matter_agent = SubjectMatterAgent(output_dir=tmp_path, llm_client=mock_llm_client)
        subject_matter_agent.process_with_timeout = AsyncMock(return_value=sample_expanded_prompt)
        
        # Mock ManimCodeGenerator 
        animation_generator = ManimCodeGenerator(output_dir=tmp_path, llm_client=mock_llm_client)
        
        # Mock the animation generation process
        mock_script_response = Mock()
        mock_script_response.description = "Animation showing derivatives as rates of change"
        mock_script_response.scene_name = "DerivativeScene"
        mock_script_response.estimated_duration = 25.0
        
        mock_video_path = tmp_path / "animations" / "DerivativeScene.mp4"
        mock_video_path.parent.mkdir(parents=True, exist_ok=True)
        mock_video_path.touch()  # Create empty file
        
        animation_generator._generate_and_render_with_expanded_prompt = AsyncMock(
            return_value=(mock_script_response, mock_video_path)
        )
        
        # Test the complete pipeline
        # 1. SubjectMatterAgent generates ExpandedPrompt
        expanded_prompt = await subject_matter_agent.process_with_timeout("explain derivatives")
        assert isinstance(expanded_prompt, ExpandedPrompt)
        assert expanded_prompt.learning_objective == "Understand how derivatives represent rates of change"
        
        # 2. ManimCodeGenerator uses ExpandedPrompt
        input_data = {
            "expanded_prompt": expanded_prompt.model_dump(),
            "style": "light",
            "quality": "low"
        }
        
        result = await animation_generator.generate(input_data)
        
        # Verify the result
        assert "video_path" in result
        assert "alt_text" in result
        assert "scene_name" in result
        assert "duration" in result
        
        assert result["alt_text"] == "Animation showing derivatives as rates of change"
        assert result["scene_name"] == "DerivativeScene"
        assert result["duration"] == 25.0
    
    @pytest.mark.asyncio
    async def test_enhanced_animation_input_validation(self, tmp_path, mock_llm_client, sample_expanded_prompt):
        """Test EnhancedAnimationInput validation logic."""
        from teachme.models.schemas import EnhancedAnimationInput
        
        # Test with expanded_prompt
        enhanced_input = EnhancedAnimationInput(
            expanded_prompt=sample_expanded_prompt,
            style="dark",
            quality="medium"
        )
        
        assert enhanced_input.expanded_prompt == sample_expanded_prompt
        assert enhanced_input.asset_prompt is None
        assert enhanced_input.style == "dark"
        assert enhanced_input.quality == "medium"
        
        # Test with asset_prompt
        direct_input = EnhancedAnimationInput(
            asset_prompt="explain derivatives",
            style="light",
            quality="low"
        )
        
        assert direct_input.asset_prompt == "explain derivatives"
        assert direct_input.expanded_prompt is None
        
        # Test validation errors
        with pytest.raises(ValueError, match="Either asset_prompt or expanded_prompt must be provided"):
            EnhancedAnimationInput(style="light", quality="low")
        
        with pytest.raises(ValueError, match="Only one of asset_prompt or expanded_prompt should be provided"):
            EnhancedAnimationInput(
                asset_prompt="test",
                expanded_prompt=sample_expanded_prompt,
                style="light",
                quality="low"
            )
    
    @pytest.mark.asyncio
    async def test_animation_generator_fallback_compatibility(self, tmp_path, mock_llm_client):
        """Test that ManimCodeGenerator still works with legacy AnimationInput."""
        animation_generator = ManimCodeGenerator(output_dir=tmp_path, llm_client=mock_llm_client)
        
        # Mock the legacy generation process
        mock_script_response = Mock()
        mock_script_response.description = "Simple animation"
        mock_script_response.scene_name = "SimpleScene"
        mock_script_response.estimated_duration = 20.0
        
        mock_video_path = tmp_path / "animations" / "SimpleScene.mp4"
        mock_video_path.parent.mkdir(parents=True, exist_ok=True)
        mock_video_path.touch()
        
        animation_generator._generate_and_render_with_retry = AsyncMock(
            return_value=(mock_script_response, mock_video_path)
        )
        
        # Test legacy input format
        legacy_input = {
            "asset_prompt": "show a circle",
            "style": "light",
            "quality": "low"
        }
        
        result = await animation_generator.generate(legacy_input)
        
        # Verify it works
        assert "video_path" in result
        assert result["alt_text"] == "Simple animation"
        assert result["scene_name"] == "SimpleScene"
        
        # Verify the correct method was called
        animation_generator._generate_and_render_with_retry.assert_called_once_with(
            "show a circle", "light", "low"
        )
    
    @pytest.mark.asyncio
    async def test_enhanced_prompt_formatting(self, sample_expanded_prompt):
        """Test that enhanced prompts are formatted correctly."""
        from teachme.prompts.animation import create_enhanced_animation_user_prompt
        
        prompt = create_enhanced_animation_user_prompt(sample_expanded_prompt, "light")
        
        # Check that key elements are included
        assert "OBJECTIVE: Understand how derivatives represent rates of change" in prompt
        assert "CONCEPTS: rate of change, slope, tangent line" in prompt
        assert "VISUAL STRATEGY: Show a car moving along a curved path" in prompt
        assert "Step 1: Show a car moving along a curved road" in prompt
        assert "Step 2: Zoom in to show instantaneous speed" in prompt
        assert '"The derivative measures instantaneous rate of change"' in prompt
        assert "Ensure smooth car animation without visual artifacts" in prompt
        assert "Position text clearly without overlapping" in prompt
        
        # Check critical requirements
        assert "CRITICAL: Ensure no text overlaps with visual elements" in prompt
        assert "Use appropriate timing so text appears synchronized" in prompt
        
    def test_prompt_formatting_edge_cases(self):
        """Test prompt formatting with edge cases."""
        from teachme.prompts.animation import create_enhanced_animation_user_prompt
        
        # Test with minimal data
        minimal_prompt = ExpandedPrompt(
            learning_objective="Simple concept",
            key_concepts=["concept1"],
            visual_strategy="Basic visualization",
            animation_sequence=[],
            explanatory_text=[],
            quality_checklist=[]
        )
        
        prompt = create_enhanced_animation_user_prompt(minimal_prompt, "dark")
        
        assert "OBJECTIVE: Simple concept" in prompt
        assert "CONCEPTS: concept1" in prompt
        assert "VISUAL STRATEGY: Basic visualization" in prompt
        assert "dark background with light text" in prompt